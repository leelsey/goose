// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package openai talks to the OpenAI API and the backends that speak it.
package openai

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

const defaultBaseURL = "https://api.openai.com/v1"

type Dialect struct {
	StreamUsage     bool
	Extensions      bool
	PerModelCeiling bool
	Responses       bool
	Connectors      bool
}

func OpenAIDialect() Dialect {
	return Dialect{StreamUsage: true, Extensions: true, PerModelCeiling: true, Responses: true, Connectors: true}
}

func CompatibleDialect() Dialect { return Dialect{} }

type Options struct {
	ID             string
	BaseURL        string
	APIKey         string
	Model          string
	Organisation   string
	Project        string
	Dialect        Dialect
	MaxRetries     int
	Timeout        time.Duration
	AllowClientKey bool
	KeyScheme      llm.CredentialScheme
	Header         http.Header
	AuthHeader     string
	AuthPrefix     string

	ChatPath      string
	ResponsesPath string
	Query         string
	Sign          func(ctx context.Context, r *http.Request, body []byte) error
	Body          map[string]any
}

type Provider struct {
	opts Options
	env  llm.Env
	call *httpcore.Caller
}

func New(o Options) (*Provider, error) {
	if o.ID == "" {
		o.ID = "openai"
	}
	if o.BaseURL == "" {
		o.BaseURL = defaultBaseURL
		o.Dialect = OpenAIDialect()
	}
	p := &Provider{opts: o}
	p.call = &httpcore.Caller{Provider: o.ID, MaxRetries: o.MaxRetries, HTTP: httpcore.NewClient(o.Timeout)}
	return p, nil
}

func (p *Provider) ID() string { return p.opts.ID }

func (p *Provider) Bind(env llm.Env) error {
	p.env = env
	if env.HTTP != nil {
		p.call.HTTP = env.HTTP
	}
	return nil
}

func (p *Provider) Capabilities() llm.Capabilities {
	return llm.Capabilities{
		KeyScheme: p.opts.KeyScheme,
		Streaming: true,
		Usage: llm.UsageShape{
			StreamUsage:      p.opts.Dialect.StreamUsage,
			StreamUsageOptIn: true,
		},
		SystemPrompt:       true,
		Tools:              true,
		ParallelTools:      true,
		ToolChoice:         true,
		ToolResults:        true,
		ImageIn:            true,
		Thinking:           p.opts.Dialect.Responses,
		ThinkingSignatures: p.opts.Dialect.Responses,
		Effort:             true,
		ResponseFormat:     true,
		Sessions:           p.opts.Dialect.Responses,
		MCPServers:         p.connectors(),
		Extensions:         p.wires(),
		Candidates:         128,
		Ceiling:            p.ceiling(),
	}
}

func (p *Provider) connectors() bool {
	return p.opts.Dialect.Responses && p.opts.Dialect.Connectors
}

func (p *Provider) useResponses(a *llm.Attempt) bool {
	if !p.opts.Dialect.Responses {
		return false
	}
	if a.Format == "openai-responses" {
		return true
	}
	if p.connectors() && a.Request != nil && len(a.Request.MCPServers) > 0 {
		return true
	}
	return imageOnlyModel(p.model(a))
}

func imageOnlyModel(model string) bool {
	m := strings.ToLower(lastSegment(model))
	return strings.HasPrefix(m, "gpt-image") || strings.HasPrefix(m, "dall-e")
}

func lastSegment(model string) string {
	if i := strings.LastIndexAny(model, "/:"); i >= 0 {
		return model[i+1:]
	}
	return model
}

func (p *Provider) model(a *llm.Attempt) string {
	switch {
	case a.Model != "":
		return a.Model
	case p.opts.Model != "":
		return p.opts.Model
	case a.Request != nil:
		return a.Request.Model
	default:
		return ""
	}
}

func (p *Provider) key(a *llm.Attempt) string {
	if p.opts.APIKey != "" {
		return p.opts.APIKey
	}
	if p.opts.AllowClientKey && a != nil {
		return a.Credential.Key
	}
	return ""
}

func (p *Provider) endpoint(path, override string) string {
	if override != "" {
		path = override
	}
	u := httpcore.JoinURL(p.opts.BaseURL, path)
	if p.opts.Query != "" {
		u += "?" + p.opts.Query
	}
	return u
}

func (p *Provider) header(a *llm.Attempt) http.Header {
	h := http.Header{}
	for k, vs := range p.opts.Header {
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	if key := p.key(a); key != "" && p.opts.Sign == nil {
		name, prefix := p.opts.AuthHeader, p.opts.AuthPrefix
		if name == "" {
			name, prefix = "Authorization", "Bearer "
		}
		h.Set(name, prefix+key)
	}
	if p.opts.Organisation != "" {
		h.Set("OpenAI-Organization", p.opts.Organisation)
	}
	if p.opts.Project != "" {
		h.Set("OpenAI-Project", p.opts.Project)
	}
	if a != nil && p.opts.AllowClientKey {
		for k, v := range a.Credential.Headers {
			h.Set(k, v)
		}
	}
	return h
}

func created(unix int64, fallback time.Time) time.Time {
	if unix > 0 {
		return time.Unix(unix, 0).UTC()
	}
	return fallback
}

func (p *Provider) now() time.Time {
	if p.env.Now != nil {
		return p.env.Now()
	}
	return time.Now()
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	if err := p.guardImageOnly(a); err != nil {
		return nil, err
	}
	if p.useResponses(a) {
		return p.completeResponses(ctx, a)
	}
	return p.completeChat(ctx, a)
}

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	if err := p.guardImageOnly(a); err != nil {
		return nil, err
	}
	if p.useResponses(a) {
		return p.streamResponses(ctx, a)
	}
	return p.streamChat(ctx, a)
}

func (p *Provider) guardImageOnly(a *llm.Attempt) error {
	if !imageOnlyModel(p.model(a)) || !a.Renders.Declared() || a.Renders.Images {
		return nil
	}
	d := llm.Diagnostic{
		Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StagePlan,
		Severity: llm.SeverityFatal, Provider: p.ID(), Model: p.model(a), Format: a.Format,
		Detail: "this model answers with an image and this format has no field for one; call /v1/responses instead",
	}
	a.Note(d)
	return &llm.Error{
		Kind: llm.KindUnsupported, Code: llm.DropImage, Provider: p.ID(), Model: p.model(a),
		Message:     "this model answers with an image, which this format cannot carry; call /v1/responses instead",
		Sent:        false,
		Diagnostics: []llm.Diagnostic{d},
	}
}

func (p *Provider) passKey(r *llm.PassRequest) string {
	if p.opts.APIKey != "" {
		return p.opts.APIKey
	}
	if p.opts.AllowClientKey {
		return r.Key
	}
	return ""
}

func (p *Provider) Pass(ctx context.Context, r *llm.PassRequest) (*llm.PassResponse, error) {
	url := httpcore.JoinURL(p.opts.BaseURL, r.Path)
	if r.Query != "" {
		url += "?" + r.Query
	}
	req, err := http.NewRequestWithContext(ctx, r.Method, url, r.Body)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: p.ID(), Message: err.Error()}
	}
	httpcore.ForwardPassthrough(req.Header, r.Header)
	if key := p.passKey(r); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := p.call.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindTransport, Provider: p.ID(), Message: err.Error(), Sent: true, Err: err}
	}
	return &llm.PassResponse{Status: resp.StatusCode, Header: resp.Header, Body: resp.Body}, nil
}

func (p *Provider) ceiling() llm.CeilingField {
	if p.opts.Dialect.PerModelCeiling {
		return llm.CeilingPerModel
	}
	return llm.CeilingMaxTokens
}

func (p *Provider) wires() []string {
	if !p.opts.Dialect.Extensions {
		return nil
	}
	if p.opts.Dialect.Responses {
		return []string{"openai", "openai-responses"}
	}
	return []string{"openai"}
}
