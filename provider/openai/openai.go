// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package openai talks to the OpenAI API and the backends that speak it.
package openai

import (
	"context"
	"net/http"
	neturl "net/url"
	"strings"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

const defaultBaseURL = "https://api.openai.com/v1"

// Dialect is what a backend understands beyond the shape every OpenAI-speaking
// backend shares. Left zero in Options it is the OpenAI dialect for OpenAI's
// own endpoint, named or not, and the compatible one for any other.
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
	// Sign authenticates a request the backend will not take a static key for.
	// Passthrough forwards a body it never reads, so it signs a nil one: a
	// signer that covers the body cannot be used on that path.
	Sign func(ctx context.Context, r *http.Request, body []byte) error
	Body map[string]any
	// ResponsesBody is Body for the Responses shape alone, for a field that
	// belongs there and nowhere else.
	ResponsesBody map[string]any
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
	} else if o.Dialect == (Dialect{}) && official(o.BaseURL) {
		o.Dialect = OpenAIDialect()
	}
	p := &Provider{opts: o}
	p.call = &httpcore.Caller{Provider: o.ID, MaxRetries: o.MaxRetries, HTTP: httpcore.NewClient(o.Timeout), Refine: refine}
	return p, nil
}

// refine reads what the status does not say. OpenAI refuses to verify an
// organisation for one model or feature at a time — streaming o3, say, or
// gpt-image-1 — and serves every other model on the same key, so the refusal
// is the model's, not the account's: it must not open the shared breaker for
// every model behind this key, and another model, or the same one once
// verified, may still be worth trying.
func refine(e *llm.Error) {
	if e.Kind == llm.KindInvalidRequest && strings.Contains(strings.ToLower(e.Message), "organization must be verified") {
		e.Kind = llm.KindUnsupported
	}
}

// official says whether a base is OpenAI's own endpoint, a regional one
// included, which speaks the OpenAI dialect however it came to be named.
func official(base string) bool {
	u, err := neturl.Parse(base)
	if err != nil || !strings.EqualFold(u.Scheme, "https") {
		return false
	}
	if host := strings.ToLower(u.Hostname()); host != "api.openai.com" && !strings.HasSuffix(host, ".api.openai.com") {
		return false
	}
	if port := u.Port(); port != "" && port != "443" {
		return false
	}
	switch strings.TrimSuffix(u.Path, "/") {
	case "", "/v1":
		return true
	}
	return false
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
		ClientKey: p.opts.AllowClientKey && p.opts.APIKey == "" && p.opts.Sign == nil,
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
		Signer:             llm.FamilyOpenAI,
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
	p.authorise(h, p.key(a))
	p.bill(h)
	if a != nil && p.opts.AllowClientKey {
		for k, v := range a.Credential.Headers {
			h.Set(k, v)
		}
	}
	return h
}

// bill names the organisation and project the operator configured.
func (p *Provider) bill(h http.Header) {
	if p.opts.Organisation != "" {
		h.Set("OpenAI-Organization", p.opts.Organisation)
	}
	if p.opts.Project != "" {
		h.Set("OpenAI-Project", p.opts.Project)
	}
}

// authorise names the credential the way this backend expects it. A signed
// backend mints its own, so the static key stays off the wire.
func (p *Provider) authorise(h http.Header, key string) {
	if key == "" || p.opts.Sign != nil {
		return
	}
	name, prefix := p.opts.AuthHeader, p.opts.AuthPrefix
	if name == "" {
		name, prefix = "Authorization", "Bearer "
	}
	h.Set(name, prefix+key)
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
	u := passURL(p.opts.BaseURL, r.Path)
	q, err := httpcore.PassQuery(p.ID(), r.Query)
	if err != nil {
		return nil, err
	}
	q, err = mergeQuery(p.opts.Query, q)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: p.ID(),
			Message: "the query string could not be read: " + err.Error(), Err: err}
	}
	if q != "" {
		u += "?" + q
	}
	req, rerr := http.NewRequestWithContext(ctx, r.Method, u, r.Body)
	if rerr != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: p.ID(), Message: rerr.Error()}
	}
	httpcore.ForwardPassthrough(req.Header, r.Header)
	httpcore.Overlay(req.Header, p.opts.Header)
	key := p.passKey(r)
	p.authorise(req.Header, key)
	p.bill(req.Header)
	if key != "" && key == r.Key && p.opts.APIKey == "" && p.opts.Sign == nil {
		// The caller's own key bills the caller, whose organisation and
		// project these are to name.
		for _, name := range []string{"OpenAI-Organization", "OpenAI-Project"} {
			if vs := r.Header.Values(name); len(vs) > 0 {
				req.Header.Del(name)
				for _, v := range vs {
					req.Header.Add(name, v)
				}
			}
		}
	}
	if p.opts.Sign != nil {
		// The body is a stream here, so a signer that covers it cannot be used
		// for passthrough; the token minters this reaches do not read it.
		if err := p.opts.Sign(ctx, req, nil); err != nil {
			return nil, &llm.Error{Kind: llm.KindAuth, Provider: p.ID(), Message: err.Error(), Err: err}
		}
	}
	client := p.call.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, httpcore.ReadError(ctx, p.ID(), err)
	}
	return &llm.PassResponse{Status: resp.StatusCode, Header: resp.Header, Body: resp.Body}, nil
}

// passURL addresses a passthrough call. An OpenAI-shaped base already ends in
// its version, and a client puts that version in front of every path it asks
// for, so what the base already says is not said twice.
func passURL(base, path string) string {
	u, err := neturl.Parse(base)
	if err != nil {
		return httpcore.JoinURL(base, path)
	}
	want := strings.TrimLeft(path, "/")
	segs := strings.Split(strings.Trim(u.Path, "/"), "/")
	for i := range segs {
		tail := strings.Join(segs[i:], "/")
		if tail == "" {
			break
		}
		if want == tail || strings.HasPrefix(want, tail+"/") {
			want = strings.TrimPrefix(want, tail)
			break
		}
	}
	return httpcore.JoinURL(base, want)
}

// mergeQuery keeps every parameter the caller sent and adds the configured ones
// it left out, so a pinned api-version reaches the backend without overruling a
// caller that named its own. A query that cannot be read is refused rather than
// guessed at: appending to it would send the same parameter twice.
func mergeQuery(configured, caller string) (string, error) {
	if configured == "" {
		return caller, nil
	}
	if caller == "" {
		return configured, nil
	}
	have, err := neturl.ParseQuery(caller)
	if err != nil {
		return "", err
	}
	add, err := neturl.ParseQuery(configured)
	if err != nil {
		return "", err
	}
	for k, vs := range add {
		if _, taken := have[k]; !taken {
			have[k] = vs
		}
	}
	return have.Encode(), nil
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
