// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package anthropic talks to the Anthropic Messages API.
package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

const (
	defaultBaseURL = "https://api.anthropic.com"
	defaultVersion = "2023-06-01"
	maxBreakpoints = 4
)

type Options struct {
	ID             string
	BaseURL        string
	APIKey         string
	Version        string
	Betas          []string
	Model          string
	MaxRetries     int
	Timeout        time.Duration
	AllowClientKey bool
	KeyScheme      llm.CredentialScheme
	Header         http.Header
	NoConnectors   bool

	Path        func(a *llm.Attempt, stream bool) string
	RewriteBody func(body map[string]any, stream bool)
	// Sign authenticates a request the backend will not take a static key for.
	// Passthrough forwards a body it never reads, so it signs a nil one: a
	// signer that covers the body cannot be used on that path.
	Sign func(ctx context.Context, r *http.Request, body []byte) error
}

type Provider struct {
	opts Options
	env  llm.Env
	call *httpcore.Caller
}

func New(o Options) (*Provider, error) {
	if o.ID == "" {
		o.ID = "anthropic"
	}
	if o.BaseURL == "" {
		o.BaseURL = defaultBaseURL
	}
	if o.Version == "" {
		o.Version = defaultVersion
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
		KeyScheme:          p.opts.KeyScheme,
		Streaming:          true,
		Usage:              llm.UsageShape{StreamUsage: true},
		SystemPrompt:       true,
		Tools:              true,
		ParallelTools:      true,
		ToolChoice:         true,
		ToolResults:        true,
		ToolErrors:         true,
		ImageIn:            true,
		Thinking:           true,
		ThinkingSignatures: true,
		RedactedThinking:   true,
		Effort:             false,
		ResponseFormat:     false,
		Sessions:           false,
		MCPServers:         !p.opts.NoConnectors,
		Extensions:         []string{anthropicmsg.Name},
		Candidates:         1,
		CacheBreakpoints:   maxBreakpoints,
		Ceiling:            llm.CeilingMaxTokens,
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

func (p *Provider) header(a *llm.Attempt) http.Header {
	h := http.Header{}
	for k, vs := range p.opts.Header {
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	h.Set("anthropic-version", p.opts.Version)
	if key := p.key(a); key != "" && p.opts.Sign == nil {
		h.Set("x-api-key", key)
	}
	betas := p.opts.Betas
	if a != nil && a.Request != nil && len(a.Request.MCPServers) > 0 && !slices.Contains(betas, mcpBeta) {
		betas = append(append([]string(nil), betas...), mcpBeta)
	}
	for _, b := range betas {
		h.Add("anthropic-beta", b)
	}
	return h
}

const mcpBeta = "mcp-client-2025-11-20"

type upstreamResponse struct {
	ID           string              `json:"id"`
	Type         string              `json:"type"`
	Model        string              `json:"model"`
	Content      []json.RawMessage   `json:"content"`
	StopReason   string              `json:"stop_reason"`
	StopSequence string              `json:"stop_sequence"`
	Usage        *anthropicmsg.Usage `json:"usage"`
	Error        *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

var responseKnown = map[string]bool{
	"id": true, "type": true, "role": true, "model": true, "content": true,
	"stop_reason": true, "stop_sequence": true, "usage": true,
}

func decodeUsage(u *anthropicmsg.Usage) llm.Usage {
	if u == nil {
		return llm.Usage{}
	}
	return anthropicmsg.DecodeUsage(*u)
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	body, diags, err := p.buildRequest(a, false)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.endpoint(a, false), Body: body, Header: p.header(a), Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		return nil, httpcore.BodyError(ctx, p.ID(), rerr)
	}
	var out upstreamResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Message: "malformed response: " + err.Error(), Sent: true, Err: err}
	}
	if out.Error != nil || out.Type == "error" {
		message, kind := "the upstream refused this request", ""
		if out.Error != nil {
			if out.Error.Message != "" {
				message = out.Error.Message
			}
			kind = out.Error.Type
		}
		return nil, &llm.Error{
			Kind: kindForType(kind), Provider: p.ID(), Sent: true, Status: 200,
			Code: kind, Message: message, Body: string(raw),
		}
	}
	parts, _, d, perr := anthropicmsg.DecodeBlocks(mustArray(out.Content))
	if perr != nil {
		return nil, perr
	}
	diags = append(diags, d...)

	stop := decodeStop(out.StopReason)
	return &llm.Response{
		ID:        out.ID,
		Provider:  p.ID(),
		Model:     out.Model,
		Requested: a.Request.Model,
		Created:   p.now(),
		Choices: []llm.Choice{{
			Index:        0,
			Message:      llm.Message{Role: llm.RoleAssistant, Content: parts},
			Stop:         stop,
			StopSequence: out.StopSequence,
		}},
		Usage:       decodeUsage(out.Usage),
		Diagnostics: diags,
		Extra:       wirecore.Extras(raw, responseKnown, a, anthropicmsg.Name),
	}, nil
}

func (p *Provider) now() time.Time {
	if p.env.Now != nil {
		return p.env.Now()
	}
	return time.Now()
}

func mustArray(items []json.RawMessage) json.RawMessage {
	b, err := json.Marshal(items)
	if err != nil {
		return json.RawMessage(`[]`)
	}
	return b
}

func decodeStop(s string) llm.StopReason {
	switch s {
	case "end_turn":
		return llm.StopEnd
	case "max_tokens":
		return llm.StopLength
	case "stop_sequence":
		return llm.StopSequence
	case "tool_use":
		return llm.StopToolUse
	case "pause_turn":
		return llm.StopPause
	case "refusal":
		return llm.StopRefusal
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

func (p *Provider) endpoint(a *llm.Attempt, stream bool) string {
	if p.opts.Path != nil {
		return httpcore.JoinURL(p.opts.BaseURL, p.opts.Path(a, stream))
	}
	return httpcore.JoinURL(p.opts.BaseURL, "/v1/messages")
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
	req.Header.Set("anthropic-version", p.opts.Version)
	if key := p.passKey(r); key != "" && p.opts.Sign == nil {
		req.Header.Set("x-api-key", key)
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
