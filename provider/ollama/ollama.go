// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package ollama talks to a local or remote Ollama server.
package ollama

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/ollamaapi"
)

const defaultBaseURL = "http://127.0.0.1:11434"

type Options struct {
	ID         string
	BaseURL    string
	APIKey     string
	Model      string
	KeepAlive  string
	MaxRetries int
	Timeout    time.Duration
	Header     http.Header
}

type Provider struct {
	opts Options
	env  llm.Env
	call *httpcore.Caller
}

func New(o Options) (*Provider, error) {
	if o.ID == "" {
		o.ID = "ollama"
	}
	if o.BaseURL == "" {
		o.BaseURL = defaultBaseURL
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
		Streaming:      true,
		Extensions:     []string{ollamaapi.NameChat},
		Usage:          llm.UsageShape{StreamUsage: true},
		SystemPrompt:   true,
		Tools:          true,
		ParallelTools:  true,
		ToolResults:    true,
		ImageIn:        true,
		Thinking:       true,
		Effort:         true,
		ResponseFormat: true,
		Candidates:     1,
		Ceiling:        llm.CeilingMaxTokens,
	}
}

func thinkLevel(e llm.Effort) string {
	switch e {
	case llm.EffortMinimal, llm.EffortLow:
		return "low"
	case llm.EffortMedium:
		return "medium"
	case llm.EffortHigh:
		return "high"
	default:
		return ""
	}
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

func (p *Provider) header() http.Header {
	h := http.Header{}
	for k, vs := range p.opts.Header {
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	if p.opts.APIKey != "" {
		h.Set("Authorization", "Bearer "+p.opts.APIKey)
	}
	return h
}

func (p *Provider) now() time.Time {
	if p.env.Now != nil {
		return p.env.Now()
	}
	return time.Now()
}

func (p *Provider) body(a *llm.Attempt, stream bool) ([]byte, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	req := a.Request
	body := map[string]any{"model": p.model(a), "stream": stream}

	var msgs []map[string]any
	system, rest := req.SystemAndMessages()
	if len(system) > 0 {
		msgs = append(msgs, map[string]any{"role": "system", "content": llm.TextOf(system)})
		diags = append(diags, llm.Flattened(system, p.ID(),
			"this backend takes a system prompt as text, so what was not text did not travel")...)
	}
	if len(req.Metadata) > 0 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "metadata",
			Detail: "this backend has no field to name the end user, so the name was not sent",
		})
	}
	for _, m := range rest {
		if m.Role == llm.RoleTool {
			for _, part := range m.Content {
				if tr, ok := part.(*llm.ToolResult); ok {
					msgs = append(msgs, map[string]any{"role": "tool", "tool_name": tr.Name, "content": llm.TextOf(tr.Content)})
					diags = append(diags, llm.Flattened(tr.Content, p.ID(),
						"this backend takes a tool result as text, so what was not text did not travel")...)
				}
			}
			continue
		}
		entry := map[string]any{"role": string(m.Role)}
		if m.Name != "" {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "name",
				Detail: "this backend has no field to name a participant, so the name was not sent",
			})
		}
		var content, thinking string
		var images []string
		var calls []map[string]any
		for _, part := range m.Content {
			switch v := part.(type) {
			case *llm.Text:
				content += v.Text
			case *llm.Refusal:
				content += v.Text
			case *llm.Thinking:
				thinking += v.Text
			case *llm.Image:
				if len(v.Data) > 0 {
					images = append(images, base64.StdEncoding.EncodeToString(v.Data))
					continue
				}
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityLoss, Provider: p.ID(), Detail: "this backend takes image bytes, not a URL",
				})
			case *llm.ToolUse:
				args := wirecore.ObjectArgs(v.Args)
				calls = append(calls, map[string]any{"function": map[string]any{"name": v.Name, "arguments": args}})
			case *llm.Raw:
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityLoss, Provider: p.ID(), Field: v.Type,
				})
			}
		}
		entry["content"] = content
		if thinking != "" {
			entry["thinking"] = thinking
		}
		if len(images) > 0 {
			entry["images"] = images
		}
		if len(calls) > 0 {
			entry["tool_calls"] = calls
		}
		msgs = append(msgs, entry)
	}
	if len(msgs) == 0 {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "no messages survived translation")
	}
	body["messages"] = msgs

	opts := map[string]any{}
	if req.Temperature != nil {
		opts["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		opts["top_p"] = *req.TopP
	}
	if req.TopK != nil {
		opts["top_k"] = *req.TopK
	}
	if req.Seed != nil {
		opts["seed"] = *req.Seed
	}
	if req.MaxOutputTokens != nil {
		opts["num_predict"] = *req.MaxOutputTokens
	}
	if len(req.Stop) > 0 {
		opts["stop"] = req.Stop
	}
	if req.FrequencyPenalty != nil {
		opts["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		opts["presence_penalty"] = *req.PresencePenalty
	}
	for k, v := range wirecore.NestedFrom(req.Extra, "options") {
		if _, taken := opts[k]; !taken {
			opts[k] = v
		}
	}
	if len(opts) > 0 {
		body["options"] = opts
	}
	diags = append(diags, llm.UnsentSampling(req, p.ID(),
		llm.FieldTemperature, llm.FieldTopP, llm.FieldTopK, llm.FieldSeed, llm.FieldStop,
		llm.FieldFrequencyPenalty, llm.FieldPresencePenalty, llm.FieldMaxOutputTokens)...)
	if req.ResponseFormat != nil {
		if len(req.ResponseFormat.Schema) > 0 {
			body["format"] = req.ResponseFormat.Schema
		} else {
			body["format"] = "json"
		}
	}
	if req.Thinking != nil {
		if level := thinkLevel(req.Effort); level != "" && req.Thinking.Enabled {
			body["think"] = level
		} else {
			body["think"] = req.Thinking.Enabled
		}
	}
	if req.Effort != "" && (req.Thinking == nil || !req.Thinking.Enabled) {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "reasoning_effort",
			Detail: "this backend grades thinking only once it is switched on, so the effort was not sent",
		})
	}
	tools := a.Tools
	if len(tools) == 0 {
		tools = req.Tools
	}
	if len(tools) > 0 {
		list := make([]map[string]any, 0, len(tools))
		for _, t := range tools {
			schema := t.Schema
			if len(schema) == 0 {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			fn := map[string]any{"name": t.Name, "parameters": schema}
			if t.Description != "" {
				fn["description"] = t.Description
			}
			list = append(list, map[string]any{"type": "function", "function": fn})
		}
		body["tools"] = list
		if llm.StrictTools(tools) {
			diags = append(diags, llm.ToolStrictness(p.ID()))
		}
	}
	for k, v := range req.Extra {
		if _, taken := body[k]; !taken {
			body[k] = json.RawMessage(v)
		}
	}
	if p.opts.KeepAlive != "" {
		body["keep_alive"] = p.opts.KeepAlive
	}

	for _, x := range diags {
		a.Note(x)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode request: %v", err)
	}
	return raw, diags, nil
}

type upstreamFrame struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role      string   `json:"role"`
		Content   string   `json:"content"`
		Thinking  string   `json:"thinking"`
		Images    []string `json:"images"`
		ToolCalls []struct {
			Function struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"message"`
	Done            bool   `json:"done"`
	DoneReason      string `json:"done_reason"`
	PromptEvalCount int    `json:"prompt_eval_count"`
	EvalCount       int    `json:"eval_count"`
	TotalDuration   int64  `json:"total_duration"`
	Error           string `json:"error"`
}

func (f *upstreamFrame) parts(toolIdx *int, mint *wirecore.IDMinter) []llm.Part {
	var out []llm.Part
	if f.Message.Thinking != "" {
		out = append(out, &llm.Thinking{Text: f.Message.Thinking})
	}
	if f.Message.Content != "" {
		out = append(out, &llm.Text{Text: f.Message.Content})
	}
	for _, img := range f.Message.Images {
		if b, err := base64.StdEncoding.DecodeString(img); err == nil {
			out = append(out, &llm.Image{Data: b})
		}
	}
	for _, tc := range f.Message.ToolCalls {
		idx := *toolIdx
		*toolIdx++
		out = append(out, &llm.ToolUse{
			ID: mint.Next(tc.Function.Name), Name: tc.Function.Name,
			Args: tc.Function.Arguments, Index: &idx,
		})
	}
	return out
}

func (f *upstreamFrame) created(fallback time.Time) time.Time {
	if t, err := time.Parse(time.RFC3339Nano, f.CreatedAt); err == nil && !t.IsZero() {
		return t.UTC()
	}
	return fallback
}

func (f *upstreamFrame) usage() llm.Usage {
	if f.PromptEvalCount == 0 && f.EvalCount == 0 && !f.Done {
		return llm.Usage{}
	}
	return llm.RawUsage{Prompt: f.PromptEvalCount, Completion: f.EvalCount, Reported: true}.Normalise()
}

func decodeStop(s string, hasTool bool) llm.StopReason {
	switch s {
	case "stop":
		if hasTool {
			return llm.StopToolUse
		}
		return llm.StopEnd
	case "length":
		return llm.StopLength
	case "load", "unload":
		return llm.StopEnd
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

var responseKnown = map[string]bool{
	"model": true, "created_at": true, "message": true, "done": true,
	"done_reason": true, "prompt_eval_count": true, "eval_count": true,
	"total_duration": true, "error": true,
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	body, diags, err := p.body(a, false)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: httpcore.JoinURL(p.opts.BaseURL, "/api/chat"), Body: body, Header: p.header(), OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		return nil, httpcore.BodyError(ctx, p.ID(), rerr)
	}
	var out upstreamFrame
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "malformed response: " + err.Error(), Err: err}
	}
	if out.Error != "" {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Status: 200, Message: out.Error}
	}
	idx := 0
	parts := out.parts(&idx, wirecore.NewIDMinter())
	u := out.usage()
	u.DurationMS = out.TotalDuration / int64(time.Millisecond)
	return &llm.Response{
		Provider: p.ID(), Model: out.Model, Requested: a.Request.Model, Created: out.created(p.now()),
		Diagnostics: diags, Usage: u,
		Extra: wirecore.Extras(raw, responseKnown, a, ollamaapi.NameChat, ollamaapi.NameGenerate),
		Choices: []llm.Choice{{
			Index:   0,
			Message: llm.Message{Role: llm.RoleAssistant, Content: parts},
			Stop:    decodeStop(out.DoneReason, llm.HasKind(parts, llm.KindToolUse)),
		}},
	}, nil
}

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	body, _, err := p.body(a, true)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: httpcore.JoinURL(p.opts.BaseURL, "/api/chat"), Body: body, Header: p.header(), OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	lines := wirecore.NewLines(resp.Body, a.Limits.MaxFrameBytes)
	st := &parser{ctx: ctx, provider: p.ID(), lines: lines, attempt: a, mint: wirecore.NewIDMinter()}
	return llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	), nil
}

type parser struct {
	ctx      context.Context
	provider string
	lines    *wirecore.Lines
	done     bool
	toolIdx  int
	hasTool  bool
	attempt  *llm.Attempt
	mint     *wirecore.IDMinter
}

func (o *parser) next() (*llm.Frame, error) {
	for {
		if o.done {
			return nil, nil
		}
		if !o.lines.Scan() {
			if err := o.lines.Err(); err != nil {
				return nil, httpcore.ReadError(o.ctx, o.provider, err)
			}
			o.done = true
			return nil, nil
		}
		data := bytes.TrimSpace(o.lines.Bytes())
		if len(data) == 0 {
			continue
		}
		var f upstreamFrame
		if err := json.Unmarshal(data, &f); err != nil {
			return nil, &llm.Error{
				Kind: llm.KindUpstream, Provider: o.provider, Sent: true, Code: llm.WireFrameMalformed,
				Message: "malformed stream frame: " + err.Error(), Err: err,
			}
		}
		if f.Error != "" {
			o.done = true
			return &llm.Frame{Err: &llm.Error{Kind: llm.KindUpstream, Provider: o.provider, Sent: true, Status: 200, Message: f.Error}}, nil
		}
		frame := &llm.Frame{
			Model: f.Model, Delta: f.parts(&o.toolIdx, o.mint),
			Extra: wirecore.Extras(data, responseKnown, o.attempt, ollamaapi.NameChat, ollamaapi.NameGenerate),
		}
		if llm.HasKind(frame.Delta, llm.KindToolUse) {
			o.hasTool = true
		}
		if u := f.usage(); !u.Empty() {
			u.DurationMS = f.TotalDuration / int64(time.Millisecond)
			frame.Usage = &u
		}
		if f.Done {
			o.done = true
			frame.Stop = decodeStop(f.DoneReason, o.hasTool)
			if frame.Stop == llm.StopNone {
				frame.Stop = llm.StopEnd
			}
			frame.TurnEnd = true
		}
		if frame.Empty() {
			continue
		}
		return frame, nil
	}
}

func (p *Provider) Pass(ctx context.Context, r *llm.PassRequest) (*llm.PassResponse, error) {
	u := httpcore.JoinURL(p.opts.BaseURL, r.Path)
	if r.Query != "" {
		u += "?" + r.Query
	}
	req, err := http.NewRequestWithContext(ctx, r.Method, u, r.Body)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: p.ID(), Message: err.Error()}
	}
	httpcore.ForwardPassthrough(req.Header, r.Header)
	if p.opts.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.opts.APIKey)
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
