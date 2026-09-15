// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package google talks to the Gemini generateContent API.
package google

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/geminiapi"
)

const (
	defaultBaseURL = "https://generativelanguage.googleapis.com"
	defaultPrefix  = "/v1beta"
)

type Options struct {
	ID             string
	BaseURL        string
	Prefix         string
	APIKey         string
	Model          string
	MaxRetries     int
	Timeout        time.Duration
	AllowClientKey bool
	KeyScheme      llm.CredentialScheme
	Header         http.Header
	Sign           func(ctx context.Context, r *http.Request, body []byte) error
}

type Provider struct {
	opts Options
	env  llm.Env
	call *httpcore.Caller
}

func New(o Options) (*Provider, error) {
	if o.ID == "" {
		o.ID = "google"
	}
	if o.BaseURL == "" {
		o.BaseURL = defaultBaseURL
	}
	if o.Prefix == "" {
		o.Prefix = defaultPrefix
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
		Extensions:         []string{geminiapi.Name},
		Streaming:          true,
		Usage:              llm.UsageShape{StreamUsage: true},
		SystemPrompt:       true,
		Tools:              true,
		ParallelTools:      true,
		ToolChoice:         true,
		ToolResults:        true,
		ImageIn:            true,
		Thinking:           true,
		ThinkingSignatures: true,
		ResponseFormat:     true,
		Candidates:         8,
		Ceiling:            llm.CeilingMaxTokens,
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
	if p.opts.Sign == nil {
		if key := p.key(a); key != "" {
			h.Set("x-goog-api-key", key)
		}
	}
	return h
}

func (p *Provider) url(a *llm.Attempt, action string, stream bool) string {
	u := httpcore.JoinURL(p.opts.BaseURL, p.opts.Prefix) + "/models/" + url.PathEscape(p.model(a)) + ":" + action
	if stream {
		u += "?alt=sse"
	}
	return u
}

func (p *Provider) now() time.Time {
	if p.env.Now != nil {
		return p.env.Now()
	}
	return time.Now()
}

func (p *Provider) body(a *llm.Attempt) ([]byte, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	req := a.Request
	body := map[string]any{}

	system, msgs := req.SystemAndMessages()
	if len(system) > 0 {
		parts, d := geminiapi.EncodeParts(system)
		diags = append(diags, d...)
		body["systemInstruction"] = map[string]any{"parts": parts}
	}

	// Roles must alternate, so consecutive user turns coalesce into one.
	var contents []map[string]any
	if len(req.Metadata) > 0 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "metadata",
			Detail: "this backend has no field to name the end user, so the name was not sent",
		})
	}
	for _, m := range msgs {
		role := "user"
		if m.Role == llm.RoleAssistant {
			role = "model"
		}
		if m.Name != "" {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "name",
				Detail: "this backend has no field to name a participant, so the name was not sent",
			})
		}
		parts, d := geminiapi.EncodeParts(m.Content)
		diags = append(diags, d...)
		if len(parts) == 0 {
			diags = append(diags, llm.DroppedMessage(m.Role, p.ID(),
				"this turn reached no part on this endpoint, so the conversation arrives a turn shorter"))
			continue
		}
		if n := len(contents); n > 0 && contents[n-1]["role"] == role {
			contents[n-1]["parts"] = append(contents[n-1]["parts"].([]json.RawMessage), parts...)
			continue
		}
		contents = append(contents, map[string]any{"role": role, "parts": parts})
	}
	if len(contents) == 0 {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "no contents survived translation")
	}
	body["contents"] = contents

	for k, v := range req.Extra {
		if _, taken := body[k]; !taken {
			body[k] = json.RawMessage(v)
		}
	}

	cfg := map[string]any{}
	if req.Temperature != nil {
		cfg["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		cfg["topP"] = *req.TopP
	}
	if req.TopK != nil {
		cfg["topK"] = *req.TopK
	}
	if req.MaxOutputTokens != nil {
		cfg["maxOutputTokens"] = *req.MaxOutputTokens
	}
	if req.Seed != nil {
		cfg["seed"] = *req.Seed
	}
	if req.FrequencyPenalty != nil {
		cfg["frequencyPenalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		cfg["presencePenalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		cfg["stopSequences"] = req.Stop
	}
	if req.Candidates > 1 {
		cfg["candidateCount"] = req.Candidates
	}
	if req.ResponseFormat != nil {
		cfg["responseMimeType"] = "application/json"
		if len(req.ResponseFormat.Schema) > 0 {
			cfg["responseSchema"] = req.ResponseFormat.Schema
		}
	}
	if req.Thinking != nil {
		th := map[string]any{"includeThoughts": true}
		if req.Thinking.Budget != 0 {
			th["thinkingBudget"] = req.Thinking.Budget
		}
		if !req.Thinking.Enabled {
			th = map[string]any{"thinkingBudget": 0}
		}
		cfg["thinkingConfig"] = th
	}
	for k, v := range wirecore.NestedFrom(req.Extra, "generationConfig") {
		if _, taken := cfg[k]; !taken {
			cfg[k] = v
		}
	}
	if len(cfg) > 0 {
		body["generationConfig"] = cfg
	}

	tools := a.Tools
	if len(tools) == 0 {
		tools = req.Tools
	}
	if len(tools) > 0 {
		decls := make([]map[string]any, 0, len(tools))
		for _, t := range tools {
			schema := t.Schema
			if len(schema) == 0 {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			decl := map[string]any{"name": t.Name, "parameters": schema}
			if t.Description != "" {
				decl["description"] = t.Description
			}
			decls = append(decls, decl)
		}
		body["tools"] = []map[string]any{{"functionDeclarations": decls}}
		if llm.StrictTools(tools) {
			diags = append(diags, llm.ToolStrictness(p.ID()))
		}
		if req.ToolChoice != nil {
			if cfg := toolConfig(req.ToolChoice); cfg != nil {
				body["toolConfig"] = cfg
			}
		}
	}

	diags = append(diags, llm.UnsentSampling(req, p.ID(),
		llm.FieldTemperature, llm.FieldTopP, llm.FieldTopK, llm.FieldSeed, llm.FieldStop,
		llm.FieldFrequencyPenalty, llm.FieldPresencePenalty, llm.FieldMaxOutputTokens)...)

	for _, x := range diags {
		a.Note(x)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode request: %v", err)
	}
	return raw, diags, nil
}

func toolConfig(tc *llm.ToolChoice) map[string]any {
	mode := ""
	switch tc.Mode {
	case llm.ToolAuto:
		mode = "AUTO"
	case llm.ToolNone:
		mode = "NONE"
	case llm.ToolRequired, llm.ToolSpecific:
		mode = "ANY"
	default:
		return nil
	}
	inner := map[string]any{"mode": mode}
	if tc.Mode == llm.ToolSpecific && tc.Name != "" {
		inner["allowedFunctionNames"] = []string{tc.Name}
	}
	return map[string]any{"functionCallingConfig": inner}
}

type candidate struct {
	Content struct {
		Role  string            `json:"role"`
		Parts []json.RawMessage `json:"parts"`
	} `json:"content"`
	FinishReason string `json:"finishReason"`
	Index        int    `json:"index"`

	raw json.RawMessage
}

func (c *candidate) UnmarshalJSON(b []byte) error {
	type alias candidate
	var a alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*c = candidate(a)
	c.raw = append(json.RawMessage(nil), b...)
	return nil
}

var candidateKnown = map[string]bool{"content": true, "finishReason": true, "index": true}

type upstreamBody struct {
	Candidates     []candidate `json:"candidates"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	UsageMetadata json.RawMessage `json:"usageMetadata"`
	ModelVersion  string          `json:"modelVersion"`
	ResponseID    string          `json:"responseId"`
	Error         *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

var responseKnown = map[string]bool{
	"candidates": true, "promptFeedback": true, "usageMetadata": true,
	"modelVersion": true, "error": true, "responseId": true,
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	body, diags, err := p.body(a)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.url(a, "generateContent", false), Body: body, Header: p.header(a), Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "response body: " + rerr.Error(), Err: rerr}
	}
	var out upstreamBody
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "malformed response: " + err.Error(), Err: err}
	}
	if out.Error != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Status: out.Error.Code, Code: out.Error.Status, Message: out.Error.Message}
	}
	if len(out.Candidates) == 0 && out.PromptFeedback == nil {
		return nil, &llm.Error{
			Kind: llm.KindUpstream, Provider: p.ID(), Model: p.model(a), Sent: true, Status: 200,
			Message:   "the backend answered with no candidate and no reason for there being none",
			Retryable: true,
		}
	}

	res := &llm.Response{
		ID:       out.ResponseID,
		Provider: p.ID(), Model: firstNonEmpty(out.ModelVersion, p.model(a)), Requested: a.Request.Model,
		Created: p.now(), Diagnostics: diags, Usage: decodeUsage(out.UsageMetadata),
		Extra: wirecore.Extras(raw, responseKnown, a, geminiapi.Name),
	}
	for _, c := range out.Candidates {
		parts, d, perr := geminiapi.DecodeParts(c.Content.Parts)
		if perr != nil {
			return nil, perr
		}
		number(parts, new(int))
		res.Diagnostics = append(res.Diagnostics, d...)
		res.Diagnostics = append(res.Diagnostics, wirecore.Unmodelled(c.raw, candidateKnown, "candidate", a)...)
		res.Choices = append(res.Choices, llm.Choice{
			Index:   c.Index,
			Message: llm.Message{Role: llm.RoleAssistant, Content: parts},
			Stop:    decodeFinish(c.FinishReason, llm.HasKind(parts, llm.KindToolUse)),
		})
	}
	if out.PromptFeedback != nil && out.PromptFeedback.BlockReason != "" {
		res.Block = &llm.Block{Reason: out.PromptFeedback.BlockReason}
		res.Stop = llm.StopContentFilter
	}
	return res, nil
}

func decodeUsage(raw json.RawMessage) llm.Usage {
	if len(raw) == 0 {
		return llm.Usage{}
	}
	var m struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		ThoughtsTokenCount      int `json:"thoughtsTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount"`
		ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount"`
	}
	if json.Unmarshal(raw, &m) != nil {
		return llm.Usage{}
	}
	return llm.RawUsage{
		Prompt: m.PromptTokenCount, Completion: m.CandidatesTokenCount,
		Reasoning: m.ThoughtsTokenCount, CacheRead: m.CachedContentTokenCount,
		ToolPrompt:                 m.ToolUsePromptTokenCount,
		ReasoningOutsideCompletion: true, ToolPromptOutsidePrompt: true, Reported: true,
	}.Normalise()
}

func decodeFinish(s string, hasTool bool) llm.StopReason {
	switch s {
	case "STOP":
		if hasTool {
			return llm.StopToolUse
		}
		return llm.StopEnd
	case "MAX_TOKENS":
		return llm.StopLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "LANGUAGE":
		return llm.StopContentFilter
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	body, _, err := p.body(a)
	if err != nil {
		return nil, err
	}
	h := p.header(a)
	h.Set("Accept", "text/event-stream")
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.url(a, "streamGenerateContent", true), Body: body, Header: h, Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	sc := wirecore.NewScanner(resp.Body, a.Limits.MaxFrameBytes, a.Limits.MaxBufferBytes)
	st := &parser{provider: p.ID(), sc: sc, attempt: a}
	return llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	), nil
}

type parser struct {
	provider string
	sc       *wirecore.Scanner
	done     bool
	ended    bool
	tools    bool
	queue    []*llm.Frame
	attempt  *llm.Attempt
	toolIdx  map[int]*int
}

func (g *parser) toolFor(choice int) *int {
	if g.toolIdx == nil {
		g.toolIdx = map[int]*int{}
	}
	n, ok := g.toolIdx[choice]
	if !ok {
		n = new(int)
		g.toolIdx[choice] = n
	}
	return n
}

func number(parts []llm.Part, next *int) {
	for _, p := range parts {
		if t, ok := p.(*llm.ToolUse); ok && t.Index == nil {
			i := *next
			*next++
			t.Index = &i
		}
	}
}

func (g *parser) next() (*llm.Frame, error) {
	for {
		if len(g.queue) > 0 {
			f := g.queue[0]
			g.queue = g.queue[1:]
			return f, nil
		}
		if g.done {
			return nil, nil
		}
		if !g.sc.Scan() {
			if err := g.sc.Err(); err != nil {
				return nil, &llm.Error{Kind: llm.KindTransport, Provider: g.provider, Sent: true, Message: err.Error(), Err: err}
			}
			g.done = true
			if !g.ended {
				g.ended = true
				return &llm.Frame{TurnEnd: true}, nil
			}
			return nil, nil
		}
		data := bytes.TrimSpace(g.sc.Event().Data)
		if len(data) == 0 {
			continue
		}
		var out upstreamBody
		if err := json.Unmarshal(data, &out); err != nil {
			return nil, &llm.Error{
				Kind: llm.KindUpstream, Provider: g.provider, Sent: true, Code: llm.WireFrameMalformed,
				Message: "malformed stream frame: " + err.Error(), Err: err,
			}
		}
		if out.Error != nil {
			g.done = true
			return &llm.Frame{Err: &llm.Error{
				Kind: llm.KindUpstream, Provider: g.provider, Sent: true,
				Status: out.Error.Code, Code: out.Error.Status, Message: out.Error.Message,
			}}, nil
		}
		var frames []*llm.Frame
		for _, c := range out.Candidates {
			idx := c.Index
			parts, _, err := geminiapi.DecodeParts(c.Content.Parts)
			if err != nil {
				return nil, err
			}
			number(parts, g.toolFor(idx))
			if llm.HasKind(parts, llm.KindToolUse) {
				g.tools = true
			}
			frames = append(frames, &llm.Frame{
				ID: out.ResponseID, Model: out.ModelVersion, Choice: &idx, Delta: parts,
				Stop:        decodeFinish(c.FinishReason, g.tools),
				Diagnostics: wirecore.Unmodelled(c.raw, candidateKnown, "candidate", g.attempt),
			})
		}
		if len(frames) == 0 {
			frames = append(frames, &llm.Frame{ID: out.ResponseID, Model: out.ModelVersion})
		}
		if u := decodeUsage(out.UsageMetadata); !u.Empty() {
			frames[0].Usage = &u
		}
		if out.PromptFeedback != nil && out.PromptFeedback.BlockReason != "" {
			frames[0].Block = &llm.Block{Reason: out.PromptFeedback.BlockReason}
			frames[0].Stop = llm.StopContentFilter
		}
		frames[0].Extra = wirecore.Extras(data, responseKnown, g.attempt, geminiapi.Name)
		kept := frames[:0]
		for _, f := range frames {
			if f.Empty() && f.Model == "" {
				continue
			}
			kept = append(kept, f)
		}
		if len(kept) == 0 {
			continue
		}
		g.queue = append(g.queue, kept[1:]...)
		return kept[0], nil
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
	u := httpcore.JoinURL(p.opts.BaseURL, r.Path)
	if r.Query != "" {
		u += "?" + r.Query
	}
	req, err := http.NewRequestWithContext(ctx, r.Method, u, r.Body)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: p.ID(), Message: err.Error()}
	}
	httpcore.ForwardPassthrough(req.Header, r.Header)
	if key := p.passKey(r); key != "" {
		req.Header.Set("x-goog-api-key", key)
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
