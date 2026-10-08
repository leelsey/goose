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
	"strings"
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
		o.ID = "google"
	}
	if o.BaseURL == "" {
		o.BaseURL = defaultBaseURL
	}
	if o.Prefix == "" {
		o.Prefix = defaultPrefix
	}
	p := &Provider{opts: o}
	p.call = &httpcore.Caller{Provider: o.ID, MaxRetries: o.MaxRetries, HTTP: httpcore.NewClient(o.Timeout), Refine: refine}
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
		ClientKey:          p.opts.AllowClientKey && p.opts.APIKey == "" && p.opts.Sign == nil,
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
		Signer:             llm.FamilyGoogle,
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
	u := httpcore.JoinURL(p.opts.BaseURL, p.opts.Prefix) + "/" + modelPath(p.model(a)) + ":" + action
	if stream {
		u += "?alt=sse"
	}
	return u
}

// modelPath names the collection the model is served from. A tuned model has
// its own; on Vertex the publisher is already a path segment, so a model
// written "google/gemini-..." names it twice, and escaping the slash would bury
// the second copy inside the model id.
func modelPath(model string) string {
	if collection, id, ok := strings.Cut(model, "/"); ok && collection == "tunedModels" && id != "" {
		return "tunedModels/" + url.PathEscape(id)
	}
	if i := strings.LastIndexByte(model, '/'); i >= 0 {
		model = model[i+1:]
	}
	return "models/" + url.PathEscape(model)
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
	calls := callNames{}
	turn := currentTurn(msgs)
	placeheld, owed := 0, false
	for i, m := range msgs {
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
		// A step is one content of the model's, which the turns coalescing
		// below may build from several messages; its first call is the one
		// that must be signed.
		var pending *bool
		if role == "model" {
			if len(contents) == 0 || contents[len(contents)-1]["role"] != role {
				owed = i >= turn
			}
			pending = &owed
		}
		content, n, err := calls.fill(m.Content, pending)
		if err != nil {
			e := llm.AsError(err)
			e.Provider = p.ID()
			return nil, diags, e
		}
		placeheld += n
		parts, d := geminiapi.EncodeParts(content)
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
	if placeheld > 0 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropSignature, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "thoughtSignature", Count: placeheld,
			Detail: "a call in this turn came back without the signature for the model's reasoning before it, so that reasoning was not carried and Google's placeholder went in its place",
		})
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
		// A caller on this wire can say more than the neutral request holds,
		// such as a thinking level. What it said is kept, and what the neutral
		// request holds is written over it.
		th := map[string]any{}
		for k, v := range wirecore.NestedFrom(wirecore.NestedFrom(req.Extra, "generationConfig"), "thinkingConfig") {
			th[k] = v
		}
		if !req.Thinking.Hidden {
			th["includeThoughts"] = true
		}
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
	if err := a.Check(); err != nil {
		return nil, diags, err
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
		return nil, httpcore.BodyError(ctx, p.ID(), rerr)
	}
	var out upstreamBody
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "malformed response: " + err.Error(), Err: err}
	}
	if out.Error != nil {
		return nil, &llm.Error{
			Kind: llm.KindUpstream, Provider: p.ID(), Sent: true,
			Status: out.Error.Code, Code: out.Error.Status, Message: out.Error.Message, Body: httpcore.ErrorBody(raw),
		}
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
	sc := wirecore.NewScanner(resp.Body, a.Limits.MaxFrameBytes, a.Limits.MaxFrameBytes)
	st := &parser{ctx: ctx, provider: p.ID(), sc: sc, attempt: a}
	return llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	), nil
}

type parser struct {
	ctx      context.Context
	provider string
	sc       *wirecore.Scanner
	done     bool
	ended    bool
	queue    []*llm.Frame
	attempt  *llm.Attempt
	toolIdx  map[int]*int
	toolSeen map[int]bool
	beside   wirecore.Beside
}

// Whether a candidate has called a tool is that candidate's own business; a
// sibling's call says nothing about how this one finishes.
func (g *parser) calledTool(choice int, now bool) bool {
	if g.toolSeen == nil {
		g.toolSeen = map[int]bool{}
	}
	if now {
		g.toolSeen[choice] = true
	}
	return g.toolSeen[choice]
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
				return nil, httpcore.ReadError(g.ctx, g.provider, err)
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
				Status: out.Error.Code, Code: out.Error.Status, Message: out.Error.Message, Body: httpcore.ErrorBody(data),
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
			called := g.calledTool(idx, llm.HasKind(parts, llm.KindToolUse))
			frames = append(frames, &llm.Frame{
				ID: out.ResponseID, Model: out.ModelVersion, Choice: &idx, Delta: parts,
				Stop:        decodeFinish(c.FinishReason, called),
				Diagnostics: g.beside.Note(c.raw, candidateKnown, "candidate", g.attempt),
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
			if f.Empty() {
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

// refine reads what the status does not say. Gemini answers a key that is not
// valid, and a region or an account it will not serve, with the 400 it gives a
// malformed request, and only a malformed request fails the same everywhere.
func refine(e *llm.Error) {
	if e.Kind != llm.KindInvalidRequest {
		return
	}
	var body struct {
		Error map[string]json.RawMessage `json:"error"`
	}
	if json.Unmarshal([]byte(e.Body), &body) != nil {
		return
	}
	var details []json.RawMessage
	_ = json.Unmarshal(body.Error["details"], &details)
	for _, raw := range details {
		var d struct {
			Reason string `json:"reason"`
		}
		if json.Unmarshal(raw, &d) != nil {
			continue
		}
		switch d.Reason {
		case "API_KEY_INVALID", "API_KEY_EXPIRED":
			e.Kind = llm.KindAuth
			return
		}
	}
	// The same status names a request that is not ready, such as one naming a
	// file still being processed, so only a region or billing refusal is the
	// account's.
	var status string
	if json.Unmarshal(body.Error["status"], &status) != nil || status != "FAILED_PRECONDITION" {
		return
	}
	said := strings.ToLower(e.Message)
	if strings.Contains(said, "user location") || strings.Contains(said, "billing") {
		e.Kind = llm.KindPermission
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
	q, err := httpcore.PassQuery(p.ID(), r.Query)
	if err != nil {
		return nil, err
	}
	if q != "" {
		u += "?" + q
	}
	req, err := http.NewRequestWithContext(ctx, r.Method, u, r.Body)
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: p.ID(), Message: err.Error()}
	}
	httpcore.ForwardPassthrough(req.Header, r.Header)
	httpcore.Overlay(req.Header, p.opts.Header)
	if key := p.passKey(r); key != "" && p.opts.Sign == nil {
		req.Header.Set("x-goog-api-key", key)
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
