// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"sort"
	"strings"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/openaichat"
)

func (p *Provider) ceilingField(model string) string {
	if p.Capabilities().Ceiling != llm.CeilingPerModel {
		return string(llm.CeilingMaxTokens)
	}
	m := strings.ToLower(lastSegment(model))
	for _, prefix := range []string{"o1", "o3", "o4", "gpt-5"} {
		if strings.HasPrefix(m, prefix) {
			return string(llm.CeilingMaxCompletion)
		}
	}
	return string(llm.CeilingMaxTokens)
}

func (p *Provider) chatBody(a *llm.Attempt, stream bool) ([]byte, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	req := a.Request
	model := p.model(a)
	body := map[string]any{"model": model}

	sentParallel := false
	msgs, d := chatMessages(req, p.ID(), p.Capabilities().Takes(openaichat.Name))
	diags = append(diags, d...)
	if len(msgs) == 0 {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "no messages survived translation")
	}
	body["messages"] = msgs

	if req.MaxOutputTokens != nil {
		field := p.ceilingField(model)
		body[field] = *req.MaxOutputTokens
		if field == string(llm.CeilingMaxCompletion) {
			diags = append(diags, llm.Diagnostic{
				Code: llm.CapCeilingRespelt, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Model: model,
				Detail: "this model family rejects max_tokens",
			})
		}
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.Seed != nil {
		body["seed"] = *req.Seed
	}
	if req.FrequencyPenalty != nil {
		body["frequency_penalty"] = *req.FrequencyPenalty
	}
	if req.PresencePenalty != nil {
		body["presence_penalty"] = *req.PresencePenalty
	}
	if len(req.Stop) > 0 {
		body["stop"] = req.Stop
	}
	if req.Candidates > 1 {
		body["n"] = req.Candidates
	}
	if req.Effort != "" {
		body["reasoning_effort"] = string(req.Effort)
	}
	if req.ResponseFormat != nil {
		body["response_format"] = responseFormat(req.ResponseFormat)
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
			if t.Strict {
				fn["strict"] = true
			}
			if t.Description != "" {
				fn["description"] = t.Description
			}
			list = append(list, map[string]any{"type": "function", "function": fn})
		}
		body["tools"] = list
		if req.ToolChoice != nil {
			if tc := toolChoice(req.ToolChoice); tc != nil {
				body["tool_choice"] = tc
			}
		}
		if req.ParallelTools != nil {
			body["parallel_tool_calls"] = *req.ParallelTools
			sentParallel = true
		}
	}

	if u, key := caller(req.Metadata); u != "" {
		body["user"] = u
		if rest := unsentMeta(req.Metadata, key); len(rest) > 0 {
			diags = append(diags, metaDropped(p.ID(), rest))
		}
	} else if rest := unsentMeta(req.Metadata, ""); len(rest) > 0 {
		diags = append(diags, metaDropped(p.ID(), rest))
	}

	if stream {
		body["stream"] = true
		shape := p.Capabilities().Usage
		if shape.StreamUsage && shape.StreamUsageOptIn {
			body["stream_options"] = map[string]any{"include_usage": true}
		} else if !shape.StreamUsage {
			diags = append(diags, llm.Diagnostic{
				Code: llm.CapNoStreamUsage, Class: llm.ClassAccounting, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(),
				Detail: "this backend reports no usage on a streamed call",
			})
		}
	}

	if req.Thinking != nil && req.Thinking.Enabled && req.Effort == "" {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropThinking, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "thinking",
			Detail: "this endpoint takes an effort rather than a thinking budget, and the request named no effort",
		})
	}
	diags = append(diags, p.extend(body, req)...)
	sent := []string{
		llm.FieldTemperature, llm.FieldTopP, llm.FieldSeed, llm.FieldStop,
		llm.FieldFrequencyPenalty, llm.FieldPresencePenalty, llm.FieldMaxOutputTokens,
	}
	if sentParallel {
		sent = append(sent, llm.FieldParallelTools)
	}
	diags = append(diags, llm.UnsentSampling(req, p.ID(), sent...)...)

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

func responseFormat(f *llm.ResponseFormat) map[string]any {
	if len(f.Schema) == 0 {
		if f.Type == "json_schema" {
			return map[string]any{"type": "json_object"}
		}
		return map[string]any{"type": f.Type}
	}
	return map[string]any{"type": "json_schema", "json_schema": map[string]any{
		"name": nameOr(f.Name, "response"), "strict": f.Strict, "schema": f.Schema,
	}}
}

func nameOr(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func toolChoice(tc *llm.ToolChoice) any {
	switch tc.Mode {
	case llm.ToolAuto:
		return "auto"
	case llm.ToolNone:
		return "none"
	case llm.ToolRequired:
		return "required"
	case llm.ToolSpecific:
		return map[string]any{"type": "function", "function": map[string]any{"name": tc.Name}}
	default:
		return nil
	}
}

func (p *Provider) extend(body map[string]any, req *llm.Request) llm.Diagnostics {
	var diags llm.Diagnostics
	if len(req.Extra) > 0 {
		if p.Capabilities().Takes(req.ExtraFrom) {
			for k, v := range req.Extra {
				if _, taken := body[k]; !taken {
					body[k] = json.RawMessage(v)
				}
			}
		} else {
			for k := range req.Extra {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
					Severity: llm.SeverityInfo, Provider: p.ID(), Field: k,
					Detail: "this backend rejects unknown request fields",
				})
			}
		}
	}
	for k, v := range p.opts.Body {
		body[k] = v
	}
	return diags
}

const toolResultText = "this endpoint takes a tool result as text, so what was not text did not travel"

func chatMessages(req *llm.Request, provider string, verbatim bool) ([]map[string]any, llm.Diagnostics) {
	var diags llm.Diagnostics
	var out []map[string]any
	system, msgs := req.SystemAndMessages()
	if len(system) > 0 {
		out = append(out, map[string]any{"role": "system", "content": llm.TextOf(system)})
		diags = append(diags, llm.Flattened(system, provider,
			"this endpoint takes a system prompt as text, so what was not text did not travel")...)
	}
	for _, m := range msgs {
		role := string(m.Role)
		switch m.Role {
		case llm.RoleDeveloper:
			role = "developer"
		case llm.RoleTool:
			for _, p := range m.Content {
				if tr, ok := p.(*llm.ToolResult); ok {
					out = append(out, map[string]any{
						"role": "tool", "tool_call_id": tr.ID, "content": llm.TextOf(tr.Content),
					})
					diags = append(diags, llm.Flattened(tr.Content, provider, toolResultText)...)
				}
			}
			continue
		}
		msg := map[string]any{"role": role}
		if m.Name != "" {
			msg["name"] = m.Name
		}
		var content []any
		var text strings.Builder
		var toolCalls []map[string]any
		listed, results := false, false
		for _, p := range m.Content {
			switch v := p.(type) {
			case *llm.Text:
				text.WriteString(v.Text)
				content = append(content, map[string]any{"type": "text", "text": v.Text})
			case *llm.Refusal:
				msg["refusal"] = v.Text
			case *llm.Image:
				url := v.URL
				if len(v.Data) > 0 {
					url = "data:" + nameOr(v.MIME, "image/png") + ";base64," + base64.StdEncoding.EncodeToString(v.Data)
				}
				img := map[string]any{"url": url}
				if v.Detail != "" {
					img["detail"] = v.Detail
				}
				content = append(content, map[string]any{"type": "image_url", "image_url": img})
				listed = true
			case *llm.ToolUse:
				args := string(v.Args)
				if args == "" {
					args = "{}"
				}
				toolCalls = append(toolCalls, map[string]any{
					"id": v.ID, "type": "function",
					"function": map[string]any{"name": v.Name, "arguments": args},
				})
			case *llm.Thinking:
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropThinking, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityInfo, Detail: "this endpoint has no field to replay thinking",
				})
			case *llm.ToolResult:
				results = true
				out = append(out, map[string]any{"role": "tool", "tool_call_id": v.ID, "content": llm.TextOf(v.Content)})
				diags = append(diags, llm.Flattened(v.Content, provider, toolResultText)...)
			case *llm.Raw:
				if verbatim && v.Vendor == openaichat.Name && json.Valid(v.JSON) {
					content = append(content, v.JSON)
					listed = true
					continue
				}
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityLoss, Field: v.Type, Detail: "content block has no shape on this endpoint",
				})
			}
		}
		switch {
		case listed:
			msg["content"] = content
		case text.Len() > 0:
			msg["content"] = text.String()
		case len(toolCalls) > 0 || msg["refusal"] != nil:
			msg["content"] = nil
		case results:
			// The wires that carry tool results inside a user turn leave that
			// turn empty once they are sent as tool messages, and an empty
			// turn after them is one the model reads as the user saying
			// nothing.
			continue
		default:
			msg["content"] = ""
		}
		if len(toolCalls) > 0 {
			msg["tool_calls"] = toolCalls
		}
		out = append(out, msg)
	}
	return out, diags
}

type chatUsage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens     int `json:"cached_tokens"`
		CacheWriteTokens int `json:"cache_write_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details"`
}

func (u *chatUsage) neutral() llm.Usage {
	return llm.RawUsage{
		Prompt: u.PromptTokens, Completion: u.CompletionTokens,
		CacheRead: u.PromptTokensDetails.CachedTokens, CacheWrite: u.PromptTokensDetails.CacheWriteTokens,
		Reasoning: u.CompletionTokensDetails.ReasoningTokens, Reported: true,
	}.Normalise()
}

type chatMessageOut struct {
	Role      string  `json:"role"`
	Content   *string `json:"content"`
	Refusal   *string `json:"refusal"`
	Reasoning string  `json:"reasoning_content"`
	ToolCalls []struct {
		Index    *int   `json:"index"`
		ID       string `json:"id"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func (m *chatMessageOut) streamParts(st *choiceTools) []llm.Part {
	out := m.contentParts()
	for _, tc := range m.ToolCalls {
		switch {
		case tc.Index != nil:
			st.cur = *tc.Index
		case tc.ID != "" && tc.ID == st.open[st.cur]:
		case tc.ID != "" || st.cur < 0:
			st.cur++
		}
		idx := st.cur
		if tc.ID != "" {
			st.open[idx] = tc.ID
		}
		out = append(out, &llm.ToolUse{
			ID: tc.ID, Name: tc.Function.Name,
			Args: json.RawMessage(tc.Function.Arguments), Index: &idx,
		})
	}
	return out
}

func (m *chatMessageOut) contentParts() []llm.Part {
	var out []llm.Part
	if m.Reasoning != "" {
		out = append(out, &llm.Thinking{Text: m.Reasoning})
	}
	if m.Content != nil && *m.Content != "" {
		out = append(out, &llm.Text{Text: *m.Content})
	}
	if m.Refusal != nil && *m.Refusal != "" {
		out = append(out, &llm.Refusal{Text: *m.Refusal})
	}
	return out
}

func (m *chatMessageOut) empty() bool {
	return m.Role == "" && m.Reasoning == "" && len(m.ToolCalls) == 0 &&
		(m.Content == nil || *m.Content == "") && (m.Refusal == nil || *m.Refusal == "")
}

func (m *chatMessageOut) parts() []llm.Part {
	out := m.contentParts()
	for i, tc := range m.ToolCalls {
		idx := i
		if tc.Index != nil {
			idx = *tc.Index
		}
		out = append(out, &llm.ToolUse{ID: tc.ID, Name: tc.Function.Name, Args: json.RawMessage(tc.Function.Arguments), Index: &idx})
	}
	return out
}

var chatResponseKnown = map[string]bool{
	"id": true, "model": true, "created": true, "object": true,
	"choices": true, "usage": true, "error": true, "service_tier": true,
}

type chatChoice struct {
	Index        int            `json:"index"`
	Message      chatMessageOut `json:"message"`
	Delta        chatMessageOut `json:"delta"`
	FinishReason *string        `json:"finish_reason"`

	raw json.RawMessage
}

func (c *chatChoice) UnmarshalJSON(b []byte) error {
	type alias chatChoice
	var a alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*c = chatChoice(a)
	c.raw = append(json.RawMessage(nil), b...)
	return nil
}

var chatChoiceKnown = map[string]bool{
	"index": true, "message": true, "delta": true, "finish_reason": true,
}

type chatResponse struct {
	ID          string       `json:"id"`
	Model       string       `json:"model"`
	Created     int64        `json:"created"`
	Choices     []chatChoice `json:"choices"`
	Usage       *chatUsage   `json:"usage"`
	ServiceTier string       `json:"service_tier"`
	Error       *struct {
		Message string        `json:"message"`
		Type    httpcore.Code `json:"type"`
		Code    httpcore.Code `json:"code"`
	} `json:"error"`
}

func (p *Provider) completeChat(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	body, diags, err := p.chatBody(a, false)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.endpoint("/chat/completions", p.opts.ChatPath), Body: body, Header: p.header(a), Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		return nil, httpcore.BodyError(ctx, p.ID(), rerr)
	}
	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "malformed response: " + err.Error(), Err: err}
	}
	if out.Error != nil {
		if len(out.Choices) == 0 {
			return nil, &llm.Error{
				Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Status: 200,
				Code: string(out.Error.Code), Message: out.Error.Message, Body: httpcore.ErrorBody(raw),
			}
		}
		// Some gateways fail partway and answer with both. The answer stands,
		// but the caller is told it is not the whole one. The note goes to the
		// ledger too, which may have been asked to hold none of the upstream's
		// words.
		detail := out.Error.Message
		if a.Redact {
			detail = "the backend reported an error beside its answer"
		}
		diags = append(diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageResponse,
			Severity: llm.SeverityLoss, Provider: p.ID(), Detail: detail,
		})
	}

	res := &llm.Response{
		ID: out.ID, Provider: p.ID(), Model: out.Model, Requested: a.Request.Model,
		Created: created(out.Created, p.now()), Diagnostics: diags,
		Extra: wirecore.Extras(raw, chatResponseKnown, a, openaichat.Name),
	}
	for _, ch := range out.Choices {
		stop := llm.StopNone
		if ch.FinishReason != nil {
			stop = decodeStop(*ch.FinishReason)
		}
		res.Choices = append(res.Choices, llm.Choice{
			Index:   ch.Index,
			Message: llm.Message{Role: llm.RoleAssistant, Content: ch.Message.parts()},
			Stop:    stop,
		})
		res.Diagnostics = append(res.Diagnostics, wirecore.Unmodelled(ch.raw, chatChoiceKnown, "choice", a)...)
	}
	if out.Usage != nil {
		u := out.Usage.neutral()
		u.Tier = out.ServiceTier
		res.Usage = u
	}
	return res, nil
}

func decodeStop(s string) llm.StopReason {
	switch s {
	case "stop":
		return llm.StopEnd
	case "length":
		return llm.StopLength
	case "tool_calls", "function_call":
		return llm.StopToolUse
	case "content_filter":
		return llm.StopContentFilter
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

func (p *Provider) streamChat(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	body, _, err := p.chatBody(a, true)
	if err != nil {
		return nil, err
	}
	h := p.header(a)
	h.Set("Accept", "text/event-stream")
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.endpoint("/chat/completions", p.opts.ChatPath), Body: body, Header: h, Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	sc := wirecore.NewScanner(resp.Body, a.Limits.MaxFrameBytes, a.Limits.MaxFrameBytes)
	st := &chatParser{ctx: ctx, provider: p.ID(), sc: sc, attempt: a}
	return llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	), nil
}

type chatParser struct {
	ctx      context.Context
	provider string
	sc       *wirecore.Scanner
	done     bool
	ended    bool
	queue    []*llm.Frame
	attempt  *llm.Attempt
	tools    map[int]*choiceTools
	beside   wirecore.Beside
}

// Which tool call is open belongs to the choice that opened it; with more than
// one candidate in flight their indices collide otherwise.
type choiceTools struct {
	cur  int
	open map[int]string
}

func (c *chatParser) toolFor(choice int) *choiceTools {
	if c.tools == nil {
		c.tools = map[int]*choiceTools{}
	}
	st, ok := c.tools[choice]
	if !ok {
		st = &choiceTools{cur: -1, open: map[int]string{}}
		c.tools[choice] = st
	}
	return st
}

func (c *chatParser) next() (*llm.Frame, error) {
	for {
		if len(c.queue) > 0 {
			f := c.queue[0]
			c.queue = c.queue[1:]
			return f, nil
		}
		if c.done {
			return nil, nil
		}
		if !c.sc.Scan() {
			if err := c.sc.Err(); err != nil {
				return nil, httpcore.ReadError(c.ctx, c.provider, err)
			}
			c.done = true
			if !c.ended {
				c.ended = true
				return &llm.Frame{TurnEnd: true}, nil
			}
			return nil, nil
		}
		data := bytes.TrimSpace(c.sc.Event().Data)
		if len(data) == 0 {
			continue
		}
		if string(data) == "[DONE]" {
			c.done = true
			c.ended = true
			return &llm.Frame{TurnEnd: true}, nil
		}
		var chunk chatResponse
		if err := json.Unmarshal(data, &chunk); err != nil {
			return nil, &llm.Error{
				Kind: llm.KindUpstream, Provider: c.provider, Sent: true, Code: llm.WireFrameMalformed,
				Message: "malformed stream frame: " + err.Error(), Err: err,
			}
		}
		if chunk.Error != nil {
			c.done = true
			return &llm.Frame{Err: &llm.Error{
				Kind: llm.KindUpstream, Provider: c.provider, Sent: true, Status: 200,
				Code: string(chunk.Error.Code), Message: chunk.Error.Message, Body: httpcore.ErrorBody(data),
			}}, nil
		}
		var frames []*llm.Frame
		for _, ch := range chunk.Choices {
			idx := ch.Index
			delta := ch.Delta
			if delta.empty() && !ch.Message.empty() {
				delta = ch.Message
			}
			f := &llm.Frame{
				ID: chunk.ID, Model: chunk.Model, Choice: &idx,
				Delta:       delta.streamParts(c.toolFor(ch.Index)),
				Diagnostics: c.beside.Note(ch.raw, chatChoiceKnown, "choice", c.attempt),
			}
			if delta.Role != "" {
				f.Role = llm.Role(delta.Role)
			}
			if ch.FinishReason != nil && *ch.FinishReason != "" {
				f.Stop = decodeStop(*ch.FinishReason)
			}
			frames = append(frames, f)
		}
		if len(frames) == 0 {
			frames = append(frames, &llm.Frame{ID: chunk.ID, Model: chunk.Model})
		}
		if chunk.Usage != nil {
			u := chunk.Usage.neutral()
			u.Tier = chunk.ServiceTier
			frames[0].Usage = &u
		}
		frames[0].Extra = c.extras(data)
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
		c.queue = append(c.queue, kept[1:]...)
		return kept[0], nil
	}
}

func (c *chatParser) extras(data []byte) map[string]json.RawMessage {
	return wirecore.Extras(data, chatChunkKnown, c.attempt, openaichat.Name)
}

var chatChunkKnown = map[string]bool{
	"id": true, "model": true, "created": true, "object": true,
	"choices": true, "usage": true, "error": true, "service_tier": true,
}

func caller(meta map[string]string) (string, string) {
	for _, key := range []string{"user", "user_id"} {
		if v := meta[key]; v != "" {
			return v, key
		}
	}
	return "", ""
}

func metaDropped(provider string, fields []string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
		Severity: llm.SeverityInfo, Provider: provider, Count: len(fields),
		Field:  strings.Join(fields, ","),
		Detail: "this backend names the end user and carries no other metadata, so the rest was not sent",
	}
}

func unsentMeta(meta map[string]string, carried string) []string {
	var rest []string
	for k := range meta {
		if k != carried {
			rest = append(rest, k)
		}
	}
	sort.Strings(rest)
	return rest
}
