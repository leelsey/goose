// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type outUsage struct {
	Prompt     int `json:"prompt_tokens"`
	Completion int `json:"completion_tokens"`
	Total      int `json:"total_tokens"`

	PromptDetails *struct {
		Cached     int `json:"cached_tokens,omitempty"`
		CacheWrite int `json:"cache_write_tokens,omitempty"`
	} `json:"prompt_tokens_details,omitempty"`
	CompletionDetails *struct {
		Reasoning int `json:"reasoning_tokens,omitempty"`
	} `json:"completion_tokens_details,omitempty"`
}

func encodeUsage(u llm.Usage) *outUsage {
	out := &outUsage{Prompt: u.Prompt, Completion: u.Completion, Total: u.Total}
	if out.Total == 0 {
		out.Total = u.Prompt + u.Completion
	}
	if u.CacheRead != 0 || u.CacheWrite != 0 {
		out.PromptDetails = &struct {
			Cached     int `json:"cached_tokens,omitempty"`
			CacheWrite int `json:"cache_write_tokens,omitempty"`
		}{Cached: u.CacheRead, CacheWrite: u.CacheWrite}
	}
	if u.Reasoning != 0 {
		out.CompletionDetails = &struct {
			Reasoning int `json:"reasoning_tokens,omitempty"`
		}{Reasoning: u.Reasoning}
	}
	return out
}

type outToolCall struct {
	Index    *int   `json:"index,omitempty"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments,omitempty"`
	} `json:"function"`
}

type outMessage struct {
	Role           string
	Content        *string
	Refusal        *string
	Reasoning      string
	ReasoningField string
	ToolCalls      []outToolCall
	Delta          bool
}

func (m outMessage) MarshalJSON() ([]byte, error) {
	obj := map[string]any{}
	if m.Role != "" {
		obj["role"] = m.Role
	}
	switch {
	case m.Content != nil:
		obj["content"] = *m.Content
	case !m.Delta:
		obj["content"] = nil
	}
	if m.Refusal != nil {
		obj["refusal"] = *m.Refusal
	}
	if m.Reasoning != "" && m.ReasoningField != "" {
		obj[m.ReasoningField] = m.Reasoning
	}
	if len(m.ToolCalls) > 0 {
		obj["tool_calls"] = m.ToolCalls
	}
	return json.Marshal(obj)
}

type outChoice struct {
	Index        int         `json:"index"`
	Message      *outMessage `json:"message,omitempty"`
	Delta        *outMessage `json:"delta,omitempty"`
	FinishReason *string     `json:"finish_reason"`
}

type outBody struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []outChoice `json:"choices"`
	Usage   *outUsage   `json:"usage,omitempty"`
	Tier    string      `json:"service_tier,omitempty"`
}

func (c *Codec) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	if resp == nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "nil response")
	}
	body := outBody{
		ID:      c.id(resp.ID),
		Object:  "chat.completion",
		Created: created(resp.Created),
		Model:   modelName(resp, opts),
		Choices: []outChoice{},
		Usage:   encodeUsage(resp.Usage),
		Tier:    resp.Usage.Tier,
	}
	diags = append(diags, resp.Usage.Uncarried(Name, llm.FigureCacheRead, llm.FigureCacheWrite, llm.FigureReasoning, llm.FigureTier)...)

	for _, ch := range resp.Choices {
		msg, d := c.message(ch.Message, resp)
		diags = append(diags, d...)
		reason, clean := finishReason(ch.Stop)
		if !clean {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
				Severity: llm.SeverityLoss, Format: Name, Field: string(ch.Stop),
				Detail: "stop reason has no spelling on this wire and was reported as stop",
			})
		}
		out := outChoice{Index: ch.Index, Message: msg}
		if reason != "" {
			r := reason
			out.FinishReason = &r
		}
		body.Choices = append(body.Choices, out)
	}

	if len(resp.Choices) == 0 && resp.Block != nil {
		reason := "content_filter"
		empty := ""
		body.Choices = append(body.Choices, outChoice{Index: 0, Message: &outMessage{Role: "assistant", Content: &empty}, FinishReason: &reason})
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropCandidate, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityInfo, Format: Name, Field: resp.Block.Reason,
			Detail: "prompt-level block rendered as a content-filtered choice",
		})
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode: %v", err)
	}
	raw, err = wire.WithExtra(raw, resp.Extra)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode: %v", err)
	}
	return &wire.Outbound{Status: http.StatusOK, Header: jsonHeader(resp.Session), Body: raw}, diags, nil
}

func (c *Codec) message(m llm.Message, resp *llm.Response) (*outMessage, llm.Diagnostics) {
	var diags llm.Diagnostics
	out := &outMessage{Role: string(m.Role), ReasoningField: c.opts.ReasoningField}
	if out.Role == "" {
		out.Role = "assistant"
	}
	var text, thinking string
	for _, p := range m.Content {
		switch v := p.(type) {
		case *llm.Text:
			text += v.Text
		case *llm.Refusal:
			r := v.Text
			out.Refusal = &r
		case *llm.Thinking:
			thinking += v.Text
		case *llm.ToolUse:
			tc := outToolCall{ID: v.ID, Type: "function"}
			tc.Function.Name = v.Name
			tc.Function.Arguments = argsString(v.Args)
			out.ToolCalls = append(out.ToolCalls, tc)
		case *llm.Image:
			diags = append(diags, drop(llm.DropImage, "an image has no place in a chat completion message"))
		case *llm.ToolResult:
			diags = append(diags, drop(llm.DropToolResult, "a tool result cannot ride on an assistant message"))
		case *llm.Raw:
			diags = append(diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" is not renderable here"))
		}
	}
	if thinking != "" {
		if c.opts.ReasoningField != "" {
			out.Reasoning = thinking
		} else {
			diags = append(diags, drop(llm.DropThinking, "this wire has no reasoning field; set Options.ReasoningField to carry it"))
		}
	}
	out.Content = &text
	if text == "" && (out.Refusal != nil || len(out.ToolCalls) > 0) {
		out.Content = nil
	}
	return out, diags
}

func errorCode(code string) any {
	if code == "" {
		return nil
	}
	return code
}

func (c *Codec) EncodeError(err *llm.Error, _ wire.Options) *wire.Outbound {
	e := err
	if e == nil {
		e = llm.Errorf(llm.KindInternal, "unknown error")
	}
	status := e.Status
	if status == 0 || status < 400 {
		status = wire.StatusFor(e.Kind)
	}
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": e.Message,
			"type":    string(e.Kind),
			"code":    errorCode(e.Code),
		},
	})
	h := jsonHeader("")
	wire.SetRetryAfter(h, e)
	return &wire.Outbound{Status: status, Header: h, Body: body}
}

func (c *Codec) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	var diags llm.Diagnostics
	sse := wire.NewSSE(w)
	meta := s.Meta()
	id := c.id(meta.ID)
	model := meta.Model
	if model == "" {
		model = opts.RequestModel
	}
	created := created(meta.Created)

	done := false
	finish := func() error {
		if done {
			return nil
		}
		done = true
		return sse.Send("", []byte("[DONE]"))
	}

	if err := sse.Open(jsonStreamHeader(meta.Session)); err != nil {
		return diags, err
	}

	tier := ""
	// An unmodelled field rides one chunk: the one its own frame produced, or
	// the next chunk out where that frame produced none. Latching it and
	// repeating it on every later chunk re-sends the upstream's own bytes once
	// per delta; dropping it loses what only that frame carried.
	var held map[string]json.RawMessage
	send := func(ch outChoice, usage *outUsage, extra map[string]json.RawMessage) error {
		if len(held) > 0 {
			if extra == nil {
				extra = held
			} else {
				merged := map[string]json.RawMessage{}
				for k, v := range held {
					merged[k] = v
				}
				for k, v := range extra {
					merged[k] = v
				}
				extra = merged
			}
			held = nil
		}
		body := outBody{ID: id, Object: "chat.completion.chunk", Created: created, Model: model, Choices: []outChoice{ch}, Usage: usage, Tier: tier}
		if ch.Delta == nil && ch.FinishReason == nil && usage != nil {
			body.Choices = []outChoice{}
		}
		raw, err := json.Marshal(body)
		if err != nil {
			return err
		}
		if raw, err = wire.WithExtra(raw, extra); err != nil {
			return err
		}
		return sse.Send("", raw)
	}

	roleSent := map[int]bool{}
	var streamErr error

	for fr := range s.Frames() {
		if err := ctx.Err(); err != nil {
			streamErr = err
			break
		}
		idx := fr.ChoiceIndex()
		sent := false
		if fr.Err != nil {
			streamErr = fr.Err
			break
		}
		if !roleSent[idx] {
			roleSent[idx] = true
			role := string(fr.Role)
			if role == "" {
				role = "assistant"
			}
			if err := send(outChoice{Index: idx, Delta: &outMessage{Role: role, Delta: true}}, nil, fr.Extra); err != nil {
				return diags, err
			}
			sent = true
		}
		delta, d := c.delta(fr.Delta)
		diags = append(diags, d...)
		if delta != nil {
			if err := send(outChoice{Index: idx, Delta: delta}, nil, fr.Extra); err != nil {
				return diags, err
			}
			sent = true
		}
		if fr.Stop != llm.StopNone {
			reason, clean := finishReason(fr.Stop)
			if !clean {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
					Severity: llm.SeverityLoss, Format: Name, Field: string(fr.Stop),
					Detail: "stop reason has no spelling on this wire and was reported as stop",
				})
			}
			r := reason
			if err := send(outChoice{Index: idx, Delta: &outMessage{Delta: true}, FinishReason: &r}, nil, fr.Extra); err != nil {
				return diags, err
			}
			sent = true
		}
		if !sent && len(fr.Extra) > 0 {
			// Two frames carrying nothing but fields this wire does not model
			// are two sets of them. The later is not a replacement for the
			// earlier, and the map is goose's own so the frame's is not written
			// through.
			if held == nil {
				held = make(map[string]json.RawMessage, len(fr.Extra))
			}
			for k, v := range fr.Extra {
				held[k] = v
			}
		}
	}

	if streamErr == nil {
		streamErr = s.Err()
	}
	if streamErr != nil {
		e := llm.AsError(streamErr)
		raw, _ := json.Marshal(map[string]any{"error": map[string]any{"message": e.Message, "type": string(e.Kind), "code": errorCode(e.Code)}})
		_ = sse.Send("", raw)
		diags = append(diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Detail: e.Message,
		})
	}

	if opts.StreamUsage {
		u := s.Usage()
		tier = u.Tier
		if err := send(outChoice{Index: 0}, encodeUsage(u), nil); err != nil {
			return diags, err
		}
		diags = append(diags, u.Uncarried(Name, llm.FigureCacheRead, llm.FigureCacheWrite, llm.FigureReasoning, llm.FigureTier)...)
	}
	if err := finish(); err != nil {
		return diags, err
	}
	diags = append(diags, s.Diagnostics()...)
	return diags, streamErr
}

func (c *Codec) delta(parts []llm.Part) (*outMessage, llm.Diagnostics) {
	if len(parts) == 0 {
		return nil, nil
	}
	var diags llm.Diagnostics
	out := &outMessage{Delta: true, ReasoningField: c.opts.ReasoningField}
	var text, thinking string
	any := false
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text:
			text += v.Text
			any = true
		case *llm.Refusal:
			r := v.Text
			out.Refusal = &r
			any = true
		case *llm.Thinking:
			thinking += v.Text
		case *llm.ToolUse:
			tc := outToolCall{Index: v.Index, ID: v.ID, Type: "function"}
			tc.Function.Name = v.Name
			tc.Function.Arguments = string(v.Args)
			out.ToolCalls = append(out.ToolCalls, tc)
			any = true
		case *llm.Image:
			diags = append(diags, drop(llm.DropImage, "an image cannot be streamed on this wire"))
		case *llm.ToolResult:
			diags = append(diags, drop(llm.DropToolResult, "a tool result cannot be streamed on this wire"))
		case *llm.Raw:
			diags = append(diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" is not renderable here"))
		}
	}
	if thinking != "" {
		if c.opts.ReasoningField != "" {
			out.Reasoning = thinking
			any = true
		} else {
			diags = append(diags, drop(llm.DropThinking, "this wire has no reasoning field; set Options.ReasoningField to carry it"))
		}
	}
	if text != "" {
		out.Content = &text
	}
	if !any {
		return nil, diags
	}
	return out, diags
}

func (c *Codec) id(id string) string {
	if id != "" {
		return id
	}
	if c.opts.IDSource != nil {
		return c.opts.IDPrefix + c.opts.IDSource()
	}
	return c.opts.IDPrefix + "goose"
}

func modelName(resp *llm.Response, opts wire.Options) string {
	if opts.RequestModel != "" {
		return opts.RequestModel
	}
	if resp.Requested != "" {
		return resp.Requested
	}
	return resp.Model
}

func created(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.Unix()
}

func argsString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "{}"
	}
	return string(raw)
}

func drop(code, detail string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: code, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Detail: detail,
	}
}

func jsonHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"application/json"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func jsonStreamHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"text/event-stream"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}
