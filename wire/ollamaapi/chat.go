// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollamaapi

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type Chat struct{ opts Options }

func NewChat(o Options) *Chat { return &Chat{opts: o} }

func (c *Chat) Name() string { return NameChat }

func (c *Chat) Routes() []wire.Route {
	return []wire.Route{{Method: "POST", Pattern: "/api/chat"}}
}

func (c *Chat) Renders() wire.Renders {
	return wire.Renders{Thinking: true, ToolCalls: true, Candidates: 1}
}

var chatKnown = map[string]bool{
	"model": true, "messages": true, "stream": true, "format": true,
	"options": true, "tools": true, "think": true,
}

type chatMessage struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	Thinking  string   `json:"thinking,omitempty"`
	Images    []string `json:"images,omitempty"`
	ToolCalls []struct {
		Function struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls,omitempty"`
	ToolName string `json:"tool_name,omitempty"`
}

func (c *Chat) Decode(in *wire.Inbound) (*llm.Request, wire.Options, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	opts := wire.Options{Format: NameChat, Header: in.Header}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(in.Body, &top); err != nil {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "malformed JSON body: %v", err)
	}
	req := &llm.Request{}
	if err := field(top, "model", &req.Model); err != nil {
		return nil, opts, diags, err
	}
	opts.RequestModel = req.Model

	var msgs []chatMessage
	if err := field(top, "messages", &msgs); err != nil {
		return nil, opts, diags, err
	}
	if len(msgs) == 0 {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "messages is required")
	}
	calls, results := wirecore.NewIDMinter(), wirecore.NewIDMinter()
	for _, m := range msgs {
		msg := llm.Message{Role: llm.Role(m.Role)}
		if m.Role == "tool" {
			result := &llm.ToolResult{Name: m.ToolName, Content: []llm.Part{&llm.Text{Text: m.Content}}}
			if m.ToolName != "" {
				result.ID = results.Next(m.ToolName)
			}
			msg.Content = []llm.Part{result}
			req.Messages = append(req.Messages, msg)
			continue
		}
		if m.Thinking != "" {
			msg.Content = append(msg.Content, &llm.Thinking{Text: m.Thinking})
		}
		if m.Content != "" {
			msg.Content = append(msg.Content, &llm.Text{Text: m.Content})
		}
		msg.Content = append(msg.Content, decodeImages(m.Images)...)
		for i, tc := range m.ToolCalls {
			idx := i
			msg.Content = append(msg.Content, &llm.ToolUse{
				ID: calls.Next(tc.Function.Name), Name: tc.Function.Name,
				Args: unwrapArgs(tc.Function.Arguments), Index: &idx,
			})
		}
		req.Messages = append(req.Messages, msg)
	}

	// Ollama streams by default, so an absent field is not false.
	opts.Stream = true
	var stream *bool
	if err := field(top, "stream", &stream); err != nil {
		return nil, opts, diags, err
	}
	if stream != nil {
		opts.Stream = *stream
	}

	var o ollamaOptions
	if err := field(top, "options", &o); err != nil {
		return nil, opts, diags, err
	}
	diags = append(diags, o.apply(req, NameChat)...)
	for _, name := range wirecore.Shadowed(top["options"], modelledOptions) {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageDecode,
			Severity: llm.SeverityLoss, Format: NameChat, Field: "options." + name,
			Detail: "this option arrived spelled more than one way, and only the last of them was read",
		})
	}
	if rest := wirecore.NestedExtra(top["options"], modelledOptions); rest != nil {
		if req.Extra == nil {
			req.Extra = map[string]json.RawMessage{}
			req.ExtraFrom = NameChat
		}
		req.Extra["options"] = rest
	}
	req.ResponseFormat = decodeFormat(top["format"])

	tools, err := toolsFrom(top["tools"])
	if err != nil {
		return nil, opts, diags, err
	}
	req.Tools = tools

	thinking, effort, err := decodeThink(top["think"])
	if err != nil {
		return nil, opts, diags, err
	}
	req.Thinking, req.Effort = thinking, effort

	for k, v := range top {
		if chatKnown[k] {
			continue
		}
		if req.Extra == nil {
			req.Extra = map[string]json.RawMessage{}
			req.ExtraFrom = NameChat
		}
		req.Extra[k] = v
	}
	opts.Key = bearer(in.Get("Authorization"))
	return req, opts, diags, nil
}

type chatBody struct {
	Model     string      `json:"model"`
	CreatedAt string      `json:"created_at,omitempty"`
	Message   *outMessage `json:"message,omitempty"`
	Done      bool        `json:"done"`
	Reason    string      `json:"done_reason,omitempty"`
	wireUsage
}

type outMessage struct {
	Role      string        `json:"role"`
	Content   string        `json:"content"`
	Thinking  string        `json:"thinking,omitempty"`
	Images    []string      `json:"images,omitempty"`
	ToolCalls []outToolCall `json:"tool_calls,omitempty"`
}

type outToolCall struct {
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

func (b chatBody) MarshalJSON() ([]byte, error) {
	obj := map[string]any{"model": b.Model, "done": b.Done}
	if b.CreatedAt != "" {
		obj["created_at"] = b.CreatedAt
	}
	if b.Message != nil {
		obj["message"] = b.Message
	}
	if b.Reason != "" {
		obj["done_reason"] = b.Reason
	}
	addUsage(obj, b.wireUsage)
	return json.Marshal(obj)
}

func addUsage(obj map[string]any, u wireUsage) {
	if u.TotalDuration != 0 {
		obj["total_duration"] = u.TotalDuration
	}
	if u.PromptEvalCount != 0 {
		obj["prompt_eval_count"] = u.PromptEvalCount
	}
	if u.EvalCount != 0 {
		obj["eval_count"] = u.EvalCount
	}
}

func (c *Chat) message(parts []llm.Part, format string) (*outMessage, llm.Diagnostics) {
	var diags llm.Diagnostics
	out := &outMessage{Role: "assistant"}
	var refusal string
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text:
			out.Content += v.Text
		case *llm.Thinking:
			out.Thinking += v.Text
			if len(v.Redacted) > 0 {
				diags = append(diags, drop(format, llm.DropThinking, "redacted thinking has no field on this wire"))
			}
		case *llm.Refusal:
			refusal = v.Text
		case *llm.ToolUse:
			tc := outToolCall{}
			tc.Function.Name = v.Name
			tc.Function.Arguments = argsObject(v.Args)
			out.ToolCalls = append(out.ToolCalls, tc)
		case *llm.Image:
			diags = append(diags, drop(format, llm.DropImage, "an assistant image has no field on this wire"))
		case *llm.ToolResult:
			diags = append(diags, drop(format, llm.DropToolResult, "a tool result cannot ride on an assistant message"))
		case *llm.Raw:
			diags = append(diags, drop(format, llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
		}
	}
	if refusal != "" {
		if out.Content != "" {
			out.Content += "\n\n"
		}
		out.Content += refusal
	}
	return out, diags
}

func argsObject(raw json.RawMessage) json.RawMessage {
	return wirecore.ObjectArgs(raw)
}

func (c *Chat) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	if resp == nil {
		return nil, nil, llm.Errorf(llm.KindInternal, "nil response")
	}
	body := chatBody{Model: modelName(resp, opts), CreatedAt: stamp(resp.Created), Done: true, wireUsage: encodeUsage(resp.Usage)}
	body.Message = &outMessage{Role: "assistant"}
	var diags llm.Diagnostics
	diags = append(diags, resp.Usage.Uncarried(NameChat)...)
	stop := resp.Stop
	if ch := resp.First(); ch != nil {
		stop = ch.Stop
		msg, d := c.message(ch.Message.Content, NameChat)
		diags = append(diags, d...)
		body.Message = msg
	}
	if len(resp.Choices) > 1 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropCandidate, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityFatal, Format: NameChat, Count: len(resp.Choices) - 1,
			Detail: "this wire carries one candidate",
		})
	}
	reason, clean := doneReason(stop)
	if !clean {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: NameChat, Field: string(stop),
			Detail: "stop reason has no spelling on this wire",
		})
	}
	body.Reason = reason
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

func (c *Chat) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	var diags llm.Diagnostics
	nd := wire.NewNDJSON(w)
	meta := s.Meta()
	model := meta.Model
	if model == "" {
		model = opts.RequestModel
	}
	if err := nd.Open(ndjsonHeader(meta.Session)); err != nil {
		return diags, err
	}

	stop := llm.StopNone
	sentContent, refusing := false, false
	var pendingTools []*llm.ToolUse
	merger := wirecore.NewToolMerger()
	var streamErr error
	// An unmodelled field rides one chunk: the one its own frame produced, or
	// the next chunk out where that frame produced none. Repeating it on every
	// later chunk re-sends the upstream's bytes per delta; dropping it loses
	// what only that frame carried.
	var held map[string]json.RawMessage
	send := func(b chatBody, extra map[string]json.RawMessage) error {
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
		raw, err := wire.WithExtra(mustJSON(b), extra)
		if err != nil {
			return err
		}
		return nd.Send(raw)
	}

	for fr := range s.Frames() {
		if err := ctx.Err(); err != nil {
			streamErr = err
			break
		}
		if fr.Err != nil {
			streamErr = fr.Err
			break
		}
		msg := &outMessage{Role: "assistant"}
		emit := false
		for _, p := range fr.Delta {
			switch v := p.(type) {
			case *llm.Text:
				msg.Content += v.Text
				sentContent = sentContent || v.Text != ""
				emit = true
			case *llm.Thinking:
				msg.Thinking += v.Text
				emit = emit || v.Text != ""
			case *llm.Refusal:
				// The whole-answer path parts a refusal from the text before it,
				// and a reader accumulating these chunks must arrive at the same
				// words, whether or not a frame boundary falls between them.
				if !refusing {
					refusing = true
					if sentContent {
						msg.Content += "\n\n"
					}
				}
				msg.Content += v.Text
				sentContent = sentContent || v.Text != ""
				emit = true
			case *llm.ToolUse:
				merger.Merge(v, nil)
				pendingTools = merger.Tools()
			case *llm.Image:
				diags = append(diags, drop(NameChat, llm.DropImage, "an assistant image has no field on this wire"))
			case *llm.Raw:
				diags = append(diags, drop(NameChat, llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
			}
		}
		if emit {
			if err := send(chatBody{Model: model, CreatedAt: stamp(c.opts.now()), Message: msg}, fr.Extra); err != nil {
				return diags, err
			}
		} else if len(fr.Extra) > 0 {
			held = fr.Extra
		}
		if fr.Stop != llm.StopNone {
			stop = fr.Stop
		}
	}

	if streamErr == nil {
		streamErr = s.Err()
	}

	final := chatBody{Model: model, CreatedAt: stamp(c.opts.now()), Done: true, wireUsage: encodeUsage(s.Usage())}
	diags = append(diags, s.Usage().Uncarried(NameChat)...)
	last := &outMessage{Role: "assistant"}
	for _, t := range pendingTools {
		tc := outToolCall{}
		tc.Function.Name = t.Name
		tc.Function.Arguments = argsObject(t.Args)
		last.ToolCalls = append(last.ToolCalls, tc)
	}
	final.Message = last
	reason, clean := doneReason(stop)
	if !clean {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: NameChat, Field: string(stop),
			Detail: "stop reason has no spelling on this wire",
		})
	}
	final.Reason = reason
	if streamErr != nil {
		e := llm.AsError(streamErr)
		final.Reason = "error"
		if err := nd.Send(mustJSON(map[string]any{"error": e.Message})); err != nil {
			return diags, err
		}
		diags = append(diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: NameChat, Detail: e.Message,
		})
	}
	if merger.Overflowed() {
		diags = append(diags, llm.Diagnostic{
			Code: llm.WireBufferFull, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: NameChat,
			Detail: "the tool arguments outgrew what one turn may hold, so the call is short of what was streamed",
		})
	}
	if err := send(final, nil); err != nil {
		return diags, err
	}
	diags = append(diags, s.Diagnostics()...)
	return diags, streamErr
}

func (c *Chat) EncodeError(err *llm.Error, _ wire.Options) *wire.Outbound {
	e := err
	if e == nil {
		e = llm.Errorf(llm.KindInternal, "unknown error")
	}
	status := e.Status
	if status == 0 || status < 400 {
		status = wire.StatusFor(e.Kind)
	}
	body, _ := json.Marshal(map[string]any{"error": e.Message})
	h := jsonHeader("")
	wire.SetRetryAfter(h, e)
	return &wire.Outbound{Status: status, Header: h, Body: body}
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

func ndjsonHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"application/x-ndjson"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func jsonHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"application/json"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		return []byte(`{"error":"encode failed"}`)
	}
	return b
}
