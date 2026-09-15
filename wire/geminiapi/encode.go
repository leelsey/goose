// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"sort"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type candidate struct {
	Content      *content `json:"content,omitempty"`
	FinishReason string   `json:"finishReason,omitempty"`
	Index        int      `json:"index"`
}

type content struct {
	Role  string            `json:"role"`
	Parts []json.RawMessage `json:"parts"`
}

type outBody struct {
	ResponseID     string         `json:"responseId,omitempty"`
	Candidates     []candidate    `json:"candidates,omitempty"`
	PromptFeedback *promptFeed    `json:"promptFeedback,omitempty"`
	UsageMetadata  *usageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion   string         `json:"modelVersion,omitempty"`
}

type promptFeed struct {
	BlockReason string `json:"blockReason,omitempty"`
}

func (c *Codec) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	if resp == nil {
		return nil, nil, llm.Errorf(llm.KindInternal, "nil response")
	}
	var diags llm.Diagnostics
	body := outBody{ResponseID: resp.ID, UsageMetadata: encodeUsage(resp.Usage), ModelVersion: resp.Model}
	diags = append(diags, resp.Usage.Uncarried(Name, llm.FigureCacheRead, llm.FigureReasoning)...)
	for _, ch := range resp.Choices {
		parts, d := EncodeParts(ch.Message.Content)
		diags = append(diags, d...)
		name, clean := finishName(ch.Stop)
		if !clean {
			diags = append(diags, unknownStop(ch.Stop))
		}
		body.Candidates = append(body.Candidates, candidate{
			Content:      &content{Role: "model", Parts: list(parts)},
			FinishReason: name,
			Index:        ch.Index,
		})
	}
	if resp.Block != nil {
		body.PromptFeedback = &promptFeed{BlockReason: resp.Block.Reason}
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

func EncodeParts(list []llm.Part) ([]json.RawMessage, llm.Diagnostics) {
	var diags llm.Diagnostics
	out := []json.RawMessage{}
	for _, p := range list {
		switch v := p.(type) {
		case *llm.Text:
			out = append(out, mustJSON(map[string]any{"text": v.Text}))
		case *llm.Refusal:
			out = append(out, mustJSON(map[string]any{"text": v.Text}))
		case *llm.Thinking:
			if len(v.Redacted) > 0 {
				diags = append(diags, drop(llm.DropThinking, "redacted thinking has no part on this wire"))
				continue
			}
			part := map[string]any{"text": v.Text, "thought": true}
			if v.Signature != "" {
				part["thoughtSignature"] = v.Signature
			}
			out = append(out, mustJSON(part))
		case *llm.ToolUse:
			args := wirecore.ObjectArgs(v.Args)
			out = append(out, mustJSON(map[string]any{"functionCall": map[string]any{"name": v.Name, "args": args}}))
		case *llm.ToolResult:
			out = append(out, mustJSON(map[string]any{"functionResponse": map[string]any{
				"name": v.Name, "response": responseObject(v.Content),
			}}))
			diags = append(diags, llm.Flattened(v.Content, "",
				"this wire takes a tool result as an object of text, so what was not text did not travel")...)
		case *llm.Image:
			if len(v.Data) > 0 {
				out = append(out, mustJSON(map[string]any{"inlineData": map[string]any{
					"mimeType": v.MIME, "data": base64.StdEncoding.EncodeToString(v.Data),
				}}))
				continue
			}
			out = append(out, mustJSON(map[string]any{"fileData": map[string]any{"mimeType": v.MIME, "fileUri": v.URL}}))
		case *llm.Raw:
			if v.Vendor == Name && json.Valid(v.JSON) {
				out = append(out, v.JSON)
				continue
			}
			diags = append(diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
		}
	}
	return out, diags
}

func responseObject(parts []llm.Part) json.RawMessage {
	text := llm.TextOf(parts)
	if json.Valid([]byte(text)) && len(text) > 0 && (text[0] == '{') {
		return json.RawMessage(text)
	}
	return mustJSON(map[string]any{"result": text})
}

func (c *Codec) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	var diags llm.Diagnostics
	sse := wire.NewSSE(w)
	meta := s.Meta()
	if err := sse.Open(streamHeader(meta.Session)); err != nil {
		return diags, err
	}

	tools := map[int]map[int]*llm.ToolUse{}
	order := map[int][]int{}
	stop := map[int]llm.StopReason{}
	var candidates []int
	seen := func(choice int) {
		if _, ok := tools[choice]; ok {
			return
		}
		tools[choice] = map[int]*llm.ToolUse{}
		candidates = append(candidates, choice)
	}
	var block *llm.Block
	var streamErr error
	emitted := false

	send := func(b outBody, extra map[string]json.RawMessage) error {
		raw, err := json.Marshal(b)
		if err != nil {
			return err
		}
		if raw, err = wire.WithExtra(raw, extra); err != nil {
			return err
		}
		emitted = true
		return sse.Send("", raw)
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
		if fr.Block != nil {
			block = fr.Block
		}
		choice := fr.ChoiceIndex()
		seen(choice)
		var live []llm.Part
		for _, p := range fr.Delta {
			if t, ok := p.(*llm.ToolUse); ok {
				key := len(order[choice])
				if t.Index != nil {
					key = *t.Index
				}
				held, open := tools[choice][key]
				if !open {
					held = &llm.ToolUse{ID: t.ID, Name: t.Name}
					tools[choice][key] = held
					order[choice] = append(order[choice], key)
				}
				if t.Name != "" {
					held.Name = t.Name
				}
				held.Args = append(held.Args, t.Args...)
				continue
			}
			live = append(live, p)
		}
		if len(live) > 0 {
			parts, d := EncodeParts(live)
			diags = append(diags, d...)
			body := outBody{Candidates: []candidate{{Content: &content{Role: "model", Parts: list(parts)}, Index: choice}}}
			if err := send(body, fr.Extra); err != nil {
				return diags, err
			}
		}
		if fr.Stop != llm.StopNone {
			stop[choice] = fr.Stop
		}
	}

	if streamErr == nil {
		streamErr = s.Err()
	}

	final := outBody{ResponseID: meta.ID, UsageMetadata: encodeUsage(s.Usage()), ModelVersion: meta.Model}
	diags = append(diags, s.Usage().Uncarried(Name, llm.FigureCacheRead, llm.FigureReasoning)...)
	if len(candidates) == 0 {
		candidates = []int{0}
		seen(0)
	}
	sort.Ints(candidates)
	for _, choice := range candidates {
		var parts []json.RawMessage
		for _, key := range order[choice] {
			t := tools[choice][key]
			args := wirecore.ObjectArgs(t.Args)
			parts = append(parts, mustJSON(map[string]any{"functionCall": map[string]any{"name": t.Name, "args": args}}))
		}
		name, clean := finishName(stop[choice])
		if !clean {
			diags = append(diags, unknownStop(stop[choice]))
		}
		if len(parts) > 0 || name != "" || !emitted {
			final.Candidates = append(final.Candidates, candidate{
				Content: &content{Role: "model", Parts: list(parts)}, FinishReason: name, Index: choice,
			})
		}
	}
	if block != nil {
		final.PromptFeedback = &promptFeed{BlockReason: block.Reason}
	}
	if streamErr != nil {
		e := llm.AsError(streamErr)
		_ = sse.Send("", mustJSON(errorBody(e)))
		diags = append(diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Detail: e.Message,
		})
	}
	if err := send(final, nil); err != nil {
		return diags, err
	}
	diags = append(diags, s.Diagnostics()...)
	return diags, streamErr
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
	body, _ := json.Marshal(errorBody(e))
	h := jsonHeader("")
	wire.SetRetryAfter(h, e)
	return &wire.Outbound{Status: status, Header: h, Body: body}
}

func errorBody(e *llm.Error) map[string]any {
	status := e.Status
	if status == 0 || status < 400 {
		status = wire.StatusFor(e.Kind)
	}
	return map[string]any{"error": map[string]any{
		"code": status, "message": e.Message, "status": statusName(e.Kind),
	}}
}

func statusName(k llm.ErrorKind) string {
	switch k {
	case llm.KindInvalidRequest, llm.KindLoss, llm.KindUnsupported:
		return "INVALID_ARGUMENT"
	case llm.KindAuth:
		return "UNAUTHENTICATED"
	case llm.KindPermission:
		return "PERMISSION_DENIED"
	case llm.KindNotFound, llm.KindNoRoute:
		return "NOT_FOUND"
	case llm.KindRateLimit:
		return "RESOURCE_EXHAUSTED"
	case llm.KindOverloaded, llm.KindSaturated:
		return "UNAVAILABLE"
	case llm.KindTimeout:
		return "DEADLINE_EXCEEDED"
	case llm.KindCancelled:
		return "CANCELLED"
	default:
		return "INTERNAL"
	}
}

func unknownStop(s llm.StopReason) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Field: string(s),
		Detail: "stop reason has no spelling on this wire",
	}
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

func streamHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"text/event-stream"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(`{"text":""}`)
	}
	return b
}

func list(parts []json.RawMessage) []json.RawMessage {
	if parts == nil {
		return []json.RawMessage{}
	}
	return parts
}
