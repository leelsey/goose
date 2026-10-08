// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type outBody struct {
	ID           string            `json:"id"`
	Type         string            `json:"type"`
	Role         string            `json:"role"`
	Model        string            `json:"model"`
	Content      []json.RawMessage `json:"content"`
	StopReason   *string           `json:"stop_reason"`
	StopSequence *string           `json:"stop_sequence"`
	Usage        wireUsage         `json:"usage"`
}

func (c *Codec) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	if resp == nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "nil response")
	}
	body := outBody{
		ID:      c.id(resp.ID),
		Type:    "message",
		Role:    "assistant",
		Model:   modelName(resp, opts),
		Content: []json.RawMessage{},
		Usage:   encodeUsage(resp.Usage),
	}
	diags = append(diags, resp.Usage.Uncarried(Name, llm.FigureCacheRead, llm.FigureCacheWrite, llm.FigureTier)...)

	stop := resp.Stop
	if ch := resp.First(); ch != nil {
		stop = ch.Stop
		if ch.StopSequence != "" {
			seq := ch.StopSequence
			body.StopSequence = &seq
		}
		blocks, d := encodeBlocks(ch.Message.Content, true)
		diags = append(diags, d...)
		body.Content = blocks
	}
	if len(resp.Choices) > 1 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropCandidate, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityFatal, Format: Name, Count: len(resp.Choices) - 1,
			Detail: "this wire carries one candidate; the rest cannot be delivered",
		})
	}
	if resp.Block != nil && len(body.Content) == 0 {
		stop = llm.StopRefusal
	}
	if name, clean := stopName(stop); name != "" {
		if !clean {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
				Severity: llm.SeverityLoss, Format: Name, Field: string(stop),
				Detail: "stop reason has no spelling on this wire",
			})
		}
		body.StopReason = &name
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode: %v", err)
	}
	raw, err = wire.WithExtra(raw, resp.Extra)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode: %v", err)
	}
	return &wire.Outbound{Status: http.StatusOK, Header: header(resp.Session, "application/json"), Body: raw}, diags, nil
}

func imageSource(v *llm.Image) (map[string]any, bool) {
	switch {
	case len(v.Data) > 0:
		// This wire requires a media type. Where the caller's own wire had no
		// field for one, the bytes say what they are.
		mime := v.MIME
		if mime == "" {
			mime, _, _ = strings.Cut(http.DetectContentType(v.Data), ";")
			if !strings.HasPrefix(mime, "image/") {
				mime = "image/png"
			}
		}
		return map[string]any{
			"type": "base64", "media_type": mime,
			"data": base64.StdEncoding.EncodeToString(v.Data),
		}, true
	case v.URL != "":
		return map[string]any{"type": "url", "url": v.URL}, true
	default:
		return nil, false
	}
}

// EncodeBlocks writes blocks for a backend on this wire, each signature exactly
// as it was issued.
func EncodeBlocks(parts []llm.Part) ([]json.RawMessage, llm.Diagnostics) {
	return encodeBlocks(parts, false)
}

// encodeBlocks writes blocks for a backend or, with client set, for a client,
// who is handed another family's signature marked as that family's.
func encodeBlocks(parts []llm.Part, client bool) ([]json.RawMessage, llm.Diagnostics) {
	var diags llm.Diagnostics
	out := []json.RawMessage{}
	unsigned := 0
	for _, p := range parts {
		if sig, _ := wirecore.Signed(p); sig != "" && p.Kind() != llm.KindThinking {
			unsigned++
		}
		switch v := p.(type) {
		case *llm.Text:
			// This wire refuses an empty text block, and one arrives from
			// Gemini carrying nothing but a signature this wire has no field for.
			if v.Text == "" {
				continue
			}
			out = append(out, mustJSON(map[string]any{"type": "text", "text": v.Text}))
		case *llm.Refusal:
			// This wire has no refusal block, and a refusal read as an answer
			// is an answer the client will act on.
			diags = append(diags, refusalAsText())
			out = append(out, mustJSON(map[string]any{"type": "text", "text": v.Text}))
		case *llm.Thinking:
			if len(v.Redacted) > 0 {
				out = append(out, mustJSON(map[string]any{"type": "redacted_thinking", "data": string(v.Redacted)}))
				continue
			}
			sig := v.Signature
			if client {
				sig = wirecore.SealSignature(llm.FamilyAnthropic, v.Signer, sig)
			}
			out = append(out, mustJSON(map[string]any{"type": "thinking", "thinking": v.Text, "signature": sig}))
		case *llm.ToolUse:
			args, broken := wirecore.ObjectArgsBroken(v.Args)
			if broken {
				diags = append(diags, brokenArgs(v.Name))
			}
			out = append(out, mustJSON(map[string]any{"type": "tool_use", "id": v.ID, "name": v.Name, "input": args}))
		case *llm.ToolResult:
			inner, d := encodeBlocks(v.Content, client)
			diags = append(diags, d...)
			out = append(out, mustJSON(map[string]any{"type": "tool_result", "tool_use_id": v.ID, "content": inner, "is_error": v.IsError}))
		case *llm.Image:
			// A request carries images; an answer on this wire has no block for
			// one, which is what Renders says and what the stream does.
			if client {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageEncode,
					Severity: llm.SeverityLoss, Format: Name,
					Detail: "an assistant image has no block on this wire",
				})
				continue
			}
			source, ok := imageSource(v)
			if !ok {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageEncode,
					Severity: llm.SeverityLoss, Format: Name,
					Detail: "the image carried neither bytes nor a URL, so there was nothing to send",
				})
				continue
			}
			out = append(out, mustJSON(map[string]any{"type": "image", "source": source}))
		case *llm.Raw:
			if v.Vendor == Name && json.Valid(v.JSON) {
				out = append(out, v.JSON)
				continue
			}
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
				Severity: llm.SeverityLoss, Format: Name, Field: v.Type,
				Detail: "content block from " + v.Vendor + " has no shape on this wire",
			})
		}
	}
	if unsigned > 0 {
		diags = append(diags, unsignedParts(unsigned))
	}
	return out, diags
}

// unsignedParts says Gemini's signatures on calls, text or images did not
// travel: this wire signs only a thinking block, and the model's reasoning
// before those parts is lost to the next turn.
func unsignedParts(n int) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropSignature, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Count: n,
		Detail: "this wire signs only a thinking block, so the reasoning signed onto a call, a text or an image did not travel with it",
	}
}

func refusalAsText() llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropRefusal, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name,
		Detail: "this format has no field for a refusal, so it travelled as ordinary text",
	}
}

func brokenArgs(name string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropToolCall, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Field: name,
		Detail: "this call's arguments did not survive, so it travelled with none rather than with what was meant",
	}
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
		"type":  "error",
		"error": map[string]any{"type": errorType(e.Kind), "message": e.Message},
	})
	h := header("", "application/json")
	wire.SetRetryAfter(h, e)
	return &wire.Outbound{Status: status, Header: h, Body: body}
}

func errorType(k llm.ErrorKind) string {
	switch k {
	case llm.KindInvalidRequest, llm.KindLoss, llm.KindUnsupported:
		return "invalid_request_error"
	case llm.KindAuth:
		return "authentication_error"
	case llm.KindPermission:
		return "permission_error"
	case llm.KindNotFound, llm.KindNoRoute:
		return "not_found_error"
	case llm.KindRateLimit:
		return "rate_limit_error"
	case llm.KindOverloaded, llm.KindSaturated:
		return "overloaded_error"
	default:
		return "api_error"
	}
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

func header(session, contentType string) http.Header {
	h := http.Header{"Content-Type": []string{contentType}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(`{"type":"text","text":""}`)
	}
	return b
}
