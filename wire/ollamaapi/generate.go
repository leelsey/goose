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

type Generate struct{ opts Options }

func NewGenerate(o Options) *Generate { return &Generate{opts: o} }

func (g *Generate) Name() string { return NameGenerate }

func (g *Generate) Routes() []wire.Route {
	return []wire.Route{{Method: "POST", Pattern: "/api/generate"}}
}

func (g *Generate) Renders() wire.Renders {
	return wire.Renders{Thinking: true, ToolCalls: false, Candidates: 1}
}

var generateKnown = map[string]bool{
	"model": true, "prompt": true, "system": true, "stream": true,
	"format": true, "options": true, "images": true, "think": true,
}

func (g *Generate) Decode(in *wire.Inbound) (*llm.Request, wire.Options, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	opts := wire.Options{Format: NameGenerate, Header: in.Header}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(in.Body, &top); err != nil {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "malformed JSON body: %v", err)
	}
	req := &llm.Request{}
	if err := field(top, "model", &req.Model); err != nil {
		return nil, opts, diags, err
	}
	opts.RequestModel = req.Model

	var prompt, system string
	var images []string
	if err := field(top, "prompt", &prompt); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "system", &system); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "images", &images); err != nil {
		return nil, opts, diags, err
	}
	if system != "" {
		req.System = []llm.Part{&llm.Text{Text: system}}
	}
	user := llm.Message{Role: llm.RoleUser}
	if prompt != "" {
		user.Content = append(user.Content, &llm.Text{Text: prompt})
	}
	user.Content = append(user.Content, decodeImages(images)...)
	if len(user.Content) == 0 {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "prompt is required")
	}
	req.Messages = []llm.Message{user}

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
	diags = append(diags, o.apply(req, NameGenerate)...)
	if rest := wirecore.NestedExtra(top["options"], modelledOptions); rest != nil {
		if req.Extra == nil {
			req.Extra = map[string]json.RawMessage{}
			req.ExtraFrom = NameGenerate
		}
		req.Extra["options"] = rest
	}
	req.ResponseFormat = decodeFormat(top["format"])

	thinking, effort, err := decodeThink(top["think"])
	if err != nil {
		return nil, opts, diags, err
	}
	req.Thinking, req.Effort = thinking, effort

	for k, v := range top {
		if generateKnown[k] {
			continue
		}
		if req.Extra == nil {
			req.Extra = map[string]json.RawMessage{}
			req.ExtraFrom = NameGenerate
		}
		req.Extra[k] = v
	}
	opts.Key = bearer(in.Get("Authorization"))
	return req, opts, diags, nil
}

type generateBody struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at,omitempty"`
	Response  string `json:"response"`
	Thinking  string `json:"thinking,omitempty"`
	Done      bool   `json:"done"`
	Reason    string `json:"done_reason,omitempty"`
	wireUsage
}

func (b generateBody) MarshalJSON() ([]byte, error) {
	obj := map[string]any{"model": b.Model, "response": b.Response, "done": b.Done}
	if b.CreatedAt != "" {
		obj["created_at"] = b.CreatedAt
	}
	if b.Thinking != "" {
		obj["thinking"] = b.Thinking
	}
	if b.Reason != "" {
		obj["done_reason"] = b.Reason
	}
	addUsage(obj, b.wireUsage)
	return json.Marshal(obj)
}

func (g *Generate) render(parts []llm.Part) (text, thinking string, diags llm.Diagnostics) {
	var refusal string
	tools := 0
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text:
			text += v.Text
		case *llm.Thinking:
			thinking += v.Text
		case *llm.Refusal:
			refusal = v.Text
		case *llm.ToolUse:
			tools++
		case *llm.Image:
			diags = append(diags, drop(NameGenerate, llm.DropImage, "an assistant image has no field on this wire"))
		case *llm.ToolResult:
			diags = append(diags, drop(NameGenerate, llm.DropToolResult, "a tool result has no field on this wire"))
		case *llm.Raw:
			diags = append(diags, drop(NameGenerate, llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
		}
	}
	if refusal != "" {
		if text != "" {
			text += "\n\n"
		}
		text += refusal
	}
	if tools > 0 {
		d := drop(NameGenerate, llm.DropToolCall, "this endpoint has no tool_calls field; use /api/chat")
		d.Count = tools
		if text == "" {
			d.Severity = llm.SeverityFatal
		}
		diags = append(diags, d)
	}
	return text, thinking, diags
}

func (g *Generate) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	if resp == nil {
		return nil, nil, llm.Errorf(llm.KindInternal, "nil response")
	}
	body := generateBody{Model: modelName(resp, opts), CreatedAt: stamp(resp.Created), Done: true, wireUsage: encodeUsage(resp.Usage)}
	var diags llm.Diagnostics
	diags = append(diags, resp.Usage.Uncarried(NameGenerate)...)
	stop := resp.Stop
	if ch := resp.First(); ch != nil {
		stop = ch.Stop
		text, thinking, d := g.render(ch.Message.Content)
		diags = append(diags, d...)
		body.Response, body.Thinking = text, thinking
	}
	reason, clean := doneReason(stop)
	if !clean {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: NameGenerate, Field: string(stop),
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

func (g *Generate) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
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
	tools := 0
	sawText := false
	var streamErr error

	for fr := range s.Frames() {
		if err := ctx.Err(); err != nil {
			streamErr = err
			break
		}
		if fr.Err != nil {
			streamErr = fr.Err
			break
		}
		frame := generateBody{Model: model, CreatedAt: stamp(g.opts.now())}
		emit := false
		for _, p := range fr.Delta {
			switch v := p.(type) {
			case *llm.Text:
				frame.Response += v.Text
				emit, sawText = true, true
			case *llm.Refusal:
				frame.Response += v.Text
				emit, sawText = true, true
			case *llm.Thinking:
				frame.Thinking += v.Text
				emit = emit || v.Text != ""
			case *llm.ToolUse:
				tools++
			case *llm.Image:
				diags = append(diags, drop(NameGenerate, llm.DropImage, "an assistant image has no field on this wire"))
			case *llm.Raw:
				diags = append(diags, drop(NameGenerate, llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
			}
		}
		if emit {
			if err := nd.Send(mustJSON(frame)); err != nil {
				return diags, err
			}
		}
		if fr.Stop != llm.StopNone {
			stop = fr.Stop
		}
	}

	if tools > 0 {
		d := drop(NameGenerate, llm.DropToolCall, "this endpoint has no tool_calls field; use /api/chat")
		d.Count = tools
		if !sawText {
			d.Severity = llm.SeverityFatal
		}
		diags = append(diags, d)
	}
	if streamErr == nil {
		streamErr = s.Err()
	}

	final := generateBody{Model: model, CreatedAt: stamp(g.opts.now()), Done: true, wireUsage: encodeUsage(s.Usage())}
	diags = append(diags, s.Usage().Uncarried(NameGenerate)...)
	reason, clean := doneReason(stop)
	if !clean {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: NameGenerate, Field: string(stop),
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
			Severity: llm.SeverityLoss, Format: NameGenerate, Detail: e.Message,
		})
	}
	if err := nd.Send(mustJSON(final)); err != nil {
		return diags, err
	}
	diags = append(diags, s.Diagnostics()...)
	return diags, streamErr
}

func (g *Generate) EncodeError(err *llm.Error, opts wire.Options) *wire.Outbound {
	return (&Chat{}).EncodeError(err, opts)
}
