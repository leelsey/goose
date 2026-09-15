// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"context"
	"encoding/json"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type streamer struct {
	c     *Codec
	sse   *wire.SSE
	diags llm.Diagnostics

	open     string
	index    int
	next     int
	tools    []*llm.ToolUse
	byIndex  map[int]*llm.ToolUse
	toolSeen bool
	err      error
}

func (c *Codec) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	st := &streamer{c: c, sse: wire.NewSSE(w), byIndex: map[int]*llm.ToolUse{}}
	meta := s.Meta()

	h := header(meta.Session, "text/event-stream")
	if err := st.sse.Open(h); err != nil {
		return st.diags, err
	}

	model := meta.Model
	if model == "" {
		model = opts.RequestModel
	}
	start := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": c.id(meta.ID), "type": "message", "role": "assistant", "model": model,
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": encodeUsage(s.Usage()),
		},
	}
	if extra := meta.Extra; len(extra) > 0 {
		msg := start["message"].(map[string]any)
		for k, v := range extra {
			if _, taken := msg[k]; !taken {
				msg[k] = v
			}
		}
	}
	if err := st.send("message_start", start); err != nil {
		return st.diags, err
	}

	stop := llm.StopNone
	stopSeq := ""
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
		for _, p := range fr.Delta {
			st.part(p)
		}
		if fr.Stop != llm.StopNone {
			stop = fr.Stop
			if fr.StopSequence != "" {
				stopSeq = fr.StopSequence
			}
		}
		if st.err != nil {
			return st.diags, st.err
		}
	}

	st.closeBlock()
	st.flushTools()

	if streamErr == nil {
		streamErr = s.Err()
	}
	if streamErr != nil {
		e := llm.AsError(streamErr)
		_ = st.send("error", map[string]any{
			"type":  "error",
			"error": map[string]any{"type": errorType(e.Kind), "message": e.Message},
		})
		st.diags = append(st.diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Detail: e.Message,
		})
	}

	name, clean := stopName(stop)
	if !clean {
		st.diags = append(st.diags, llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Field: string(stop),
			Detail: "stop reason has no spelling on this wire",
		})
	}
	delta := map[string]any{"stop_reason": nil, "stop_sequence": nil}
	if name != "" {
		delta["stop_reason"] = name
	}
	if stopSeq != "" {
		delta["stop_sequence"] = stopSeq
	}
	st.diags = append(st.diags, s.Usage().Uncarried(Name, llm.FigureCacheRead, llm.FigureCacheWrite, llm.FigureTier)...)
	if err := st.send("message_delta", map[string]any{
		"type": "message_delta", "delta": delta, "usage": encodeUsage(s.Usage()),
	}); err != nil {
		return st.diags, err
	}
	if err := st.send("message_stop", map[string]any{"type": "message_stop"}); err != nil {
		return st.diags, err
	}
	st.diags = append(st.diags, s.Diagnostics()...)
	return st.diags, streamErr
}

func (s *streamer) send(name string, v any) error {
	if s.err != nil {
		return s.err
	}
	b, err := json.Marshal(v)
	if err != nil {
		s.err = err
		return err
	}
	s.err = s.sse.Send(name, b)
	return s.err
}

func (s *streamer) openBlock(kind string, block map[string]any) {
	if s.open == kind && kind != "redacted" {
		return
	}
	s.closeBlock()
	s.index = s.next
	s.next++
	s.open = kind
	_ = s.send("content_block_start", map[string]any{
		"type": "content_block_start", "index": s.index, "content_block": block,
	})
}

func (s *streamer) closeBlock() {
	if s.open == "" {
		return
	}
	_ = s.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.index})
	s.open = ""
}

func (s *streamer) delta(v map[string]any) {
	_ = s.send("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": s.index, "delta": v,
	})
}

func (s *streamer) part(p llm.Part) {
	switch v := p.(type) {
	case *llm.Text:
		if v.Text == "" {
			return
		}
		s.textAfterTool()
		s.openBlock("text", map[string]any{"type": "text", "text": ""})
		s.delta(map[string]any{"type": "text_delta", "text": v.Text})

	case *llm.Refusal:
		if v.Text == "" {
			return
		}
		s.openBlock("text", map[string]any{"type": "text", "text": ""})
		s.delta(map[string]any{"type": "text_delta", "text": v.Text})

	case *llm.Thinking:
		s.thinking(v)

	case *llm.ToolUse:
		s.tool(v)

	case *llm.Image:
		s.note(llm.DropImage, "an assistant image has no block on this wire")

	case *llm.ToolResult:
		s.note(llm.DropToolResult, "a tool result cannot be streamed from an assistant turn")

	case *llm.Raw:
		if v.Vendor == Name && json.Valid(v.JSON) {
			var block map[string]any
			if json.Unmarshal(v.JSON, &block) == nil {
				s.closeBlock()
				s.index = s.next
				s.next++
				_ = s.send("content_block_start", map[string]any{"type": "content_block_start", "index": s.index, "content_block": block})
				_ = s.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.index})
				return
			}
		}
		s.note(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire")
	}
}

func (s *streamer) thinking(v *llm.Thinking) {
	if len(v.Redacted) > 0 {
		s.closeBlock()
		s.index = s.next
		s.next++
		_ = s.send("content_block_start", map[string]any{
			"type": "content_block_start", "index": s.index,
			"content_block": map[string]any{"type": "redacted_thinking", "data": string(v.Redacted)},
		})
		_ = s.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.index})
		return
	}
	if v.Text != "" {
		s.openBlock("thinking", map[string]any{"type": "thinking", "thinking": ""})
		s.delta(map[string]any{"type": "thinking_delta", "thinking": v.Text})
	}
	if v.Signature != "" {
		s.openBlock("thinking", map[string]any{"type": "thinking", "thinking": ""})
		s.delta(map[string]any{"type": "signature_delta", "signature": v.Signature})
	}
}

func (s *streamer) tool(v *llm.ToolUse) {
	s.toolSeen = true
	key := 0
	if v.Index != nil {
		key = *v.Index
	} else if v.ID == "" && len(s.tools) > 0 {
		key = len(s.tools) - 1
	} else {
		key = len(s.tools)
	}
	t, ok := s.byIndex[key]
	if !ok {
		t = &llm.ToolUse{ID: v.ID, Name: v.Name}
		s.byIndex[key] = t
		s.tools = append(s.tools, t)
	}
	if v.ID != "" {
		t.ID = v.ID
	}
	if v.Name != "" {
		t.Name = v.Name
	}
	t.Args = append(t.Args, v.Args...)
}

func (s *streamer) flushTools() {
	for _, t := range s.tools {
		args := wirecore.ObjectArgs(t.Args)
		s.index = s.next
		s.next++
		s.open = "tool"
		_ = s.send("content_block_start", map[string]any{
			"type": "content_block_start", "index": s.index,
			"content_block": map[string]any{"type": "tool_use", "id": t.ID, "name": t.Name, "input": map[string]any{}},
		})
		s.delta(map[string]any{"type": "input_json_delta", "partial_json": string(args)})
		s.closeBlock()
	}
	s.tools = nil
	s.byIndex = map[int]*llm.ToolUse{}
}

func (s *streamer) textAfterTool() {
	if !s.toolSeen {
		return
	}
	s.toolSeen = false
	s.note(llm.DropUnknownBlock, "text arrived after a tool call; tool blocks are emitted after the text on this wire")
	s.diags[len(s.diags)-1].Severity = llm.SeverityInfo
}

func (s *streamer) note(code, detail string) {
	s.diags = append(s.diags, llm.Diagnostic{
		Code: code, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Detail: detail,
	})
}
