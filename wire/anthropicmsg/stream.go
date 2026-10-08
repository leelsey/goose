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

	open  string
	index int
	next  int
	// thinkID and signed describe the thinking block open, if one is: the
	// backend's name for it, and whether its signature has been written.
	thinkID  string
	signed   bool
	tools    []*llm.ToolUse
	merger   *wirecore.ToolMerger
	toolSeen bool
	refused  bool
	err      error
}

func (c *Codec) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	st := &streamer{c: c, sse: wire.NewSSE(w), merger: wirecore.NewToolMerger()}
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
	checked := 0

	for fr := range s.Frames() {
		if err := ctx.Err(); err != nil {
			streamErr = err
			break
		}
		if fr.Err != nil {
			streamErr = fr.Err
			break
		}
		// A part this wire has no block for is lost as it is met, so a loss the
		// stream's policy will not take is refused there, before the part
		// itself or anything after it is written.
		for _, p := range fr.Delta {
			st.notePart(p)
			if streamErr = s.Check(st.diags[checked:]); streamErr != nil {
				break
			}
			checked = len(st.diags)
			st.writePart(p)
		}
		if streamErr != nil {
			break
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
	toolBlocks := st.prepareTools()
	name, clean := stopName(stop)
	if !clean {
		st.diags = append(st.diags, llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Field: string(stop),
			Detail: "stop reason has no spelling on this wire",
		})
	}
	if streamErr == nil {
		streamErr = s.Err()
	}
	if streamErr == nil {
		streamErr = s.Check(st.diags[checked:])
	}
	// A tool call is only ever written here; when it is itself what a loss
	// noted above got refused for, it must not still reach the client once
	// the error that refused it is about to be sent.
	if streamErr == nil {
		st.emitTools(toolBlocks)
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
	s.thinkID, s.signed = "", false
}

func (s *streamer) delta(v map[string]any) {
	_ = s.send("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": s.index, "delta": v,
	})
}

// notePart says what a part loses on this wire and writes none of it. Noted
// here, before this part's own Check, rather than accumulated for one
// diagnostic at the end, and kept apart from the writing: a loss the policy
// will not take must be caught before what it lost is written, not after.
func (s *streamer) notePart(p llm.Part) {
	if sig, _ := wirecore.Signed(p); sig != "" && p.Kind() != llm.KindThinking {
		s.diags = append(s.diags, unsignedParts(1))
	}
	switch v := p.(type) {
	case *llm.Text:
		if v.Text != "" {
			s.textAfterTool()
		}
	case *llm.Refusal:
		// A refusal streams as many deltas, and it is one refusal.
		if v.Text != "" && !s.refused {
			s.refused = true
			s.diags = append(s.diags, refusalAsText())
		}
	case *llm.Image:
		s.note(llm.DropImage, "an assistant image has no block on this wire")
	case *llm.ToolResult:
		s.note(llm.DropToolResult, "a tool result cannot be streamed from an assistant turn")
	case *llm.Raw:
		if _, ok := rawBlock(v); !ok {
			s.note(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire")
		}
	}
}

// writePart puts a part on the stream once what it loses has been let through.
func (s *streamer) writePart(p llm.Part) {
	switch v := p.(type) {
	case *llm.Text:
		if v.Text == "" {
			return
		}
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

	case *llm.Raw:
		if block, ok := rawBlock(v); ok {
			s.closeBlock()
			s.index = s.next
			s.next++
			_ = s.send("content_block_start", map[string]any{"type": "content_block_start", "index": s.index, "content_block": block})
			_ = s.send("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.index})
		}
	}
}

// rawBlock is a backend's own block on this wire, carried whole.
func rawBlock(v *llm.Raw) (map[string]any, bool) {
	if v.Vendor != Name || !json.Valid(v.JSON) {
		return nil, false
	}
	var block map[string]any
	if json.Unmarshal(v.JSON, &block) != nil {
		return nil, false
	}
	return block, true
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
	if v.Text == "" && v.Signature == "" {
		return
	}
	// A signature ends its block, and a backend that names its blocks says
	// where one ends; a call buffered in between writes nothing that would
	// close it. Written into the block still open, this part would join two
	// blocks' text under one of their signatures, which signs neither.
	if s.open == "thinking" && (s.signed || v.ID != "" && s.thinkID != "" && v.ID != s.thinkID) {
		s.closeBlock()
	}
	s.openBlock("thinking", map[string]any{"type": "thinking", "thinking": ""})
	if s.thinkID == "" {
		s.thinkID = v.ID
	}
	if v.Text != "" {
		s.delta(map[string]any{"type": "thinking_delta", "thinking": v.Text})
	}
	if v.Signature != "" {
		s.delta(map[string]any{"type": "signature_delta", "signature": wirecore.SealSignature(llm.FamilyAnthropic, v.Signer, v.Signature)})
		s.signed = true
	}
}

func (s *streamer) tool(v *llm.ToolUse) {
	s.toolSeen = true
	s.merger.Merge(v, nil)
	s.tools = s.merger.Tools()
}

// toolBlock is a tool call built and diagnosed, ready to write to the
// stream once it is known the answer will not be refused.
type toolBlock struct {
	id, name string
	args     json.RawMessage
}

// prepareTools builds each buffered call's block and notes what was lost
// building it, without writing anything: the caller Checks the result
// before emitTools puts a refused call on the wire.
func (s *streamer) prepareTools() []toolBlock {
	if s.merger.Overflowed() {
		s.diags = append(s.diags, llm.Diagnostic{
			Code: llm.WireBufferFull, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name,
			Detail: "the tool arguments outgrew what one turn may hold, so the call is short of what was streamed",
		})
	}
	blocks := make([]toolBlock, 0, len(s.tools))
	for _, t := range s.tools {
		args, broken := wirecore.ObjectArgsBroken(t.Args)
		if broken {
			s.diags = append(s.diags, brokenArgs(t.Name))
		}
		blocks = append(blocks, toolBlock{id: t.ID, name: t.Name, args: args})
	}
	s.tools = nil
	s.merger.Reset()
	return blocks
}

func (s *streamer) emitTools(blocks []toolBlock) {
	for _, t := range blocks {
		s.index = s.next
		s.next++
		s.open = "tool"
		_ = s.send("content_block_start", map[string]any{
			"type": "content_block_start", "index": s.index,
			"content_block": map[string]any{"type": "tool_use", "id": t.id, "name": t.name, "input": map[string]any{}},
		})
		s.delta(map[string]any{"type": "input_json_delta", "partial_json": string(t.args)})
		s.closeBlock()
	}
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
