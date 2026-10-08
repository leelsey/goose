// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type streamer struct {
	sse *wire.SSE
	seq int

	respID  string
	model   string
	created int64

	open        string
	itemID      string
	outputIndex int
	// What the open item's closing events carry, built as it streams: each
	// delta added by copying all before it would cost the square of their
	// number.
	buf        strings.Builder
	refusalBuf strings.Builder
	// A part is announced when something arrives for it, and its index is the
	// order it was announced in. A message that carries only a refusal has no
	// text part at all, which is what the unstreamed shape says too.
	textIndex    int
	refusalIndex int
	parts        int
	summaryOpen  bool
	// named is the open reasoning item's id when the backend gave it one, and
	// empty when goose minted it.
	named     string
	signature string

	imageBuf    []byte
	imageFmt    string
	imagePrompt string
	imageParts  int

	budget   int
	overflow bool

	items []item
	diags llm.Diagnostics
	err   error

	tools  []*llm.ToolUse
	merger *wirecore.ToolMerger
	ids    int
	extra  map[string]json.RawMessage
}

func (c *Codec) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	meta := s.Meta()
	model := meta.Model
	if model == "" {
		model = opts.RequestModel
	}
	st := &streamer{
		sse:    wire.NewSSE(w),
		respID: c.id(meta.ID), model: model, created: created(meta.Created), extra: meta.Extra,
		items: []item{}, merger: wirecore.NewToolMerger(),
		textIndex: -1, refusalIndex: -1,
		budget: c.opts.MaxBuffer,
	}
	if st.budget <= 0 {
		st.budget = DefaultMaxBuffer
	}

	h := http.Header{"Content-Type": []string{"text/event-stream"}}
	if meta.Session != "" {
		h.Set("X-Goose-Session", meta.Session)
	}
	if err := st.sse.Open(h); err != nil {
		return st.diags, err
	}

	st.event("response.created", map[string]any{"response": st.render(st.snapshot("in_progress", nil, llm.Usage{}))})
	st.event("response.in_progress", map[string]any{"response": st.render(st.snapshot("in_progress", nil, llm.Usage{}))})

	stop := llm.StopNone
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
		// A part this wire has no item for is lost as it is met, so a loss the
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
		}
		if st.err != nil {
			return st.diags, st.err
		}
	}

	st.closeItem()
	toolBlocks := st.prepareTools()
	if _, _, clean := status(stop, false, true); !clean {
		st.diags = append(st.diags, unknownStop(stop))
	}

	if streamErr == nil {
		streamErr = s.Err()
	}
	if streamErr == nil {
		streamErr = s.Check(st.diags[checked:])
	}
	// A tool call is only ever written here; when it is itself what a loss
	// noted above got refused for, it must not still reach the client, or the
	// closing snapshot's output array, once the error that refused it is sent.
	if streamErr == nil {
		st.emitTools(toolBlocks)
	}

	state, incomplete, _ := status(stop, streamErr != nil, true)
	if st.overflow && state == "completed" {
		state, incomplete = "incomplete", "max_output_tokens"
		st.diags = append(st.diags, llm.Diagnostic{
			Code: llm.WireBufferFull, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name,
			Detail: "the answer outgrew what one stream may hold, so the response object is short of what was streamed",
		})
	}
	var info *incompleteInfo
	if incomplete != "" {
		info = &incompleteInfo{Reason: incomplete}
	}
	snap := st.snapshot(state, info, s.Usage())
	st.diags = append(st.diags, s.Usage().Uncarried(Name, llm.FigureCacheRead, llm.FigureReasoning, llm.FigureTier)...)
	if streamErr != nil {
		e := llm.AsError(streamErr)
		snap.Error = &errorInfo{Code: e.Code, Message: e.Message}
		st.event("error", map[string]any{"type": "error", "code": errorCode(e.Code), "message": e.Message})
		st.diags = append(st.diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Detail: e.Message,
		})
	}
	st.event("response."+state, map[string]any{"response": st.render(snap)})
	st.diags = append(st.diags, s.Diagnostics()...)
	return st.diags, streamErr
}

func (s *streamer) render(obj responseObject) any {
	if len(s.extra) == 0 {
		return obj
	}
	raw, err := json.Marshal(obj)
	if err != nil {
		return obj
	}
	out, err := wire.WithExtra(raw, s.extra)
	if err != nil {
		return obj
	}
	return json.RawMessage(out)
}

func (s *streamer) snapshot(state string, info *incompleteInfo, u llm.Usage) responseObject {
	obj := responseObject{
		ID: s.respID, Object: "response", CreatedAt: s.created, Status: state, Model: s.model,
		Output: s.items, IncompleteDetails: info,
	}
	if obj.Output == nil {
		obj.Output = []item{}
	}
	if !u.Empty() {
		w := encodeUsage(u)
		obj.Usage = &w
		obj.Tier = u.Tier
	}
	return obj
}

func (s *streamer) event(name string, body map[string]any) {
	if s.err != nil {
		return
	}
	body["type"] = name
	body["sequence_number"] = s.seq
	s.seq++
	raw, err := json.Marshal(body)
	if err != nil {
		s.err = err
		return
	}
	s.err = s.sse.Send(name, raw)
}

func (s *streamer) nextID(prefix string) string {
	s.ids++
	return prefix + "_" + strconv.Itoa(s.ids)
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
	case *llm.Thinking:
		if len(v.Redacted) > 0 {
			s.diags = append(s.diags, drop(llm.DropThinking, "redacted thinking has no item on this wire"))
		}
	case *llm.ToolResult:
		s.diags = append(s.diags, drop(llm.DropToolResult, "a tool result cannot ride on an assistant turn here"))
	case *llm.Raw:
		if _, ok := rawItem(v); !ok {
			s.diags = append(s.diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
		}
	}
}

// writePart puts a part on the stream once what it loses has been let through.
func (s *streamer) writePart(p llm.Part) {
	switch v := p.(type) {
	case *llm.Thinking:
		if len(v.Redacted) > 0 {
			return
		}
		// A backend's item that holds neither a summary nor a signature is
		// still one it stored, and the unstreamed answer hands it over too.
		if v.Text == "" && v.Signature == "" && v.ID == "" {
			return
		}
		if s.open == "reasoning" && !sameReasoning(s.named, s.signature, v) {
			s.closeItem()
		}
		s.openReasoning(v.ID)
		if v.Text != "" {
			if !s.summaryOpen {
				s.event("response.reasoning_summary_part.added", map[string]any{
					"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0,
					"part": summaryPart{Type: "summary_text", Text: ""},
				})
				s.summaryOpen = true
			}
			s.hold(&s.buf, v.Text)
			s.event("response.reasoning_summary_text.delta", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0, "delta": v.Text,
			})
		}
		if v.Signature != "" {
			s.signature = wirecore.SealSignature(llm.FamilyOpenAI, v.Signer, v.Signature)
		}
	case *llm.Text:
		if v.Text == "" {
			return
		}
		s.openMessage()
		at := s.announce(&s.textIndex, contentPart{Type: "output_text", Annotations: []string{}})
		s.hold(&s.buf, v.Text)
		s.event("response.output_text.delta", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": at, "delta": v.Text,
		})
	case *llm.Refusal:
		if v.Text == "" {
			return
		}
		s.openMessage()
		// A refusal sits beside the text rather than replacing it.
		at := s.announce(&s.refusalIndex, contentPart{Type: "refusal"})
		s.hold(&s.refusalBuf, v.Text)
		s.event("response.refusal.delta", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": at, "delta": v.Text,
		})
	case *llm.ToolUse:
		s.merger.Merge(v, s.charge)
		s.tools = s.merger.Tools()
	case *llm.Image:
		s.image(v)
	case *llm.Raw:
		it, ok := rawItem(v)
		if !ok {
			return
		}
		s.closeItem()
		idx := len(s.items)
		s.items = append(s.items, it)
		s.event("response.output_item.added", map[string]any{"output_index": idx, "item": it})
		s.event("response.output_item.done", map[string]any{"output_index": idx, "item": it})
	}
}

func (s *streamer) hold(dst *strings.Builder, text string) {
	if s.charge(len(text)) {
		dst.WriteString(text)
	}
}

func (s *streamer) charge(n int) bool {
	if n > s.budget {
		s.overflow = true
		return false
	}
	s.budget -= n
	return true
}

func (s *streamer) image(v *llm.Image) {
	if len(v.Data) == 0 {
		return
	}
	if s.open != "image" || (v.ID != "" && v.ID != s.itemID) {
		s.closeItem()
		s.open = "image"
		s.itemID = v.ID
		if s.itemID == "" {
			s.itemID = s.nextID("ig")
		}
		s.outputIndex = len(s.items)
		s.imageFmt = imageFormat(v.MIME)
		s.imageParts = 0
		s.event("response.output_item.added", map[string]any{
			"output_index": s.outputIndex,
			"item":         item{Type: "image_generation_call", ID: s.itemID, Status: "in_progress"},
		})
		s.event("response.image_generation_call.in_progress", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex,
		})
	}
	if v.MIME != "" {
		s.imageFmt = imageFormat(v.MIME)
	}
	if v.Prompt != "" {
		s.imagePrompt = v.Prompt
	}
	s.imageBuf = v.Data
	if v.Partial {
		s.event("response.image_generation_call.partial_image", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex,
			"partial_image_index": s.imageParts,
			"partial_image_b64":   base64.StdEncoding.EncodeToString(v.Data),
			"output_format":       s.imageFmt,
		})
		s.imageParts++
		return
	}
	s.closeItem()
}

func (s *streamer) openReasoning(id string) {
	if s.open == "reasoning" {
		return
	}
	s.closeItem()
	s.open = "reasoning"
	s.named = id
	s.itemID = id
	if s.itemID == "" {
		s.itemID = s.nextID(mintedReasoning)
	}
	s.outputIndex = len(s.items)
	s.event("response.output_item.added", map[string]any{
		"output_index": s.outputIndex,
		"item":         item{Type: "reasoning", ID: s.itemID, Summary: []summaryPart{}},
	})
}

func (s *streamer) openMessage() {
	if s.open == "message" {
		return
	}
	s.closeItem()
	s.open = "message"
	s.itemID = s.nextID("msg")
	s.outputIndex = len(s.items)
	s.event("response.output_item.added", map[string]any{
		"output_index": s.outputIndex,
		"item":         item{Type: "message", ID: s.itemID, Role: "assistant", Status: "in_progress", Content: []contentPart{}},
	})
}

// announce names a content part the first time something arrives for it.
func (s *streamer) announce(index *int, p contentPart) int {
	if *index < 0 {
		*index = s.parts
		s.parts++
		s.event("response.content_part.added", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": *index,
			"part": p,
		})
	}
	return *index
}

func (s *streamer) closeItem() {
	text, refusal := s.buf.String(), s.refusalBuf.String()
	switch s.open {
	case "reasoning":
		if s.summaryOpen {
			s.event("response.reasoning_summary_text.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0, "text": text,
			})
			s.event("response.reasoning_summary_part.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0,
				"part": summaryPart{Type: "summary_text", Text: text},
			})
			s.summaryOpen = false
		}
		it := item{Type: "reasoning", ID: s.itemID, Summary: []summaryPart{}, EncryptedContent: s.signature}
		if text != "" {
			it.Summary = append(it.Summary, summaryPart{Type: "summary_text", Text: text})
		}
		s.items = append(s.items, it)
		s.event("response.output_item.done", map[string]any{"output_index": s.outputIndex, "item": it})
	case "image":
		s.event("response.image_generation_call.completed", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex,
		})
		it := item{
			Type: "image_generation_call", ID: s.itemID, Status: "completed",
			Result:        base64.StdEncoding.EncodeToString(s.imageBuf),
			OutputFormat:  s.imageFmt,
			RevisedPrompt: s.imagePrompt,
		}
		s.items = append(s.items, it)
		s.event("response.output_item.done", map[string]any{"output_index": s.outputIndex, "item": it})
		s.imageBuf, s.imagePrompt, s.imageFmt, s.imageParts = nil, "", "", 0
	case "message":
		it := item{Type: "message", ID: s.itemID, Role: "assistant", Status: "completed"}
		// Every part announced is closed and carried, at the index it was
		// announced under, so a client that followed the events and one that
		// read the closing object agree about what the message holds.
		parts := make([]contentPart, s.parts)
		if s.textIndex >= 0 {
			parts[s.textIndex] = contentPart{Type: "output_text", Text: text, Annotations: []string{}}
		}
		if s.refusalIndex >= 0 {
			parts[s.refusalIndex] = contentPart{Type: "refusal", Refusal: refusal}
		}
		// The closing events go out in the order the parts were announced in,
		// which is the order a client is entitled to read them in.
		for at, part := range parts {
			switch at {
			case s.textIndex:
				s.event("response.output_text.done", map[string]any{
					"item_id": s.itemID, "output_index": s.outputIndex, "content_index": at, "text": text,
				})
			case s.refusalIndex:
				s.event("response.refusal.done", map[string]any{
					"item_id": s.itemID, "output_index": s.outputIndex, "content_index": at, "refusal": refusal,
				})
			}
			s.event("response.content_part.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "content_index": at, "part": part,
			})
		}
		it.Content = append(it.Content, parts...)
		s.items = append(s.items, it)
		s.event("response.output_item.done", map[string]any{"output_index": s.outputIndex, "item": it})
	}
	s.open = ""
	s.buf.Reset()
	s.refusalBuf.Reset()
	s.named, s.signature = "", ""
	s.textIndex, s.refusalIndex, s.parts = -1, -1, 0
}

// toolBlock is a function call built and diagnosed, ready to write to the
// stream once it is known the answer will not be refused.
type toolBlock struct {
	id, callID, name, args string
}

// prepareTools builds each buffered call's block and notes what was lost
// building it, without writing anything: the caller Checks the result
// before emitTools puts a refused call on the wire.
func (s *streamer) prepareTools() []toolBlock {
	blocks := make([]toolBlock, 0, len(s.tools))
	for _, t := range s.tools {
		args := string(t.Args)
		if args == "" || !json.Valid([]byte(args)) {
			if _, broken := wirecore.ObjectArgsBroken(t.Args); broken {
				d := drop(llm.DropToolCall, "this call's arguments did not survive, so it travelled with none rather than with what was meant")
				d.Field = t.Name
				s.diags = append(s.diags, d)
			}
			args = "{}"
		}
		blocks = append(blocks, toolBlock{id: s.nextID("fc"), callID: t.ID, name: t.Name, args: args})
	}
	s.tools = nil
	s.merger.Reset()
	return blocks
}

func (s *streamer) emitTools(blocks []toolBlock) {
	for _, b := range blocks {
		idx := len(s.items)
		it := item{Type: "function_call", ID: b.id, CallID: b.callID, Name: b.name, Args: b.args, Status: "completed"}
		s.event("response.output_item.added", map[string]any{
			"output_index": idx,
			"item":         item{Type: "function_call", ID: b.id, CallID: b.callID, Name: b.name, Args: "", Status: "in_progress"},
		})
		s.event("response.function_call_arguments.delta", map[string]any{
			"item_id": b.id, "output_index": idx, "delta": b.args,
		})
		s.event("response.function_call_arguments.done", map[string]any{
			"item_id": b.id, "output_index": idx, "arguments": b.args,
		})
		s.items = append(s.items, it)
		s.event("response.output_item.done", map[string]any{"output_index": idx, "item": it})
	}
}
