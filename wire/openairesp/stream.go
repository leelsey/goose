// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"

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
	buf         string
	refusalBuf  string
	refusalOpen bool
	summaryOpen bool

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
		}
		if st.err != nil {
			return st.diags, st.err
		}
	}

	st.closeItem()
	st.flushTools()

	if streamErr == nil {
		streamErr = s.Err()
	}

	state, incomplete := status(stop, streamErr != nil)
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

func (s *streamer) part(p llm.Part) {
	switch v := p.(type) {
	case *llm.Thinking:
		if len(v.Redacted) > 0 {
			s.diags = append(s.diags, drop(llm.DropThinking, "redacted thinking has no item on this wire"))
			return
		}
		if v.Text == "" {
			return
		}
		s.openReasoning()
		s.hold(&s.buf, v.Text)
		s.event("response.reasoning_summary_text.delta", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0, "delta": v.Text,
		})
	case *llm.Text:
		if v.Text == "" {
			return
		}
		s.openMessage()
		s.hold(&s.buf, v.Text)
		s.event("response.output_text.delta", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 0, "delta": v.Text,
		})
	case *llm.Refusal:
		if v.Text == "" {
			return
		}
		s.openMessage()
		// A part is announced before anything is streamed into it, and a
		// refusal sits beside the text rather than replacing it.
		if !s.refusalOpen {
			s.refusalOpen = true
			s.event("response.content_part.added", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 1,
				"part": contentPart{Type: "refusal"},
			})
		}
		s.hold(&s.refusalBuf, v.Text)
		s.event("response.refusal.delta", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 1, "delta": v.Text,
		})
	case *llm.ToolUse:
		s.merger.Merge(v, s.charge)
		s.tools = s.merger.Tools()
	case *llm.Image:
		s.image(v)
	case *llm.ToolResult:
		s.diags = append(s.diags, drop(llm.DropToolResult, "a tool result cannot ride on an assistant turn here"))
	case *llm.Raw:
		s.diags = append(s.diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
	}
}

func (s *streamer) hold(dst *string, text string) {
	if s.charge(len(text)) {
		*dst += text
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

func (s *streamer) openReasoning() {
	if s.open == "reasoning" {
		return
	}
	s.closeItem()
	s.open = "reasoning"
	s.itemID = s.nextID("rs")
	s.outputIndex = len(s.items)
	s.event("response.output_item.added", map[string]any{
		"output_index": s.outputIndex,
		"item":         item{Type: "reasoning", ID: s.itemID, Summary: []summaryPart{}},
	})
	s.event("response.reasoning_summary_part.added", map[string]any{
		"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0,
		"part": summaryPart{Type: "summary_text", Text: ""},
	})
	s.summaryOpen = true
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
	s.event("response.content_part.added", map[string]any{
		"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 0,
		"part": contentPart{Type: "output_text", Annotations: []string{}},
	})
}

func (s *streamer) closeItem() {
	switch s.open {
	case "reasoning":
		if s.summaryOpen {
			s.event("response.reasoning_summary_text.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0, "text": s.buf,
			})
			s.event("response.reasoning_summary_part.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "summary_index": 0,
				"part": summaryPart{Type: "summary_text", Text: s.buf},
			})
			s.summaryOpen = false
		}
		it := item{Type: "reasoning", ID: s.itemID, Summary: []summaryPart{}}
		if s.buf != "" {
			it.Summary = append(it.Summary, summaryPart{Type: "summary_text", Text: s.buf})
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
		s.event("response.output_text.done", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 0, "text": s.buf,
		})
		s.event("response.content_part.done", map[string]any{
			"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 0,
			"part": contentPart{Type: "output_text", Text: s.buf, Annotations: []string{}},
		})
		it := item{Type: "message", ID: s.itemID, Role: "assistant", Status: "completed"}
		// The text part was announced when the message opened, so it is carried
		// whether or not anything was written into it: a client that followed
		// the events and one that read the closing object have to agree on
		// which part sits at which index.
		it.Content = append(it.Content, contentPart{Type: "output_text", Text: s.buf, Annotations: []string{}})
		// The part was announced when the refusal opened, so it is closed on
		// the way out whether or not anything fitted in it.
		if s.refusalOpen {
			s.event("response.refusal.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 1, "refusal": s.refusalBuf,
			})
			s.event("response.content_part.done", map[string]any{
				"item_id": s.itemID, "output_index": s.outputIndex, "content_index": 1,
				"part": contentPart{Type: "refusal", Refusal: s.refusalBuf},
			})
			it.Content = append(it.Content, contentPart{Type: "refusal", Refusal: s.refusalBuf})
		}
		s.items = append(s.items, it)
		s.event("response.output_item.done", map[string]any{"output_index": s.outputIndex, "item": it})
	}
	s.open = ""
	s.buf = ""
	s.refusalBuf = ""
	s.refusalOpen = false
}

func (s *streamer) flushTools() {
	for _, t := range s.tools {
		args := string(t.Args)
		if args == "" || !json.Valid([]byte(args)) {
			args = "{}"
		}
		id := s.nextID("fc")
		idx := len(s.items)
		it := item{Type: "function_call", ID: id, CallID: t.ID, Name: t.Name, Args: args, Status: "completed"}
		s.event("response.output_item.added", map[string]any{
			"output_index": idx,
			"item":         item{Type: "function_call", ID: id, CallID: t.ID, Name: t.Name, Args: "", Status: "in_progress"},
		})
		s.event("response.function_call_arguments.delta", map[string]any{
			"item_id": id, "output_index": idx, "delta": args,
		})
		s.event("response.function_call_arguments.done", map[string]any{
			"item_id": id, "output_index": idx, "arguments": args,
		})
		s.items = append(s.items, it)
		s.event("response.output_item.done", map[string]any{"output_index": idx, "item": it})
	}
	s.tools = nil
	s.merger.Reset()
}
