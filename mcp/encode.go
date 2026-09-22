// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/leelsey/goose/llm"
)

// A tool result holds parts, and a part can be a tool result. Nothing about a
// structure goose did not build says that it ends, and a stack that runs out is
// not an error a process recovers from. Depth is not the only way one of these
// grows: a result holding itself twice doubles at every level it is allowed, so
// how much comes out is bounded as well as how deep it is read.
const (
	maxNesting = 16
	maxBlocks  = 4096
)

func encodeContent(parts []llm.Part) []json.RawMessage {
	// One block is held back, so that running out of room can be said in the
	// same breath and the whole still fits.
	room := &budget{left: maxBlocks - 1}
	out := encodeParts(parts, 0, room)
	if room.short {
		out = append(out, mustJSON(map[string]any{"type": "text", "text": "the rest did not fit"}))
	}
	return out
}

// budget is what is left to say and whether anything went unsaid. It counts
// blocks as they come out rather than parts as they go in: one part can be an
// error marker and everything it holds.
type budget struct {
	left  int
	short bool
}

func (b *budget) take() bool {
	if b.left <= 0 {
		b.short = true
		return false
	}
	b.left--
	return true
}

func encodeParts(parts []llm.Part, depth int, room *budget) []json.RawMessage {
	out := make([]json.RawMessage, 0, len(parts))
	add := func(block json.RawMessage) bool {
		if !room.take() {
			return false
		}
		out = append(out, block)
		return true
	}
	for _, p := range parts {
		if room.left <= 0 {
			room.short = true
			break
		}
		switch v := p.(type) {
		case *llm.Text:
			add(mustJSON(map[string]any{"type": "text", "text": v.Text}))
		case *llm.Refusal:
			add(mustJSON(map[string]any{"type": "text", "text": v.Text}))
		case *llm.Image:
			kind := "image"
			if strings.HasPrefix(v.MIME, "audio/") {
				kind = "audio"
			}
			add(mustJSON(map[string]any{
				"type": kind, "mimeType": v.MIME,
				"data": base64.StdEncoding.EncodeToString(v.Data),
			}))
		case *llm.ToolResult:
			// Its parts are content this wire has blocks for, so they travel as
			// themselves rather than flattened into a sentence.
			if v.IsError && !add(mustJSON(map[string]any{"type": "text", "text": "error"})) {
				continue
			}
			if depth >= maxNesting {
				add(mustJSON(map[string]any{
					"type": "text", "text": "nested too deeply to carry",
				}))
				continue
			}
			out = append(out, encodeParts(v.Content, depth+1, room)...)
		case *llm.Raw:
			if v.Vendor == "mcp" && json.Valid(v.JSON) {
				add(v.JSON)
				continue
			}
			add(mustJSON(map[string]any{"type": "text", "text": string(v.JSON)}))
		default:
			// This wire has only these blocks. A part that reached here is still
			// a part the tool produced, so what it says travels.
			if text := describe(p); text != "" {
				add(mustJSON(map[string]any{"type": "text", "text": text}))
			}
		}
	}
	return out
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(`{"type":"text","text":""}`)
	}
	return b
}

// describe says what a part is in words, for a wire with no block of its own
// for it: its text where it has some, and otherwise which kind of block it
// was. Never goose's own field or type names — a kind is the same word every
// wire here uses for that block, and is what is left to say when the part
// carries nothing a caller could read.
func describe(p llm.Part) string {
	switch v := p.(type) {
	case *llm.Text:
		if v.Text != "" {
			return v.Text
		}
	case *llm.Thinking:
		if v.Text != "" {
			return v.Text
		}
	case *llm.Refusal:
		if v.Text != "" {
			return v.Text
		}
	case *llm.ToolUse:
		if len(v.Args) > 0 {
			return v.Name + " " + string(v.Args)
		}
		if v.Name != "" {
			return v.Name
		}
	}
	return string(p.Kind())
}
