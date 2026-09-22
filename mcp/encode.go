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
// structure goose did not build says that it ends, and a stack that runs out
// is not an error a process recovers from.
const maxNesting = 16

func encodeContent(parts []llm.Part) []json.RawMessage { return encodeParts(parts, 0) }

func encodeParts(parts []llm.Part, depth int) []json.RawMessage {
	out := make([]json.RawMessage, 0, len(parts))
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text:
			out = append(out, mustJSON(map[string]any{"type": "text", "text": v.Text}))
		case *llm.Refusal:
			out = append(out, mustJSON(map[string]any{"type": "text", "text": v.Text}))
		case *llm.Image:
			kind := "image"
			if strings.HasPrefix(v.MIME, "audio/") {
				kind = "audio"
			}
			out = append(out, mustJSON(map[string]any{
				"type": kind, "mimeType": v.MIME,
				"data": base64.StdEncoding.EncodeToString(v.Data),
			}))
		case *llm.ToolResult:
			// Its parts are content this wire has blocks for, so they travel as
			// themselves rather than flattened into a sentence.
			if v.IsError {
				out = append(out, mustJSON(map[string]any{"type": "text", "text": "error"}))
			}
			if depth >= maxNesting {
				out = append(out, mustJSON(map[string]any{
					"type": "text", "text": "nested too deeply to carry",
				}))
				continue
			}
			out = append(out, encodeParts(v.Content, depth+1)...)
		case *llm.Raw:
			if v.Vendor == "mcp" && json.Valid(v.JSON) {
				out = append(out, v.JSON)
				continue
			}
			out = append(out, mustJSON(map[string]any{"type": "text", "text": string(v.JSON)}))
		default:
			// This wire has only these blocks. A part that reached here is still
			// a part the tool produced, so what it says travels.
			if text := describe(p); text != "" {
				out = append(out, mustJSON(map[string]any{"type": "text", "text": text}))
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
		return v.Name
	}
	return string(p.Kind())
}
