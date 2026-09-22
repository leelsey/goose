// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/leelsey/goose/llm"
)

func encodeContent(parts []llm.Part) []json.RawMessage {
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
			out = append(out, encodeContent(v.Content)...)
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
// for it: its text where it has some, and otherwise what it is. Never goose's
// own field names, which are not something a model should be reading.
// describe says what a part is in words, for a wire with no block of its own
// for it. Never goose's own names, which are not something a model should be
// reading: a part with nothing readable in it says nothing.
func describe(p llm.Part) string {
	switch v := p.(type) {
	case *llm.Text:
		return v.Text
	case *llm.Thinking:
		return v.Text
	case *llm.Refusal:
		return v.Text
	case *llm.ToolUse:
		if len(v.Args) > 0 {
			return v.Name + " " + string(v.Args)
		}
		return v.Name
	}
	return ""
}
