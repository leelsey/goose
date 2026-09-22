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
	case *llm.ToolResult:
		if inner := llm.TextOf(v.Content); inner != "" {
			if v.IsError {
				return "error: " + inner
			}
			return inner
		}
	}
	// Nothing readable came out of it, but a part that reached here is still a
	// part the tool produced, so its kind travels rather than nothing at all.
	return string(p.Kind())
}
