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
