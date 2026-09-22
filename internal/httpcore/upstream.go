// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"encoding/json"
	"strconv"
	"strings"
	"unicode/utf8"
)

const maxMessage = 2 << 10

func Upstream(body []byte) (message, code string) {
	body = trim(body)
	if len(body) == 0 || body[0] != '{' {
		return "", ""
	}
	// Every field is read on its own. A vendor is free to give one of them a
	// shape goose did not expect, and decoding the whole object at once would
	// let that one field throw away the sentence the client needs.
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil {
		return "", ""
	}
	// Each shape is tried in turn rather than switched on, because a field
	// being present is not the same as it carrying anything.
	if m, c := read(top["error"]); m != "" {
		return m, c
	}
	var many []detail
	if json.Unmarshal(top["errors"], &many) == nil && len(many) > 0 {
		if m := clip(many[0].Message); m != "" {
			return m, first(number(many[0].Code), number(many[0].Type))
		}
	}
	if m, c := read(top["detail"]); m != "" {
		return m, c
	}
	if m := clip(number(top["message"])); m != "" {
		return m, first(shortType(number(top["__type"])), number(top["code"]), number(top["status"]))
	}
	return "", ""
}

type detail struct {
	Message string          `json:"message"`
	Type    json.RawMessage `json:"type"`
	Code    json.RawMessage `json:"code"`
	Status  json.RawMessage `json:"status"`
}

func read(raw json.RawMessage) (string, string) {
	var d detail
	if json.Unmarshal(raw, &d) == nil && d.Message != "" {
		return clip(d.Message), first(number(d.Code), number(d.Type), number(d.Status))
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return clip(s), ""
	}
	return "", ""
}

func number(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var n int64
	if json.Unmarshal(raw, &n) == nil {
		return strconv.FormatInt(n, 10)
	}
	return ""
}

func shortType(s string) string {
	if i := strings.LastIndex(s, "#"); i >= 0 {
		return s[i+1:]
	}
	return s
}

func first(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func clip(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= maxMessage {
		return s
	}
	// The ceiling is in bytes and the message is in characters, so the cut
	// falls back to the last boundary rather than through one.
	cut := maxMessage
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + "…"
}

func trim(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}
