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
			return m, codeOf(many[0].Code, many[0].Type)
		}
	}
	if m, c := read(top["detail"]); m != "" {
		return m, c
	}
	if m := clip(number(top["message"])); m != "" {
		return m, first(shortType(number(top["__type"])), codeOf(top["code"], top["status"]))
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
		return clip(d.Message), codeOf(d.Code, d.Type, d.Status)
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

// codeOf names a failure the way the vendor does. A number beside a name is the
// status again as often as not, and the name is what says which failure of
// that status it was, so a name is taken wherever one was given.
func codeOf(raws ...json.RawMessage) string {
	for _, raw := range raws {
		var s string
		if json.Unmarshal(raw, &s) == nil && s != "" {
			return s
		}
	}
	for _, raw := range raws {
		if n := number(raw); n != "" {
			return n
		}
	}
	return ""
}

// Code is an error code as a vendor spells it, a string or a number. A backend
// that sends a number where another sends a name has still said what went
// wrong, and decoding the error must not fail on the difference.
type Code string

func (c *Code) UnmarshalJSON(raw []byte) error {
	*c = Code(number(raw))
	return nil
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
