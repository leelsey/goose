// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"encoding/json"
	"strconv"
	"strings"
)

const maxMessage = 2 << 10

func Upstream(body []byte) (message, code string) {
	body = trim(body)
	if len(body) == 0 || body[0] != '{' {
		return "", ""
	}
	var top struct {
		Error   json.RawMessage `json:"error"`
		Errors  []detail        `json:"errors"`
		Detail  json.RawMessage `json:"detail"`
		Message string          `json:"message"`
		Type    string          `json:"__type"`
		Code    json.RawMessage `json:"code"`
		Status  string          `json:"status"`
	}
	if json.Unmarshal(body, &top) != nil {
		return "", ""
	}
	switch {
	case len(top.Error) > 0:
		if m, c := read(top.Error); m != "" {
			return m, c
		}
	case len(top.Errors) > 0:
		if m := clip(top.Errors[0].Message); m != "" {
			return m, first(number(top.Errors[0].Code), top.Errors[0].Type)
		}
	case len(top.Detail) > 0:
		if m, c := read(top.Detail); m != "" {
			return m, c
		}
	}
	if m := clip(top.Message); m != "" {
		return m, first(shortType(top.Type), number(top.Code), top.Status)
	}
	return "", ""
}

type detail struct {
	Message string          `json:"message"`
	Type    string          `json:"type"`
	Code    json.RawMessage `json:"code"`
	Status  string          `json:"status"`
}

func read(raw json.RawMessage) (string, string) {
	var d detail
	if json.Unmarshal(raw, &d) == nil && d.Message != "" {
		return clip(d.Message), first(number(d.Code), d.Type, d.Status)
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
	if len(s) > maxMessage {
		return s[:maxMessage] + "…"
	}
	return s
}

func trim(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}
