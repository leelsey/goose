// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wire_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

func framesFrom(script []byte) []*llm.Frame {
	var out []*llm.Frame
	for _, b := range script {
		switch b % 12 {
		case 0:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Text{Text: "t"}}})
		case 1:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "r"}}})
		case 2:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "sig"}}})
		case 3:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Redacted: []byte("x")}}})
		case 4:
			i := 0
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "a", Name: "f", Args: json.RawMessage(`{"k":1}`), Index: &i}}})
		case 5:
			i := 1
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.ToolUse{Args: json.RawMessage(`{"k":2}`), Index: &i}}})
		case 6:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: "no"}}})
		case 7:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Image{MIME: "image/png", Data: []byte("ab")}}})
		case 8:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Raw{Vendor: "anthropic", Type: "odd", JSON: json.RawMessage(`{"type":"odd"}`)}}})
		case 9:
			out = append(out, &llm.Frame{Role: llm.RoleAssistant})
		case 10:
			u := llm.Usage{Prompt: 1, Completion: 1, Reported: true}
			out = append(out, &llm.Frame{Usage: &u, TurnEnd: true})
		case 11:
			out = append(out, &llm.Frame{Stop: llm.StopEnd})
		}
	}
	return out
}

func FuzzAnthropicStreamStaysWellFormed(f *testing.F) {
	for _, seed := range [][]byte{
		{0, 11}, {1, 2, 0, 11}, {4, 5, 11}, {0, 4, 0, 11}, {3, 1, 2, 11},
		{6, 11}, {7, 11}, {8, 0, 11}, {9, 0, 10, 11}, {},
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, script []byte) {
		if len(script) > 64 {
			script = script[:64]
		}
		s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "m"}, framesFrom(script))
		w := &wire.BufferWriter{}
		codec := anthropicmsg.New(anthropicmsg.Options{})
		if _, err := codec.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
			_ = err
		}
		checkAnthropic(t, w.Body.String(), script)
	})
}

func checkAnthropic(t *testing.T, body string, script []byte) {
	t.Helper()
	open := -1
	seen := map[int]bool{}
	starts, stops := 0, 0
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var data string
		for _, line := range strings.Split(block, "\n") {
			if rest, ok := strings.CutPrefix(line, "data: "); ok {
				data = rest
			}
		}
		if data == "" {
			continue
		}
		var ev struct {
			Type  string `json:"type"`
			Index *int   `json:"index"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			t.Fatalf("script %v: unreadable event %q", script, data)
		}
		switch ev.Type {
		case "content_block_start":
			if open != -1 {
				t.Fatalf("script %v: block %d opened while %d was open", script, *ev.Index, open)
			}
			if ev.Index == nil {
				t.Fatalf("script %v: a block opened without an index", script)
			}
			if seen[*ev.Index] {
				t.Fatalf("script %v: index %d opened twice", script, *ev.Index)
			}
			seen[*ev.Index] = true
			open = *ev.Index
			starts++
		case "content_block_delta":
			if ev.Index == nil || *ev.Index != open {
				t.Fatalf("script %v: delta for %v with %d open", script, ev.Index, open)
			}
		case "content_block_stop":
			if ev.Index == nil || *ev.Index != open {
				t.Fatalf("script %v: stop for %v with %d open", script, ev.Index, open)
			}
			open = -1
			stops++
		}
	}
	if open != -1 {
		t.Fatalf("script %v: block %d was never closed:\n%s", script, open, body)
	}
	if starts != stops {
		t.Fatalf("script %v: %d starts, %d stops", script, starts, stops)
	}
}
