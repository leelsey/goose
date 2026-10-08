// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// An answer cut off by the context window is not one cut off by max_tokens: a
// client told the latter raises the ceiling, where the former needs its context
// made smaller. Claude's own wire has a spelling for it, so it keeps it, on the
// unstreamed answer and the stream alike. Regression test for r4-wire-9.
func TestAContextOverflowKeepsItsOwnSpelling(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "cut"}}},
		Stop:    llm.StopContextOverflow,
	}}}
	out, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.StopReason != "model_context_window_exceeded" || diags.Has(llm.DropUnknownBlock) {
		t.Fatalf("stop_reason = %q, diagnostics = %v: want model_context_window_exceeded, cleanly", body.StopReason, diags.Codes())
	}

	s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "cut"}}},
		{Stop: llm.StopContextOverflow},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	var reason any
	for _, e := range parseEvents(t, w.Body.String()) {
		if e.name == "message_delta" {
			reason = e.body["delta"].(map[string]any)["stop_reason"]
		}
	}
	if reason != "model_context_window_exceeded" {
		t.Fatalf("streamed stop_reason = %v, want model_context_window_exceeded", reason)
	}
}
