// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// A loss noted only after the whole stream has been folded is caught too
// late for strict to refuse it in place of what it lost: everything after
// it has already gone out. Regression test for r3-wire-3.
//
// The call whose own signature loss causes the refusal is only ever
// written by the deferred tool flush, so it must not reach the client
// either, even though it is buffered before the refusal is known.
// Regression test for r3-wire-3-gap-anthropicmsg-flushtools-unconditional.
func TestASignatureLossIsRefusedBeforeLaterContentIsWritten(t *testing.T) {
	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.ToolUse{ID: "call_1", Name: "f", Args: json.RawMessage(`{}`), Index: &i0,
			Signature: "SIG-abc", Signer: llm.FamilyGoogle}}},
		{Delta: []llm.Part{&llm.Text{Text: "LATER-TEXT"}}},
		{Stop: llm.StopEnd},
	})
	c := New(Options{})
	s.Enforce(llm.LossPolicy{Strict: true}, c.Renders())
	w := &wire.BufferWriter{}
	if _, err := c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err == nil {
		t.Fatal("a signature this wire cannot carry must be refused under strict")
	}
	body := w.Body.String()
	if strings.Contains(body, "LATER-TEXT") {
		t.Fatalf("body = %s: content after the loss must not be written once it is refused", body)
	}
	if strings.Contains(body, "tool_use") || strings.Contains(body, "call_1") {
		t.Fatalf("body = %s: the tool call whose own signature loss caused the refusal must not be delivered either", body)
	}
}

// A part whose own loss strict refuses must not be written ahead of the error
// that refuses it: noting the loss and writing the part in one step, then
// checking, puts on the wire exactly what was refused. Regression test for
// r4-wire-3.
func TestARefusedPartIsNotWrittenBeforeItsError(t *testing.T) {
	for _, tc := range []struct {
		name string
		part llm.Part
	}{
		{"signed text", &llm.Text{Text: "REFUSED-PART", Signature: "SIG", Signer: llm.FamilyGoogle}},
		{"refusal as text", &llm.Refusal{Text: "REFUSED-PART"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "m"}, []*llm.Frame{
				{Delta: []llm.Part{tc.part}},
				{Stop: llm.StopEnd},
			})
			c := New(Options{})
			s.Enforce(llm.LossPolicy{Strict: true}, c.Renders())
			w := &wire.BufferWriter{}
			if _, err := c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err == nil {
				t.Fatal("a loss strict will not take must be refused")
			}
			if body := w.Body.String(); strings.Contains(body, "REFUSED-PART") {
				t.Fatalf("body = %s: the part strict refused must not be written ahead of the error", body)
			}
		})
	}
}
