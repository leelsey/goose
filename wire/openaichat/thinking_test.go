// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"context"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// Redacted thinking has no text, so neither the reasoning field nor the
// destination's declared renders account for it, and it went without a word.
// Its loss is said on the unstreamed answer and the stream alike, with or
// without a reasoning field. Regression test for r4-wire-2.
func TestRedactedThinkingIsNotedAsLost(t *testing.T) {
	redacted := func() []llm.Part {
		return []llm.Part{&llm.Thinking{Redacted: []byte("opaque")}, &llm.Text{Text: "hi"}}
	}
	for _, field := range []string{"", "reasoning_content"} {
		c := New(Options{ReasoningField: field})
		resp := &llm.Response{Choices: []llm.Choice{{
			Message: llm.Message{Role: llm.RoleAssistant, Content: redacted()}, Stop: llm.StopEnd,
		}}}
		_, diags, err := c.EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if !diags.Has(llm.DropThinking) {
			t.Fatalf("reasoning field %q: unary diagnostics = %v, want drop.thinking", field, diags.Codes())
		}

		s := llm.StaticStream(llm.StreamMeta{ID: "c", Model: "m"}, []*llm.Frame{
			{Delta: redacted()},
			{Stop: llm.StopEnd},
		})
		diags, err = c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
		if err != nil {
			t.Fatal(err)
		}
		if !diags.Has(llm.DropThinking) {
			t.Fatalf("reasoning field %q: stream diagnostics = %v, want drop.thinking", field, diags.Codes())
		}
	}
}
