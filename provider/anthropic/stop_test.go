// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

// Claude's model_context_window_exceeded stop must read as a truncation, not
// as an ordinary clean end: an answer cut off by the window is not the same
// as one the model chose to stop. Regression test for r3-wire-2.
func TestModelContextWindowExceededReadsAsContextOverflow(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"cut off"}],
			"stop_reason":"model_context_window_exceeded","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Stop != llm.StopContextOverflow {
		t.Fatalf("stop = %q, want context_overflow: a window cut-off must not read as a clean end", resp.Choices[0].Stop)
	}

	sp, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range []string{
			`event: message_start` + "\n" + `data: {"message":{"id":"msg_9","model":"claude-test"}}`,
			`event: content_block_start` + "\n" + `data: {"index":0,"content_block":{"type":"text","text":""}}`,
			`event: content_block_delta` + "\n" + `data: {"index":0,"delta":{"type":"text_delta","text":"cut off"}}`,
			`event: content_block_stop` + "\n" + `data: {"index":0}`,
			`event: message_delta` + "\n" + `data: {"delta":{"stop_reason":"model_context_window_exceeded"},"usage":{"output_tokens":1}}`,
			`event: message_stop` + "\n" + `data: {}`,
		} {
			_, _ = w.Write([]byte(line + "\n\n"))
		}
	})
	s, err := sp.Stream(context.Background(), attempt(&llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}))
	if err != nil {
		t.Fatal(err)
	}
	sresp, err := wirecore.Fold(s, wirecore.FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if sresp.Choices[0].Stop != llm.StopContextOverflow {
		t.Fatalf("streamed stop = %q, want context_overflow", sresp.Choices[0].Stop)
	}
}

// The same answer, replayed to an Anthropic client on this Anthropic route,
// must carry Claude's own spelling for it rather than end_turn or max_tokens,
// and without the drop.unknown_block that end_turn's fallback would leave
// behind: a client told max_tokens raises a ceiling where it needs a smaller
// context.
func TestModelContextWindowExceededReachesAnAnthropicClientAsItself(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"cut off"}],
			"stop_reason":"model_context_window_exceeded","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	out, diags, err := anthropicmsg.New(anthropicmsg.Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if diags.Has(llm.DropUnknownBlock) {
		t.Fatalf("diagnostics = %+v: a mapped stop reason is not an unknown one", diags)
	}
	var body struct {
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.StopReason != "model_context_window_exceeded" {
		t.Fatalf("stop_reason = %q, want model_context_window_exceeded: end_turn would read as a clean answer, max_tokens as a ceiling to raise", body.StopReason)
	}
}
