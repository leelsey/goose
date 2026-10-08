// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"os"
	"testing"

	"github.com/leelsey/goose/llm"
)

// The binary restates a message one block at a time, a line as each block
// closes and all of them before the message is finished, so every line of a
// message the deltas carried is a copy, not only the first.
func TestEveryBlockTheDeltasCarriedArrivesOnce(t *testing.T) {
	p := provider(t, Claude, "fake-claude-blocks.sh", nil)
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	blocksAnswered(t, resp.Choices[0].Message.Content)

	u := resp.Usage
	if u.Prompt != 1010+1120 || u.CacheRead != 2100 || u.Completion != 35 {
		t.Fatalf("usage = %d/%d/%d, want 2130/2100/35: each line restates what message_start announced", u.Prompt, u.CacheRead, u.Completion)
	}
	if u.Turns != 2 {
		t.Fatalf("turns = %d: two messages were billed, however many lines restated them", u.Turns)
	}
	if len(u.Anomalies) > 0 {
		t.Fatalf("anomalies = %+v: the turns and the summary agree once nothing is counted twice", u.Anomalies)
	}

	text, _ := drain(t, p)
	if text != "Let me look.Found it." {
		t.Fatalf("streamed text = %q: the deltas delivered it, so no line may deliver it again", text)
	}
}

// Without the deltas the lines are the message itself, and a message is one
// turn however many lines it takes.
func TestAMessagePrintedABlockAtATimeIsOneTurn(t *testing.T) {
	p := provider(t, Claude, "fake-claude-blocks.sh", func(o *Options) {
		o.Env = append(os.Environ(), "GOOSE_TEST_WHOLE=1")
	})
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	blocksAnswered(t, resp.Choices[0].Message.Content)
	if u := resp.Usage; u.Turns != 2 || u.Prompt != 1010+1120 || u.CacheRead != 2100 {
		t.Fatalf("usage = %d turns, %d/%d: each message's figures are counted once", u.Turns, u.Prompt, u.CacheRead)
	}
}

func blocksAnswered(t *testing.T, parts []llm.Part) {
	t.Helper()
	var thinking []*llm.Thinking
	for _, part := range parts {
		if th, ok := part.(*llm.Thinking); ok {
			thinking = append(thinking, th)
		}
	}
	if len(thinking) != 1 || thinking[0].Text != "hmm" || thinking[0].Signature != "sig" {
		t.Fatalf("thinking = %+v", thinking)
	}
	if got := llm.TextOf(parts); got != "Let me look.Found it." {
		t.Fatalf("text = %q", got)
	}
	calls := llm.ToolUses(parts)
	if len(calls) != 1 || calls[0].ID != "toolu_1" || string(calls[0].Args) != `{"file_path":"/x"}` {
		t.Fatalf("calls = %+v", calls)
	}
}
