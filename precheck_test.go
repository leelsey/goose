// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/agentcli"
	"github.com/leelsey/goose/usage"
)

// agentCLI is an agent CLI provider in front of a stand-in for the claude
// binary.
func agentCLI(t *testing.T, mutate func(*agentcli.Options)) *agentcli.Provider {
	t.Helper()
	bin, err := filepath.Abs(filepath.Join("provider", "agentcli", "testdata", "fake-claude.sh"))
	if err != nil {
		t.Fatal(err)
	}
	o := agentcli.Options{ID: "cli", Kind: agentcli.Claude, Binary: bin, Env: os.Environ(), Grace: time.Second}
	if mutate != nil {
		mutate(&o)
	}
	p, err := agentcli.New(o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close(context.Background()) })
	return p
}

// said is what a call said of itself, and the row it left, once it is over.
func said(t *testing.T, c *Client, mem *usage.Memory, r *llm.Request, stream bool) (llm.Diagnostics, llm.Diagnostics) {
	t.Helper()
	var out llm.Diagnostics
	if stream {
		s, err := c.Stream(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		for range s.Frames() {
		}
		_ = s.Close()
		out = s.Diagnostics()
	} else {
		resp, err := c.Chat(context.Background(), r)
		if err != nil {
			t.Fatal(err)
		}
		out = resp.Diagnostics
	}
	rows := mem.Records()
	if len(rows) == 0 {
		t.Fatal("the call left no row")
	}
	return out, rows[len(rows)-1].Diags
}

// What a provider finds ahead of its gate it notes again once it builds the
// request, as it would without a gate, so the chain notes it only where it
// turns the attempt away. Noted by both, every loss in the prompt was said
// twice, on the answer and on the row.
func TestWhatAPrecheckFoundIsSaidOnce(t *testing.T) {
	lost := llm.DroppedMessage(llm.RoleAssistant, "cli", "this turn carried nothing this backend can read as text")
	for _, stream := range []bool{false, true} {
		mem := usage.NewMemory()
		cli := &prechecking{capped: capped{id: "cli", caps: full()}, diags: llm.Diagnostics{lost}}
		c, err := New(WithProvider(cli), WithUsage(mem))
		if err != nil {
			t.Fatal(err)
		}
		answer, row := said(t, c, mem, req(), stream)
		if n, m := len(answer.Find(llm.DropMessage)), len(row.Find(llm.DropMessage)); n != 1 || m != 1 {
			t.Fatalf("stream %v: the loss is said %d times on the answer and %d on the row", stream, n, m)
		}
	}

	r := &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "hmm"}}},
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
	}}
	for _, stream := range []bool{false, true} {
		mem := usage.NewMemory()
		c, err := New(WithProvider(agentCLI(t, nil)), WithUsage(mem))
		if err != nil {
			t.Fatal(err)
		}
		answer, row := said(t, c, mem, r, stream)
		if n, m := len(answer.Find(llm.DropMessage)), len(row.Find(llm.DropMessage)); n != 1 || m != 1 {
			t.Fatalf("stream %v: the agent CLI's dropped turn is said %d times on the answer and %d on the row", stream, n, m)
		}
	}
}

// A request a provider will not send, whatever the policy, is refused ahead of
// its gate as surely as a loss the policy will not take. Left to the send, it
// waited for a slot it could never use, and behind busy ones came back as the
// wait running out rather than as the request it is.
func TestARequestTheProviderRefusesOutrightNeverQueues(t *testing.T) {
	for _, stream := range []bool{false, true} {
		mem := usage.NewMemory()
		cli := &prechecking{capped: capped{id: "cli", caps: full()},
			err: llm.Errorf(llm.KindInvalidRequest, "a model name cannot begin with a dash")}
		c, err := New(WithProvider(cli), WithUsage(mem))
		if err != nil {
			t.Fatal(err)
		}
		if stream {
			_, err = c.Stream(context.Background(), req())
		} else {
			_, err = c.Chat(context.Background(), req())
		}
		if e := llm.AsError(err); e == nil || e.Kind != llm.KindInvalidRequest || e.Sent || e.Provider != "cli" {
			t.Fatalf("stream %v: err = %+v", stream, err)
		}
		if cli.acquired.Load() || cli.seen != nil {
			t.Fatalf("stream %v: a request the provider was never going to send queued for its slot", stream)
		}
		if rows := mem.Records(); len(rows) != 1 || rows[0].Sent || rows[0].Outcome != llm.OutcomeRejected {
			t.Fatalf("stream %v: rows = %+v: the refusal is a rejected attempt that cost nothing", stream, rows)
		}
	}

	cli := agentCLI(t, func(o *agentcli.Options) { o.MaxConcurrent = 1 })
	busy, err := cli.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer busy()
	mem := usage.NewMemory()
	c, err := New(WithProvider(cli), WithUsage(mem), WithLimits(llm.Limits{Timeout: 200 * time.Millisecond}))
	if err != nil {
		t.Fatal(err)
	}
	flag := req()
	flag.Model = "--yolo"
	empty := &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Thinking{Text: "nothing to say"}}},
	}}
	for _, r := range []*llm.Request{flag, empty} {
		for _, stream := range []bool{false, true} {
			if stream {
				_, err = c.Stream(context.Background(), r)
			} else {
				_, err = c.Chat(context.Background(), r)
			}
			if e := llm.AsError(err); e == nil || e.Kind != llm.KindInvalidRequest || e.Sent {
				t.Fatalf("model %q, stream %v: err = %v: a request the binary is never run for waited for a slot instead",
					r.Model, stream, err)
			}
		}
	}
	rows := mem.Records()
	if len(rows) != 4 || rows[3].Outcome != llm.OutcomeRejected || !llm.Diagnostics(rows[3].Diags).Has(llm.DropMessage) {
		t.Fatalf("rows = %+v: a request refused for having no prompt says what left it with none", rows)
	}
}
