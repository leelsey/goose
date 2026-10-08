// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"net/http"
	"runtime"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// discard is a client that reads nothing back, so what the encoder allocates
// is all that is measured.
type discard struct{}

func (discard) Open(int, http.Header) error { return nil }
func (discard) Write(p []byte) (int, error) { return len(p), nil }
func (discard) Flush() error                { return nil }

// A thinking block waits for its signature, and every delta added to it by
// copying all that came before costs the square of how many there are: tens
// of thousands of small ones cost gigabytes before the client sees a word.
// Regression test for r4-wire-4.
func TestAHeldThinkingBlockCostsWhatItHolds(t *testing.T) {
	const n = 80000
	frames := make([]*llm.Frame, 0, n+3)
	for range n {
		frames = append(frames, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "12345678"}}})
	}
	frames = append(frames,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "SIG", Signer: llm.FamilyAnthropic}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "done"}}},
		&llm.Frame{Stop: llm.StopEnd})
	s := llm.StaticStream(llm.StreamMeta{Model: "gemini"}, frames)

	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, discard{}); err != nil {
		t.Fatal(err)
	}
	runtime.ReadMemStats(&after)
	if spent := after.TotalAlloc - before.TotalAlloc; spent > 256<<20 {
		t.Fatalf("holding %d thinking deltas allocated %d MiB: it must cost what it holds", n, spent>>20)
	}
}

// What may wait for a signature is bounded. A block that outgrows it goes out
// unsigned rather than wait, the loss is said once, and the signature that
// follows lands only on what came after: never on text it did not sign.
func TestAThinkingBlockThatOutgrowsTheHoldGoesOutUnsigned(t *testing.T) {
	h := newThoughtHold(8)
	var out []llm.Part
	full := 0
	for _, v := range []*llm.Thinking{
		{Text: "abcd"}, {Text: "efgh"}, {Text: "ijkl"}, {Text: "mnop"},
		{Signature: "SIG", Signer: llm.FamilyAnthropic},
	} {
		ready, over := h.join(0, v)
		out = append(out, ready...)
		if over {
			full++
		}
	}
	if full != 1 {
		t.Fatalf("overflow reported %d times, want once", full)
	}
	var texts []string
	for i, p := range out {
		th := p.(*llm.Thinking)
		texts = append(texts, th.Text)
		if th.Signature != "" && i != len(out)-1 {
			t.Fatalf("parts = %+v: only the last, which the signature arrived on, is signed", out)
		}
	}
	if got := strings.Join(texts, "|"); got != "abcdefgh|ijklmnop" {
		t.Fatalf("parts = %q, want the held text flushed whole when the next delta would not fit", got)
	}
	if h.size != 0 || len(h.held) != 0 {
		t.Fatalf("hold = %+v: a signed block leaves nothing waiting", h)
	}
}

// The part that outgrows the hold, said through the stream: the client is told
// the signature did not travel on the text that went out before it.
func TestAnOutgrownThinkingBlockIsNotedOnTheStream(t *testing.T) {
	big := strings.Repeat("x", maxHeldThought/2+1)
	s := llm.StaticStream(llm.StreamMeta{Model: "gemini"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Thinking{Text: big}}},
		{Delta: []llm.Part{&llm.Thinking{Text: big}}},
		{Delta: []llm.Part{&llm.Thinking{Signature: "SIG", Signer: llm.FamilyAnthropic}}},
		{Stop: llm.StopEnd},
	})
	diags, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, discard{})
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.WireBufferFull); len(d) != 1 {
		t.Fatalf("diagnostics = %v: an outgrown thinking block is a loss, said once", diags.Codes())
	}
}
