// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

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

// Each delta is held for the closing events of its item. Added to all that
// came before by copying it, the answer costs the square of how many deltas
// it streamed in: tens of thousands of small ones cost gigabytes. Regression
// test for r4-wire-8.
func TestHeldTextCostsWhatItHolds(t *testing.T) {
	const n = 40000
	delta := "12345678"
	for _, tc := range []struct {
		name string
		part func() llm.Part
	}{
		{"text", func() llm.Part { return &llm.Text{Text: delta} }},
		{"refusal", func() llm.Part { return &llm.Refusal{Text: delta} }},
		{"summary", func() llm.Part { return &llm.Thinking{Text: delta} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frames := make([]*llm.Frame, 0, n+1)
			for range n {
				frames = append(frames, &llm.Frame{Delta: []llm.Part{tc.part()}})
			}
			frames = append(frames, &llm.Frame{Stop: llm.StopEnd})
			s := llm.StaticStream(llm.StreamMeta{ID: "r", Model: "m"}, frames)

			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, discard{}); err != nil {
				t.Fatal(err)
			}
			runtime.ReadMemStats(&after)
			if spent := after.TotalAlloc - before.TotalAlloc; spent > 512<<20 {
				t.Fatalf("streaming %d deltas of %d bytes allocated %d MiB: holding them must cost what they hold", n, len(delta), spent>>20)
			}
		})
	}
}

// What is held is what the closing events carry, whole and in order.
func TestHeldTextClosesWhole(t *testing.T) {
	evs := events(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "th"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "ink"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "ans"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: "no"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "wer"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: "pe"}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	got := map[string]string{}
	for _, e := range evs {
		switch e.name {
		case "response.reasoning_summary_text.done", "response.output_text.done":
			got[e.name] = e.body["text"].(string)
		case "response.refusal.done":
			got[e.name] = e.body["refusal"].(string)
		}
	}
	want := map[string]string{
		"response.reasoning_summary_text.done": "think",
		"response.output_text.done":            "answer",
		"response.refusal.done":                "nope",
	}
	for k, v := range want {
		if got[k] != v {
			t.Fatalf("%s = %q, want %q (all: %s)", k, got[k], v, strings.Join(names(evs), ","))
		}
	}
}
