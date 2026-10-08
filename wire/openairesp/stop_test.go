// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// A stop this wire has no spelling for is not a completed answer said
// cleanly: every other wire notes that it had to guess, and strict refuses
// the guess. Regression test for r4-wire-6.
func TestAnUnknownStopIsNotedRatherThanCompleted(t *testing.T) {
	resp := &llm.Response{ID: "r", Model: "m", Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		Stop:    llm.StopUnknown,
	}}}
	_, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.DropUnknownBlock); len(d) != 1 || d[0].Field != string(llm.StopUnknown) {
		t.Fatalf("diagnostics = %+v: the unspelt stop must be noted", diags)
	}

	s := llm.StaticStream(llm.StreamMeta{ID: "r", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopUnknown},
	})
	c := New(Options{})
	s.Enforce(llm.LossPolicy{Strict: true}, c.Renders())
	w := &wire.BufferWriter{}
	diags, err = c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if err == nil || !diags.Has(llm.DropUnknownBlock) {
		t.Fatalf("err = %v, diagnostics = %v: strict must refuse a stop the stream cannot spell", err, diags.Codes())
	}
}

// A cancelled response has its own status here, and is not told as complete.
func TestACancelledStopIsSpeltAsCancelled(t *testing.T) {
	resp := &llm.Response{ID: "r", Model: "m", Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		Stop:    llm.StopCancelled,
	}}}
	out, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	if body.Status != "cancelled" || diags.Has(llm.DropUnknownBlock) {
		t.Fatalf("status = %q, diagnostics = %v: want cancelled, cleanly", body.Status, diags.Codes())
	}
}

// No stream event ends a cancelled response, so a stream does not invent one:
// it says the stop had no spelling, as it does for any other.
func TestACancelledStreamIsNotedRatherThanGivenAnEventOfItsOwn(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "r", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopCancelled},
	})
	w := &wire.BufferWriter{}
	diags, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.DropUnknownBlock); len(d) != 1 || d[0].Field != string(llm.StopCancelled) {
		t.Fatalf("diagnostics = %+v: the unspelt stop must be noted", diags)
	}
	if body := w.Body.String(); strings.Contains(body, "response.cancelled") {
		t.Fatalf("body = %s: no stream event ends a cancelled response", body)
	}
}
