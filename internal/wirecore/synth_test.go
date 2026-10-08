// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
)

func richResponse() *llm.Response {
	return &llm.Response{
		ID:       "resp_1",
		Provider: "stub",
		Model:    "m",
		Choices: []llm.Choice{{
			Index: 0,
			Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
				&llm.Thinking{Text: "first thought", Signature: "sig-1"},
				&llm.ToolUse{ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":"go"}`)},
				&llm.Thinking{Text: "second thought", Signature: "sig-2"},
				&llm.Text{Text: "here is the answer"},
				&llm.ToolUse{ID: "call_b", Name: "fetch", Args: json.RawMessage(`{"url":"x"}`)},
				&llm.Refusal{Text: "but not that part"},
				&llm.Image{MIME: "image/png", Data: []byte("png-bytes")},
			}},
			Stop: llm.StopToolUse,
		}},
		Usage: llm.Usage{Prompt: 120, Completion: 45, CacheRead: 100, Reported: true, Turns: 1},
	}
}

func TestSynthesiseFoldRoundTrip(t *testing.T) {
	orig := richResponse()
	got, err := Fold(Synthesise(orig, llm.Renders{}), FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("choices = %d", len(got.Choices))
	}
	want := orig.Choices[0].Message.Content
	have := got.Choices[0].Message.Content
	if len(have) != len(want) {
		t.Fatalf("parts = %d, want %d", len(have), len(want))
	}
	for i := range want {
		if have[i].Kind() != want[i].Kind() {
			t.Fatalf("part %d = %s, want %s", i, have[i].Kind(), want[i].Kind())
		}
	}
	if have[0].(*llm.Thinking).Signature != "sig-1" || have[2].(*llm.Thinking).Signature != "sig-2" {
		t.Fatal("per-block signatures crossed or lost in the round trip")
	}
	if llm.RefusalOf(have) != "but not that part" {
		t.Fatal("refusal lost in the round trip")
	}
	tools := llm.ToolUses(have)
	if len(tools) != 2 || string(tools[1].Args) != `{"url":"x"}` {
		t.Fatalf("tool calls = %+v", tools)
	}
	if string(have[6].(*llm.Image).Data) != "png-bytes" {
		t.Fatal("image lost in the round trip")
	}
	if got.Choices[0].Stop != llm.StopToolUse {
		t.Fatalf("stop = %q", got.Choices[0].Stop)
	}
	if got.Usage.Prompt != 120 || got.Usage.Completion != 45 || got.Usage.CacheRead != 100 {
		t.Fatalf("usage = %+v", got.Usage)
	}
}

func TestSynthesiseIndexesToolCalls(t *testing.T) {
	frames := SynthesiseFrames(richResponse())
	var seen []int
	for _, f := range frames {
		for _, p := range f.Delta {
			if tu, ok := p.(*llm.ToolUse); ok {
				if tu.Index == nil {
					t.Fatal("a streamed tool call without an index cannot be reassembled")
				}
				seen = append(seen, *tu.Index)
			}
		}
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("indices = %v", seen)
	}
}

func TestSynthesiseDoesNotMutateTheResponse(t *testing.T) {
	orig := richResponse()
	SynthesiseFrames(orig)
	for _, p := range orig.Choices[0].Message.Content {
		if tu, ok := p.(*llm.ToolUse); ok && tu.Index != nil {
			t.Fatal("synthesis wrote an index back into the caller's response")
		}
	}
}

func TestSynthesiseEmitsUsageLast(t *testing.T) {
	frames := SynthesiseFrames(richResponse())
	last := frames[len(frames)-1]
	if last.Usage == nil || !last.TurnEnd {
		t.Fatalf("last frame = %+v: usage must come after the finish", last)
	}
	for _, f := range frames[:len(frames)-1] {
		if f.Usage != nil {
			t.Fatal("usage emitted before the finish event")
		}
	}
}

func TestSynthesiseCarriesPromptBlock(t *testing.T) {
	resp := &llm.Response{Block: &llm.Block{Reason: "SAFETY"}, Stop: llm.StopContentFilter}
	got, err := Fold(Synthesise(resp, llm.Renders{}), FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Block == nil || got.Block.Reason != "SAFETY" || got.Stop != llm.StopContentFilter {
		t.Fatalf("response = %+v", got)
	}
	if len(got.Choices) != 0 {
		t.Fatal("a blocked prompt gained a candidate")
	}
}

// A synthesised stream has the whole answer before any of it is delivered, so
// what the answer noted is known before the first frame, where a policy can
// still refuse it, and is counted once.
func TestASynthesisedStreamKnowsItsNotesBeforeItsFirstFrame(t *testing.T) {
	resp := richResponse()
	resp.Diagnostics = llm.Diagnostics{{
		Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageResponse, Severity: llm.SeverityLoss,
	}}
	s := Synthesise(resp, llm.Renders{})
	if !s.Diagnostics().Has(llm.DropImage) {
		t.Fatal("the answer's notes arrive only with its last frame, after everything else was delivered")
	}
	got, err := Fold(s, FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if n := len(got.Diagnostics.Find(llm.DropImage)); n != 1 {
		t.Fatalf("the note was counted %d times", n)
	}
}

func TestSynthesiseEmptyAnswerKeepsOneChoice(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{Index: 0, Message: llm.Message{Role: llm.RoleAssistant}, Stop: llm.StopEnd}}}
	got, err := Fold(Synthesise(resp, llm.Renders{}), FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Choices) != 1 || got.Choices[0].Stop != llm.StopEnd {
		t.Fatalf("response = %+v: an empty answer is still an answer", got)
	}
}

// A refusal ends a synthesised stream before its last frame, the one carrying
// the answer's usage, so booking what it cost cannot wait for that frame to
// be absorbed: the call was still sent and billed.
func TestSynthesisedStreamKeepsItsUsageWhenRefused(t *testing.T) {
	resp := richResponse()
	resp.Diagnostics = llm.Diagnostics{{
		Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageResponse, Severity: llm.SeverityLoss,
	}}
	s := Synthesise(resp, llm.Renders{})
	if err := s.Prime(); err != nil {
		t.Fatalf("prime: %v", err)
	}
	s.Enforce(llm.LossPolicy{Strict: true}, llm.Renders{})
	if e := llm.AsError(s.Err()); e == nil || e.Kind != llm.KindLoss {
		t.Fatalf("err = %v, want a loss refusal", s.Err())
	}
	got := s.Usage().Unmeasured()
	if got.Prompt != resp.Usage.Prompt || got.Completion != resp.Usage.Completion || !got.Reported {
		t.Fatalf("usage = %+v, want the answer's own %+v", got, resp.Usage)
	}
	for _, a := range got.Anomalies {
		if a.Code == llm.AnomalyNotMeasurable {
			t.Fatalf("usage carries %s: a sent, reported call must not book as if the backend reported nothing", a.Code)
		}
	}
}

// Renders.Unrendered weighs the whole answer, every choice and every part of
// it, not only what a stream has buffered so far. A synthesised stream must
// be held to that before its first frame too, so strict either delivers the
// whole of it or none, never a prefix ending where the loss was found.
func TestSynthesiseRefusesTheWholeAnswerRatherThanPartOfIt(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
		&llm.Text{Text: "per the result, "},
		&llm.Raw{Vendor: "bedrock", Type: "x", JSON: json.RawMessage(`{}`)},
		&llm.Text{Text: "geese honk"},
	}}, Stop: llm.StopEnd}}}
	to := llm.Renders{ToolCalls: true, Thinking: true, Candidates: 1, Raw: []string{"anthropic"}}

	s := Synthesise(resp, to)
	if err := s.Prime(); err != nil {
		t.Fatalf("prime: %v", err)
	}
	s.Enforce(llm.LossPolicy{Strict: true}, to)
	if e := llm.AsError(s.Err()); e == nil || e.Kind != llm.KindLoss {
		t.Fatalf("err = %v, want a loss refusal before any frame", s.Err())
	}

	var delivered string
	for f := range s.Frames() {
		delivered += llm.TextOf(f.Delta)
	}
	if delivered != "" {
		t.Fatalf("delivered %q: a stream strict refuses must not hand over part of the answer first", delivered)
	}
}

// An answer with no candidate came back whole all the same, whether it carried
// only what it cost or only why it was stopped, so its stream ends with a
// reason rather than reading as one cut off.
func TestAnAnswerWithNoCandidateStillSaysWhyItEnded(t *testing.T) {
	for _, tc := range []struct {
		what string
		resp *llm.Response
		stop llm.StopReason
	}{
		{"only what it cost", &llm.Response{Usage: llm.Usage{Prompt: 3, Reported: true}}, llm.StopEnd},
		{"only why it was stopped", &llm.Response{Block: &llm.Block{Reason: "safety"}}, llm.StopContentFilter},
		{"a reason of its own", &llm.Response{Block: &llm.Block{Reason: "safety"}, Stop: llm.StopRefusal}, llm.StopRefusal},
	} {
		var stop llm.StopReason
		for _, f := range SynthesiseFrames(tc.resp) {
			if f.Stop != llm.StopNone {
				stop = f.Stop
			}
		}
		if stop != tc.stop {
			t.Errorf("%s: stop = %q, want %q", tc.what, stop, tc.stop)
		}
		s := Synthesise(tc.resp, llm.Renders{})
		if err := s.Prime(); err != nil {
			t.Errorf("%s: err = %v: a whole answer read as one cut off", tc.what, err)
		}
		_ = s.Close()
	}
}
