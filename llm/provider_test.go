// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"encoding/json"
	"testing"
)

// An attempt refuses, before anything is sent, a loss noted of it that its
// policy will not take, and says whose it was. What the policy takes, or was
// never a loss, lets it go.
func TestAnAttemptRefusesALossItsPolicyWillNotTake(t *testing.T) {
	loss := Diagnostic{Code: DropExtraField, Class: ClassContent, Stage: StageRequest, Severity: SeverityInfo, Field: "name"}
	capability := Diagnostic{Code: CapCacheBreakpoint, Class: ClassCapability, Stage: StageRequest, Severity: SeverityLoss}

	var noted Diagnostics
	a := &Attempt{Provider: "p", Model: "m", Policy: LossPolicy{Strict: true}, Recorder: &noted}
	if err := a.Check(); err != nil {
		t.Fatalf("err = %v: nothing has been noted", err)
	}
	a.Note(capability)
	if err := a.Check(); err != nil {
		t.Fatalf("err = %v: strict holds content, not capability", err)
	}
	a.Note(loss)
	e := AsError(a.Check())
	if e == nil || e.Kind != KindLoss || e.Sent || e.Code != DropExtraField || e.Provider != "p" || e.Model != "m" {
		t.Fatalf("err = %+v", e)
	}
	if len(noted) != 2 || noted[1].Severity != SeverityFatal {
		t.Fatalf("noted = %+v: the record keeps what the policy made of it", noted)
	}

	allowing := &Attempt{Policy: LossPolicy{Strict: true, Allow: []string{DropExtraField}}}
	allowing.Note(loss)
	if err := allowing.Check(); err != nil {
		t.Fatalf("err = %v: a code the policy allows is not refused", err)
	}

	unrecorded := &Attempt{Policy: LossPolicy{Deny: []string{CapCacheBreakpoint}}}
	unrecorded.Note(capability)
	if e := AsError(unrecorded.Check()); e == nil || e.Kind != KindLoss {
		t.Fatalf("err = %v: an attempt with nowhere to record a note still holds it to its policy", e)
	}
}

// A destination that declared nothing takes every part as it is. One that did
// loses a block carried from a wire it does not write, and reasoning it has no
// field for.
func TestUnrenderedIsWhatTheDestinationHasNoFieldFor(t *testing.T) {
	resp := &Response{Choices: []Choice{{Message: Message{Content: []Part{
		&Raw{Vendor: "anthropic", Type: "web_search_tool_result", JSON: json.RawMessage(`{}`)},
		&Thinking{Text: "because"},
		&Thinking{Signature: "sig"},
		&Text{Text: "so"},
	}}}}}
	if got := (Renders{}).Unrendered(resp); len(got) != 0 {
		t.Fatalf("got = %+v: a library caller takes parts as they are", got)
	}
	chat := Renders{ToolCalls: true, Candidates: 128}
	got := chat.Unrendered(resp)
	if len(got) != 2 || got[0].Code != DropUnknownBlock || got[0].Field != "web_search_tool_result" ||
		got[1].Code != DropThinking || got[0].Severity != SeverityLoss || got[0].Class != ClassContent {
		t.Fatalf("got = %+v", got)
	}
	own := Renders{Thinking: true, ToolCalls: true, Candidates: 1, Raw: []string{"anthropic"}}
	if got := own.Unrendered(resp); len(got) != 0 {
		t.Fatalf("got = %+v: the block's own wire writes it out whole", got)
	}
	if got := (LossPolicy{Strict: true}).Check(chat.Unrendered(resp)); AsError(got) == nil {
		t.Fatal("strict takes a block the destination drops as the loss it is")
	}
}

// A destination with no field for an image or a tool call loses one handed to
// it as surely as reasoning it cannot carry, so it is weighed before anything
// is delivered rather than found by the encoder after the text before it went.
func TestUnrenderedCountsImagesAndToolCallsTheDestinationCannotCarry(t *testing.T) {
	resp := &Response{Choices: []Choice{{Message: Message{Content: []Part{
		&Text{Text: "here"},
		&Image{MIME: "image/png", Data: []byte("x")},
		&ToolUse{ID: "1", Name: "act", Args: json.RawMessage(`{}`)},
	}}}}}
	chat := Renders{ToolCalls: true, Candidates: 128}
	if got := chat.Unrendered(resp); len(got) != 1 || got[0].Code != DropImage || got[0].Severity != SeverityLoss {
		t.Fatalf("chat: got = %+v", got)
	}
	generate := Renders{Thinking: true, Candidates: 1}
	if got := generate.Unrendered(resp).Codes(); len(got) != 2 || got[0] != DropImage || got[1] != DropToolCall {
		t.Fatalf("generate: got = %v", got)
	}
	gemini := Renders{Thinking: true, ToolCalls: true, Images: true, Candidates: 8}
	if got := gemini.Unrendered(resp); len(got) != 0 {
		t.Fatalf("gemini: got = %+v: a destination with a field for each loses nothing", got)
	}
	if got := (Renders{}).Unrendered(resp); len(got) != 0 {
		t.Fatalf("got = %+v: a library caller takes parts as they are", got)
	}
}

// A stream refused for a part its destination cannot write out is refused
// before any encoder meets that part, so the stream itself must say what was
// lost; otherwise the loss survives only in the error's message, and the
// diagnostics a proxy reports and books never name it.
func TestAStreamRefusedForAnUnrenderedPartSaysWhatItLost(t *testing.T) {
	for _, held := range []bool{true, false} {
		frames := []*Frame{
			{Delta: []Part{&Image{MIME: "image/png", Data: []byte("x")}}},
			{Delta: []Part{&Text{Text: "after"}}},
			{Stop: StopEnd},
		}
		s := StaticStream(StreamMeta{Model: "m"}, frames)
		if held {
			if err := s.Prime(); err != nil {
				t.Fatal(err)
			}
		}
		s.Enforce(LossPolicy{Strict: true}, Renders{ToolCalls: true, Candidates: 1})
		for range s.Frames() {
		}
		if s.Err() == nil {
			t.Fatalf("held=%v: an image the destination has no field for was not refused", held)
		}
		if d := s.Diagnostics().Find(DropImage); len(d) != 1 || d[0].Severity != SeverityFatal {
			t.Fatalf("held=%v: diagnostics = %+v: the refused image is not said, or not said once", held, s.Diagnostics())
		}
	}
}
