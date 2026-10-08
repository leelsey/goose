// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollamaapi

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
func TestASignatureLossIsRefusedBeforeLaterContentIsWritten(t *testing.T) {
	for _, tc := range []struct {
		name string
		c    wire.Codec
	}{
		{"chat", NewChat(Options{})},
		{"generate", NewGenerate(Options{})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
				{Delta: []llm.Part{&llm.Thinking{Text: "plan", Signature: "SIG", Signer: llm.FamilyAnthropic}}},
				{Delta: []llm.Part{&llm.Text{Text: "LATER-TEXT"}}},
				{Stop: llm.StopEnd},
			})
			renders := tc.c.(interface{ Renders() wire.Renders }).Renders()
			s.Enforce(llm.LossPolicy{Strict: true}, renders)
			w := &wire.BufferWriter{}
			if _, err := tc.c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err == nil {
				t.Fatal("a signature this wire cannot carry must be refused under strict")
			}
			if strings.Contains(w.Body.String(), "LATER-TEXT") {
				t.Fatalf("body = %s: content after the loss must not be written once it is refused", w.Body.String())
			}
		})
	}
}

// A broken tool call refused under strict must not still follow the error
// it caused onto the wire in the closing line. Regression test for
// r3-wire-4.
func TestChatRefusedFinalLineWithholdsWhatItRefused(t *testing.T) {
	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "Checking."}}},
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Name: "rm", Args: json.RawMessage(`{"path":"/tmp/x","recursive":tr`)}}},
		{Stop: llm.StopToolUse},
	})
	c := NewChat(Options{})
	s.Enforce(llm.LossPolicy{Strict: true}, c.Renders())
	w := &wire.BufferWriter{}
	if _, err := c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err == nil {
		t.Fatal("a broken tool call's arguments must be refused under strict")
	}
	body := w.Body.String()
	if strings.Contains(body, "tool_calls") {
		t.Fatalf("body = %s: the call strict refused must not still be delivered after the error", body)
	}
	errAt := strings.Index(body, `"error"`)
	doneAt := strings.LastIndex(body, `"done":true`)
	if errAt < 0 || doneAt < errAt {
		t.Fatalf("body = %s: want the error line before the closing line", body)
	}
}

// Redacted thinking has no text, so the destination's declared renders do not
// account for it, and a tool result has no place on an assistant turn: the
// unstreamed answer says both are lost, and the stream must say so too, before
// the frame they came on is written. Regression test for r4-wire-2.
func TestRedactedThinkingAndToolResultsAreNotedAsLost(t *testing.T) {
	for _, tc := range []struct {
		name string
		c    wire.Codec
		part llm.Part
		code string
	}{
		{"chat redacted", NewChat(Options{}), &llm.Thinking{Redacted: []byte("opaque")}, llm.DropThinking},
		{"chat tool result", NewChat(Options{}), &llm.ToolResult{ID: "t", Content: []llm.Part{&llm.Text{Text: "x"}}}, llm.DropToolResult},
		{"generate redacted", NewGenerate(Options{}), &llm.Thinking{Redacted: []byte("opaque")}, llm.DropThinking},
		{"generate tool result", NewGenerate(Options{}), &llm.ToolResult{ID: "t", Content: []llm.Part{&llm.Text{Text: "x"}}}, llm.DropToolResult},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := &llm.Response{Choices: []llm.Choice{{
				Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{tc.part, &llm.Text{Text: "hi"}}},
				Stop:    llm.StopEnd,
			}}}
			_, diags, err := tc.c.EncodeResponse(resp, wire.Options{})
			if err != nil {
				t.Fatal(err)
			}
			if !diags.Has(tc.code) {
				t.Fatalf("unary diagnostics = %v, want %s", diags.Codes(), tc.code)
			}

			s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
				{Delta: []llm.Part{tc.part, &llm.Text{Text: "LATER-TEXT"}}},
				{Stop: llm.StopEnd},
			})
			renders := tc.c.(interface{ Renders() wire.Renders }).Renders()
			s.Enforce(llm.LossPolicy{Strict: true}, renders)
			w := &wire.BufferWriter{}
			diags, err = tc.c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
			if err == nil || !diags.Has(tc.code) {
				t.Fatalf("err = %v, diagnostics = %v: strict must refuse the stream's %s", err, diags.Codes(), tc.code)
			}
			if strings.Contains(w.Body.String(), "LATER-TEXT") {
				t.Fatalf("body = %s: the frame holding the loss must not be written", w.Body.String())
			}
		})
	}
}

// A call this endpoint cannot carry is lost in the frame it arrives in, so
// strict refuses it there, before any text after it goes out. Noted only once
// the stream had ended, the loss let that text through first. A stream that
// was nothing but calls is still refused outright. Regression test for
// r4-wire-7.
func TestGenerateRefusesACallWhereItArrives(t *testing.T) {
	i0 := 0
	call := func() *llm.ToolUse {
		return &llm.ToolUse{ID: "c", Name: "f", Args: json.RawMessage(`{}`), Index: &i0}
	}
	c := NewGenerate(Options{})
	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{call()}},
		{Delta: []llm.Part{&llm.Text{Text: "LATER-TEXT"}}},
		{Stop: llm.StopToolUse},
	})
	s.Enforce(llm.LossPolicy{Strict: true}, c.Renders())
	w := &wire.BufferWriter{}
	diags, err := c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if err == nil {
		t.Fatal("a call this endpoint cannot carry must be refused under strict")
	}
	if strings.Contains(w.Body.String(), "LATER-TEXT") {
		t.Fatalf("body = %s: text after the lost call must not be written once it is refused", w.Body.String())
	}
	// The stream itself refuses a call its destination declares no field for,
	// before the encoder meets it, so the loss is the stream's to say, and the
	// encoder hands back the stream's diagnostics beside its own.
	if d := diags.Find(llm.DropToolCall); len(d) != 1 {
		t.Fatalf("diagnostics = %+v: the lost call is noted once", diags)
	}

	// Under a policy that takes the loss, text after the call is what the
	// client gets, and the call is still said to be lost.
	s = llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{call()}},
		{Delta: []llm.Part{&llm.Text{Text: "LATER-TEXT"}}},
		{Stop: llm.StopToolUse},
	})
	s.Enforce(llm.LossPolicy{}, c.Renders())
	w = &wire.BufferWriter{}
	diags, err = c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if err != nil || !strings.Contains(w.Body.String(), "LATER-TEXT") {
		t.Fatalf("err = %v, body = %s: a loss the policy takes must not end the stream", err, w.Body.String())
	}
	if d := diags.Find(llm.DropToolCall); len(d) != 1 || d[0].Severity != llm.SeverityLoss {
		t.Fatalf("diagnostics = %+v", diags)
	}

	// Nothing but calls is no answer at all.
	s = llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{call()}},
		{Delta: []llm.Part{&llm.ToolUse{ID: "d", Name: "g", Args: json.RawMessage(`{}`)}}},
		{Stop: llm.StopToolUse},
	})
	s.Enforce(llm.LossPolicy{}, c.Renders())
	diags, err = c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
	if err == nil {
		t.Fatal("an answer that was nothing but calls must be refused")
	}
	if d := diags.Find(llm.DropToolCall); len(d) != 1 || d[0].Count != 2 || d[0].Severity != llm.SeverityFatal {
		t.Fatalf("diagnostics = %+v: want one fatal note counting both calls", diags)
	}
}
