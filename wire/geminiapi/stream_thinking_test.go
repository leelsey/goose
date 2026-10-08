// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// A Claude-family backend streams a thinking block as text-only deltas
// followed by a delta that carries only the signature. This wire must not
// hand a client those as two parts, one with the text and an unrelated one
// with the signature: a client that replays them separately would sign the
// wrong block. Regression test for r3-wire-1.
func TestStreamedThinkingKeepsItsSignatureOnItsOwnText(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{Model: "gemini"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Thinking{Text: "I should check "}}},
		{Delta: []llm.Part{&llm.Thinking{Text: "the weather."}}},
		{Delta: []llm.Part{&llm.Thinking{Signature: "SIG123", Signer: llm.FamilyAnthropic}}},
		{Delta: []llm.Part{&llm.Text{Text: "It's sunny."}}},
		{Stop: llm.StopEnd},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}

	var allParts []json.RawMessage
	for _, line := range strings.Split(w.Body.String(), "\n") {
		raw, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var body struct {
			Candidates []struct {
				Content struct {
					Parts []json.RawMessage `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal([]byte(raw), &body); err != nil {
			t.Fatalf("chunk %q: %v", raw, err)
		}
		if len(body.Candidates) > 0 {
			allParts = append(allParts, body.Candidates[0].Content.Parts...)
		}
	}
	turn, err := json.Marshal(allParts)
	if err != nil {
		t.Fatal(err)
	}
	// A client replays exactly the parts it was streamed, on its next turn.
	reqBody := `{"contents":[{"role":"user","parts":[{"text":"weather?"}]},{"role":"model","parts":` + string(turn) + `}]}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(reqBody)})
	if err != nil {
		t.Fatal(err)
	}
	var thoughts []*llm.Thinking
	for _, p := range r.Messages[1].Content {
		if th, ok := p.(*llm.Thinking); ok {
			thoughts = append(thoughts, th)
		}
	}
	if len(thoughts) != 1 {
		t.Fatalf("thinking parts = %d: %+v, want the streamed text joined with its own signature", len(thoughts), thoughts)
	}
	if thoughts[0].Text != "I should check the weather." {
		t.Fatalf("text = %q, want the two streamed deltas joined", thoughts[0].Text)
	}
	if thoughts[0].Signature == "" || thoughts[0].Signer != llm.FamilyAnthropic {
		t.Fatalf("thinking = %+v: the signature that arrived after the text must still land on it", thoughts[0])
	}
}

// A tool call must flush a still-unsigned held thinking block the same way
// any other content does: joinThinking merges purely on ID, and a
// Claude-family backend never sets one on a Thinking part, so a block left
// unsigned across an intervening tool call must not silently absorb a
// later, unrelated block's text and signature. Regression test for
// r3-wire-1-gap-tooluse-does-not-flush-pending-thinking.
func TestToolCallFlushesAPendingUnsignedThinkingBlock(t *testing.T) {
	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{Model: "gemini"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Thinking{Text: "FIRST-THOUGHT"}}},
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Name: "f", Args: json.RawMessage(`{}`)}}},
		{Delta: []llm.Part{&llm.Thinking{Text: "SECOND-THOUGHT", Signature: "SIG2", Signer: llm.FamilyAnthropic}}},
		{Stop: llm.StopToolUse},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}

	var allParts []json.RawMessage
	for _, line := range strings.Split(w.Body.String(), "\n") {
		raw, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var body struct {
			Candidates []struct {
				Content struct {
					Parts []json.RawMessage `json:"parts"`
				} `json:"content"`
			} `json:"candidates"`
		}
		if err := json.Unmarshal([]byte(raw), &body); err != nil {
			t.Fatalf("chunk %q: %v", raw, err)
		}
		if len(body.Candidates) > 0 {
			allParts = append(allParts, body.Candidates[0].Content.Parts...)
		}
	}

	type thought struct {
		Text      string `json:"text"`
		Thought   bool   `json:"thought"`
		Signature string `json:"thoughtSignature"`
	}
	var thoughts []thought
	for _, raw := range allParts {
		var p thought
		if err := json.Unmarshal(raw, &p); err != nil {
			t.Fatal(err)
		}
		if p.Thought {
			thoughts = append(thoughts, p)
		}
	}
	if len(thoughts) != 2 {
		t.Fatalf("thought parts = %d: %+v, want the tool call to flush FIRST-THOUGHT before SECOND-THOUGHT arrives", len(thoughts), thoughts)
	}
	if thoughts[0].Text != "FIRST-THOUGHT" || thoughts[0].Signature != "" {
		t.Fatalf("first thought = %+v: it must go out unsigned, not carrying a later block's signature", thoughts[0])
	}
	if thoughts[1].Text != "SECOND-THOUGHT" || thoughts[1].Signature == "" {
		t.Fatalf("second thought = %+v: it must keep its own signature, not one merged onto the first block's text", thoughts[1])
	}
}

// A broken tool call refused under strict must not still follow the error
// it caused onto the wire in the closing chunk. Regression test for
// r3-wire-4.
func TestRefusedFinalChunkWithholdsWhatItRefused(t *testing.T) {
	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{Model: "gemini"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "Checking."}}},
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Name: "rm", Args: json.RawMessage(`{"path":"/tmp/x","recursive":tr`)}}},
		{Stop: llm.StopToolUse},
	})
	s.Enforce(llm.LossPolicy{Strict: true}, llm.Renders{ToolCalls: true, Candidates: 128})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err == nil {
		t.Fatal("a broken tool call's arguments must be refused under strict")
	}
	body := w.Body.String()
	if strings.Contains(body, "functionCall") {
		t.Fatalf("body = %s: the call strict refused must not still be delivered after the error", body)
	}
	errAt := strings.Index(body, `"error"`)
	doneAt := strings.LastIndex(body, "data: ")
	if errAt < 0 || doneAt < errAt {
		t.Fatalf("body = %s: want the error event before the closing chunk", body)
	}
}
