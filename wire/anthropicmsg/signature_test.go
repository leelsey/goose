// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// This wire signs only a thinking block, so a Gemini call's signature cannot
// reach the client, and the reasoning it stood for is gone from the next turn.
// That is said on the answer, streamed or not.
func TestAGeminiCallSignatureThisWireCannotCarryIsReported(t *testing.T) {
	i0 := 0
	call := &llm.ToolUse{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`), Index: &i0,
		Signature: "SIG-abc", Signer: llm.FamilyGoogle}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{call}}, Stop: llm.StopToolUse}}}
	_, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 1 {
		t.Fatalf("diagnostics = %+v", diags)
	}
	s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{call}}, {Stop: llm.StopToolUse},
	})
	diags, err = New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 1 {
		t.Fatalf("stream diagnostics = %+v", diags)
	}
}

// Another family's thinking signature reaches the client marked as theirs, and
// comes back attributed, so it is not replayed to Claude to be refused.
func TestAnotherFamilysThinkingSignatureIsMarkedForTheClient(t *testing.T) {
	thought := &llm.Thinking{Text: "plan", Signature: "gAAAA", Signer: llm.FamilyOpenAI}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{thought}}, Stop: llm.StopEnd}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"signature":"goose.v1.openai.gAAAA"`) {
		t.Fatalf("answer = %s", out.Body)
	}
	w := &wire.BufferWriter{}
	s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "m"}, []*llm.Frame{{Delta: []llm.Part{thought}}, {Stop: llm.StopEnd}})
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(w.Body.String(), `"signature":"goose.v1.openai.gAAAA"`) {
		t.Fatalf("stream = %s", w.Body.String())
	}

	body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"},
		{"role":"assistant","content":[{"type":"thinking","thinking":"plan","signature":"goose.v1.openai.gAAAA"},
		 {"type":"thinking","thinking":"mine","signature":"EqQB"}]},{"role":"user","content":"go on"}]}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1/messages", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	theirs := r.Messages[1].Content[0].(*llm.Thinking)
	ours := r.Messages[1].Content[1].(*llm.Thinking)
	if theirs.Signer != llm.FamilyOpenAI || theirs.Signature != "gAAAA" || ours.Signer != llm.FamilyAnthropic || ours.Signature != "EqQB" {
		t.Fatalf("theirs = %+v, ours = %+v", theirs, ours)
	}

	parts, _, _, err := DecodeBlocks(json.RawMessage(`[{"type":"thinking","thinking":"x","signature":"goose.v1.openai.gAAAA"}]`))
	if err != nil {
		t.Fatal(err)
	}
	if th := parts[0].(*llm.Thinking); th.Signer != llm.FamilyAnthropic || th.Signature != "goose.v1.openai.gAAAA" {
		t.Fatalf("from a backend = %+v: what it issued is its own, as it stands", th)
	}
}
