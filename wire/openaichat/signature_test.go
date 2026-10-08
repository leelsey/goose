// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const signedCall = `"extra_content":{"google":{"thought_signature":"SIG-abc"}}`

// Gemini 3 refuses the next turn of a tool loop whose call has lost its
// signature. Google's own spelling of this wire carries it on the call, both
// ways and streamed or not.
func TestAGeminiCallKeepsItsSignatureOnThisWire(t *testing.T) {
	i0 := 0
	call := &llm.ToolUse{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{"city":"Paris"}`), Index: &i0,
		Signature: "SIG-abc", Signer: llm.FamilyGoogle}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{call}}, Stop: llm.StopToolUse}}}
	out, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), signedCall) || diags.Has(llm.DropSignature) {
		t.Fatalf("answer = %s, diagnostics = %+v", out.Body, diags)
	}

	w, diags, err := streamOf(t, wire.Options{Stream: true}, &llm.Frame{Delta: []llm.Part{call}}, &llm.Frame{Stop: llm.StopToolUse})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(w.Body.String(), signedCall) || diags.Has(llm.DropSignature) {
		t.Fatalf("stream = %s, diagnostics = %+v", w.Body.String(), diags)
	}

	body := `{"model":"m","messages":[{"role":"user","content":"weather?"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function",
		 "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"},` + signedCall + `}]},
		{"role":"tool","tool_call_id":"call_1","content":"20C"}]}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1/chat/completions", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	back := llm.ToolUses(r.Messages[1].Content)
	if len(back) != 1 || back[0].Signature != "SIG-abc" || back[0].Signer != llm.FamilyGoogle {
		t.Fatalf("replayed call = %+v: the signature comes back with the call", back)
	}
}

// A signature on anything but a call has no field here, and its loss is said.
func TestASignatureThisWireCannotCarryIsReported(t *testing.T) {
	parts := []llm.Part{
		&llm.Thinking{Text: "plan", Signature: "EqQB", Signer: llm.FamilyAnthropic},
		&llm.Text{Text: "ok", Signature: "CiQB", Signer: llm.FamilyGoogle},
	}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: parts}, Stop: llm.StopEnd}}}
	_, diags, err := New(Options{ReasoningField: "reasoning_content"}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 2 {
		t.Fatalf("diagnostics = %+v", diags)
	}
	_, diags, err = streamOf(t, wire.Options{Stream: true}, &llm.Frame{Delta: parts}, &llm.Frame{Stop: llm.StopEnd})
	if err != nil {
		t.Fatal(err)
	}
	if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 2 {
		t.Fatalf("stream diagnostics = %+v", diags)
	}
}
