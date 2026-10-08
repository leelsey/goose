// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func outputOf(t *testing.T, parts ...llm.Part) (string, llm.Diagnostics) {
	t.Helper()
	resp := &llm.Response{ID: "r", Model: "m", Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: parts}, Stop: llm.StopEnd,
	}}}
	out, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Output json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	return string(body.Output), diags
}

// Gemini ends a tool call with a part carrying only a signature. It holds
// nothing a message could, the stream leaves it out, and so does the answer.
// The signature itself has no field here, and its loss is said.
func TestASignatureOnlyPartOpensNoMessage(t *testing.T) {
	i0 := 0
	output, diags := outputOf(t,
		&llm.ToolUse{ID: "c1", Name: "f", Args: json.RawMessage(`{}`), Index: &i0, Signature: "SIG-1", Signer: llm.FamilyGoogle},
		&llm.Text{Signature: "SIG-2", Signer: llm.FamilyGoogle})
	if strings.Contains(output, `"type":"message"`) {
		t.Fatalf("output = %s: an agent reads a message after its call as the answer", output)
	}
	if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 2 {
		t.Fatalf("diagnostics = %+v", diags)
	}
}

// A reasoning item's id is how a backend that stored it finds it again, so a
// replayed item keeps the one it was issued. One goose made up names nothing a
// backend holds and is not handed on as if it did.
func TestAReplayedReasoningItemKeepsTheIDItWasIssued(t *testing.T) {
	r, _ := decode(t, `{"model":"m","input":[{"role":"user","content":"hi"},
		{"type":"reasoning","id":"rs_up","summary":[],"encrypted_content":"gAAAA"},
		{"type":"reasoning","id":"rs_stored","summary":[]},
		{"type":"reasoning","id":"rs_goose_1","summary":[{"type":"summary_text","text":"why"}],"encrypted_content":"goose.v1.anthropic.EqQB"},
		{"role":"user","content":"go on"}]}`)
	got := r.Messages[1].Content
	up, stored, minted := got[0].(*llm.Thinking), got[1].(*llm.Thinking), got[2].(*llm.Thinking)
	if up.ID != "rs_up" || up.Signature != "gAAAA" || up.Signer != llm.FamilyOpenAI {
		t.Fatalf("stateless item = %+v", up)
	}
	if stored.ID != "rs_stored" {
		t.Fatalf("stored item = %+v: its id is all it carries", stored)
	}
	if minted.ID != "" || minted.Signature != "EqQB" || minted.Signer != llm.FamilyAnthropic {
		t.Fatalf("goose's own item = %+v", minted)
	}

	output, _ := outputOf(t, &llm.Thinking{Text: "why", Signature: "EqQB", Signer: llm.FamilyAnthropic})
	if !strings.Contains(output, `"id":"rs_goose_1"`) || !strings.Contains(output, `"encrypted_content":"goose.v1.anthropic.EqQB"`) {
		t.Fatalf("output = %s: an item goose names and a signature another family minted say so", output)
	}
}

// A reasoning item with neither a summary nor a signature is one the backend
// stored, and the unstreamed answer hands it over; the stream does too.
func TestAStoredReasoningItemIsStreamed(t *testing.T) {
	var items []string
	for _, e := range events(t, &llm.Frame{Delta: []llm.Part{&llm.Thinking{ID: "rs_stored", Signer: llm.FamilyOpenAI}}},
		&llm.Frame{Stop: llm.StopEnd, TurnEnd: true}) {
		if e.name == "response.output_item.done" {
			b, _ := json.Marshal(e.body["item"])
			items = append(items, string(b))
		}
	}
	if len(items) != 1 || !strings.Contains(items[0], `"id":"rs_stored"`) || !strings.Contains(items[0], `"type":"reasoning"`) {
		t.Fatalf("items = %v", items)
	}
}
