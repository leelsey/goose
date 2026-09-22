// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"encoding/json"
	"testing"
)

func TestCloneIsolatesAnAttempt(t *testing.T) {
	orig := &Request{
		Model:    "claude-sonnet-5",
		System:   []Part{&Text{Text: "be brief"}},
		Messages: []Message{{Role: RoleUser, Content: []Part{&Text{Text: "hello"}}, Cache: &CacheHint{}}},
		Tools:    []Tool{{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)}},
		Stop:     []string{"END"},
		Extra:    map[string]json.RawMessage{"logit_bias": json.RawMessage(`{"1":2}`)},
	}
	c := orig.Clone()
	c.Model = "gpt-5"
	c.Messages[0].Content[0].(*Text).Text = "goodbye"
	c.Messages[0].Cache.TTL = 3600
	c.Tools[0].Schema[0] = 'X'
	c.Stop[0] = "STOP"
	c.Extra["logit_bias"] = json.RawMessage(`{}`)
	c.System[0].(*Text).Text = "be verbose"

	if orig.Model != "claude-sonnet-5" {
		t.Fatal("model mutated through the clone")
	}
	if TextOf(orig.Messages[0].Content) != "hello" {
		t.Fatal("message content mutated through the clone")
	}
	if orig.Messages[0].Cache.TTL != 0 {
		t.Fatal("cache hint mutated through the clone")
	}
	if string(orig.Tools[0].Schema) != `{"type":"object"}` {
		t.Fatal("tool schema mutated through the clone")
	}
	if orig.Stop[0] != "END" {
		t.Fatal("stop sequence mutated through the clone")
	}
	if string(orig.Extra["logit_bias"]) != `{"1":2}` {
		t.Fatal("extra field mutated through the clone")
	}
	if TextOf(orig.System) != "be brief" {
		t.Fatal("system prompt mutated through the clone")
	}
}

func TestPartsKeepOrderAndSignatures(t *testing.T) {
	turn := []Part{
		&Thinking{Text: "first", Signature: "sig-1"},
		&ToolUse{ID: "call_1", Name: "search", Args: json.RawMessage(`{}`)},
		&Thinking{Text: "second", Signature: "sig-2"},
		&Text{Text: "done"},
	}
	c := CloneParts(turn)
	kinds := make([]PartKind, len(c))
	for i, p := range c {
		kinds[i] = p.Kind()
	}
	want := []PartKind{KindThinking, KindToolUse, KindThinking, KindText}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("order = %v, want %v", kinds, want)
		}
	}
	if c[0].(*Thinking).Signature != "sig-1" || c[2].(*Thinking).Signature != "sig-2" {
		t.Fatal("per-block signatures crossed")
	}
}

// An empty list is a setting — "none of this server's tools" — and a clone
// that turns it back into "no setting" widens what the caller asked for on
// every attempt after the first.
func TestCloneKeepsAnEmptyListDistinctFromNone(t *testing.T) {
	req := &Request{
		Model: "m",
		MCPServers: []MCPServer{{
			Name: "x", URL: "https://h/",
			AllowedTools: []string{}, ApprovalTools: []string{}, Approval: MCPApprovalNever,
		}},
	}
	got := req.Clone().MCPServers[0]
	if got.AllowedTools == nil {
		t.Error("allowed_tools became unset, so the clone offers every tool the server serves")
	}
	if got.ApprovalTools == nil || !got.NeedsApproval() {
		t.Error("the approval list became unset, so the clone waits for nobody")
	}
}
