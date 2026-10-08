// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"testing"

	"github.com/leelsey/goose/llm"
)

func TestLegacyFunctionsBecomeTools(t *testing.T) {
	req, _, _ := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],
		"functions":[{"name":"f","description":"d","parameters":{"type":"object"}}],
		"function_call":{"name":"f"}}`)

	if len(req.Tools) != 1 || req.Tools[0].Name != "f" || req.Tools[0].Description != "d" {
		t.Fatalf("tools = %+v", req.Tools)
	}
	if req.ToolChoice == nil || req.ToolChoice.Mode != llm.ToolSpecific || req.ToolChoice.Name != "f" {
		t.Fatalf("tool choice = %+v", req.ToolChoice)
	}
}

func TestCurrentToolSpellingWinsOverTheLegacyOne(t *testing.T) {
	req, _, diags := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"new"}}],
		"functions":[{"name":"old"}],
		"tool_choice":"none","function_call":"auto"}`)

	if len(diags.Find(llm.DropToolDefinition)) != 2 {
		t.Errorf("diagnostics = %v", diags.Codes())
	}

	if len(req.Tools) != 1 || req.Tools[0].Name != "new" {
		t.Fatalf("tools = %+v", req.Tools)
	}
	if req.ToolChoice == nil || req.ToolChoice.Mode != llm.ToolNone {
		t.Fatalf("tool choice = %+v", req.ToolChoice)
	}
}

func TestLegacyCallAndItsResultKeepAPairing(t *testing.T) {
	req, _, _ := decode(t, `{"model":"m","messages":[
		{"role":"user","content":"hi"},
		{"role":"assistant","content":null,"function_call":{"name":"f","arguments":"{\"a\":1}"}},
		{"role":"function","name":"f","content":"42"}]}`)

	if len(req.Messages) != 3 {
		t.Fatalf("messages = %d", len(req.Messages))
	}
	calls := llm.ToolUses(req.Messages[1].Content)
	if len(calls) != 1 || calls[0].Name != "f" || string(calls[0].Args) != `{"a":1}` {
		t.Fatalf("call = %+v", calls)
	}
	result, ok := req.Messages[2].Content[0].(*llm.ToolResult)
	if !ok {
		t.Fatalf("result = %T", req.Messages[2].Content[0])
	}
	if result.ID == "" || result.ID != calls[0].ID {
		t.Fatalf("call id %q, result id %q", calls[0].ID, result.ID)
	}
}

func TestEveryKnownFieldIsReadBySomething(t *testing.T) {
	req, _, _ := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"function_call":"none"}`)
	if _, carried := req.Extra["function_call"]; carried {
		t.Fatal("function_call reached Extra, so the known set and the decoder disagree")
	}
	if req.ToolChoice == nil {
		t.Fatal("function_call was neither decoded nor carried")
	}
}

// A tool or function message's name says which call it answers, and the
// result holds it. Copied onto the message as well, it reads as a participant
// name, which every other backend has no field for: strict then refuses an
// ordinary tool loop. Regression test for r4-seam-2.
func TestAToolResultsNameIsNotAParticipantName(t *testing.T) {
	req, _, _ := decode(t, `{"model":"m","messages":[
		{"role":"user","name":"alice","content":"hi"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","name":"f","content":"42"},
		{"role":"function","name":"f","content":"43"}]}`)
	if req.Messages[0].Name != "alice" {
		t.Fatalf("user name = %q: a participant's name is still read", req.Messages[0].Name)
	}
	for _, m := range req.Messages[2:] {
		if m.Name != "" {
			t.Fatalf("message = %+v: the name belongs to the result, not to a participant", m)
		}
		result, ok := m.Content[0].(*llm.ToolResult)
		if !ok || result.Name != "f" {
			t.Fatalf("result = %#v: the result keeps the name of the call it answers", m.Content[0])
		}
	}
}
