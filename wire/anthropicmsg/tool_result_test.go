// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// nestedResults is a request whose one tool result holds another, depth deep.
func nestedResults(depth int) string {
	open := `[{"type":"tool_result","tool_use_id":"t","content":`
	return `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":` +
		strings.Repeat(open, depth) + `"x"` + strings.Repeat(`}]`, depth) + `}]}`
}

// This wire holds no tool result inside another, so one that arrives there is
// refused as the backend would refuse it, rather than read level by level at a
// cost that grows with the square of its depth. Regression test for
// r4-wire-5.
func TestAToolResultInsideAToolResultIsRefused(t *testing.T) {
	_, _, _, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Body: []byte(nestedResults(2))})
	if err == nil || llm.AsError(err).Kind != llm.KindInvalidRequest {
		t.Fatalf("err = %v, want a tool result nested in another refused as an invalid request", err)
	}

	body := nestedResults(4000)
	start := time.Now()
	_, _, _, err = New(Options{}).Decode(&wire.Inbound{Method: "POST", Body: []byte(body)})
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("a deeply nested tool result must be refused")
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("decoding %d bytes of nested tool results took %s: it must not cost the square of their depth", len(body), elapsed)
	}
}

// A tool result read straight from its block keeps its id, its error flag and
// what it holds, as it did when it was read through a second decoding.
func TestAToolResultKeepsItsFields(t *testing.T) {
	r, _, _ := decode(t, `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
		{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":[{"type":"text","text":"boom"}]},
		{"type":"tool_result","tool_use_id":"t2","is_error":"yes","content":"fine"}]}]}`)
	got := r.Messages[0].Content
	first, ok := got[0].(*llm.ToolResult)
	if !ok || first.ID != "t1" || !first.IsError || llm.TextOf(first.Content) != "boom" {
		t.Fatalf("first result = %#v", got[0])
	}
	second, ok := got[1].(*llm.ToolResult)
	if !ok || second.ID != "t2" || second.IsError || llm.TextOf(second.Content) != "fine" {
		t.Fatalf("second result = %#v: a malformed flag is read as unset, as before", got[1])
	}
}
