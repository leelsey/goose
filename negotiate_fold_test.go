// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// noted collects what negotiation said about an attempt.
type noted struct{ diags llm.Diagnostics }

func (n *noted) Note(d llm.Diagnostic) { n.diags = append(n.diags, d) }

func plan(req *llm.Request, caps llm.Capabilities) (*noted, error) {
	rec := &noted{}
	a := &llm.Attempt{Provider: "p", Model: "m", Request: req, Recorder: rec}
	return rec, negotiate(a, caps)
}

func (n *noted) has(code string) bool {
	for _, d := range n.diags {
		if d.Code == code {
			return true
		}
	}
	return false
}

func (n *noted) severity(code string) (llm.Severity, bool) {
	for _, d := range n.diags {
		if d.Code == code {
			return d.Severity, true
		}
	}
	return 0, false
}

func user(text string) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: text}}}
}

// A prompt is only folded if it is actually folded. Saying so otherwise fails a
// caller that refuses the loss, and by then the backend has answered and billed.
func TestTheFoldIsReportedOnlyWhereItHappens(t *testing.T) {
	req := &llm.Request{Model: "m", Messages: []llm.Message{
		user("first"),
		{Role: llm.RoleSystem, Content: []llm.Part{&llm.Text{Text: "be brief"}}},
		user("second"),
	}}
	a, err := plan(req, llm.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if !a.has(llm.DropSystemPrompt) {
		t.Fatal("a backend with no system channel cannot carry a system turn")
	}
	for _, m := range req.Messages {
		if m.Role == llm.RoleSystem || m.Role == llm.RoleDeveloper {
			t.Fatalf("messages = %+v: the turn was claimed folded and left where it was", req.Messages)
		}
	}
	if len(req.Messages) != 3 || llm.TextOf(req.Messages[1].Content) != "be brief" {
		t.Fatalf("messages = %+v: the prompt must stay where it was said", req.Messages)
	}
}

func TestNothingIsReportedWhenThereIsNoSystemTurn(t *testing.T) {
	req := &llm.Request{Model: "m", Messages: []llm.Message{user("hi")}}
	a, err := plan(req, llm.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	if a.has(llm.DropSystemPrompt) {
		t.Fatal("nothing was folded, so nothing is owed")
	}
}

// A fold that swallows a cache breakpoint bills the whole prompt fresh, and
// that is the one thing the caller marked it to avoid.
func TestAFoldKeepsTheBreakpointItAbsorbs(t *testing.T) {
	req := &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleSystem, Cache: &llm.CacheHint{}, Content: []llm.Part{&llm.Text{Text: "be brief"}}},
		user("hi"),
	}}
	a, err := plan(req, llm.Capabilities{CacheBreakpoints: 4})
	if err != nil {
		t.Fatal(err)
	}
	kept := 0
	for _, m := range req.Messages {
		if m.Cache != nil {
			kept++
		}
	}
	if kept != 1 {
		t.Fatalf("breakpoints that reached the backend = %d, diagnostics = %+v", kept, a.diags)
	}
}

// The ceiling exists so that a request too big to serve is refused before it is
// paid for, and what the caller sends is all of what it sends.
func TestTheInputCeilingWeighsEverythingTheCallerSends(t *testing.T) {
	big := json.RawMessage(`{"pad":"` + strings.Repeat("x", 1<<20) + `"}`)
	req := &llm.Request{
		Model: "m", Messages: []llm.Message{user("hi")},
		Extra: map[string]json.RawMessage{"stuff": big}, ExtraFrom: "openai",
	}
	if _, err := plan(req, llm.Capabilities{MaxInputBytes: 4096, Extensions: []string{"openai"}}); err == nil {
		t.Fatal("a megabyte of extras weighed nothing against a four-kilobyte ceiling")
	}
}

// The keys are the caller's to name, so the diagnostic built from them is the
// caller's to size, and it is written to the ledger once per link of a chain.
func TestTheDroppedFieldListIsBounded(t *testing.T) {
	extra := map[string]json.RawMessage{}
	for i := 0; i < 400; i++ {
		extra[strings.Repeat("k", 200)+string(rune('a'+i%26))+string(rune('a'+i/26))] = json.RawMessage(`1`)
	}
	req := &llm.Request{Model: "m", Messages: []llm.Message{user("hi")}, Extra: extra, ExtraFrom: "openai"}
	a, err := plan(req, llm.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range a.diags {
		if len(d.Field) > 1024 {
			t.Fatalf("the field list is %d bytes, and it is written once per attempt", len(d.Field))
		}
	}
}

// Losing every breakpoint costs more than losing one of them, so it cannot be
// the quieter of the two.
func TestLosingEveryBreakpointIsNotQuieterThanLosingOne(t *testing.T) {
	hints := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Cache: &llm.CacheHint{}, Content: []llm.Part{&llm.Text{Text: "a"}}},
			{Role: llm.RoleUser, Cache: &llm.CacheHint{}, Content: []llm.Part{&llm.Text{Text: "b"}}},
			{Role: llm.RoleUser, Cache: &llm.CacheHint{}, Content: []llm.Part{&llm.Text{Text: "c"}}},
		}}
	}
	all, err := plan(hints(), llm.Capabilities{CacheBreakpoints: 0})
	if err != nil {
		t.Fatal(err)
	}
	some, err := plan(hints(), llm.Capabilities{CacheBreakpoints: 2})
	if err != nil {
		t.Fatal(err)
	}
	lost, ok := all.severity(llm.CapCacheBreakpoint)
	if !ok {
		t.Fatalf("losing every breakpoint went unreported: %+v", all.diags)
	}
	partial, ok := some.severity(llm.DropCacheHint)
	if !ok {
		t.Fatalf("losing one breakpoint went unreported: %+v", some.diags)
	}
	if lost < partial {
		t.Fatalf("all three lost = %v but one lost = %v", lost, partial)
	}
}

// Dropping every tool is the same loss as dropping every tool server: the model
// answered without what it was offered.
func TestDroppingEveryToolIsNotQuieterThanDroppingEveryServer(t *testing.T) {
	tools, err := plan(&llm.Request{Model: "m", Messages: []llm.Message{user("hi")},
		Tools: []llm.Tool{{Name: "f"}}}, llm.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	servers, err := plan(&llm.Request{Model: "m", Messages: []llm.Message{user("hi")},
		MCPServers: []llm.MCPServer{{Name: "s", URL: "https://h/"}}}, llm.Capabilities{})
	if err != nil {
		t.Fatal(err)
	}
	got, _ := tools.severity(llm.DropToolDefinition)
	want, _ := servers.severity(llm.DropMCPServer)
	if got != want {
		t.Fatalf("tools = %v, servers = %v: the same loss, reported two ways", got, want)
	}
}

// A tool the backend was never offered is a tool the model cannot reach for, so
// the format having no field to report one is nothing to refuse over.
func TestAToolTheBackendNeverSawIsNotAReasonToRefuse(t *testing.T) {
	a := &llm.Attempt{
		Provider: "p", Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{user("hi")},
			Tools: []llm.Tool{{Name: "f"}}},
		Tools:   []llm.Tool{{Name: "f"}},
		Renders: llm.Renders{Thinking: true},
	}
	if err := negotiate(a, llm.Capabilities{}); err != nil {
		t.Fatalf("err = %v: this backend takes no tools, so none could be called", err)
	}
}
