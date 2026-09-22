// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
)

func merge(fragments ...*llm.ToolUse) []*llm.ToolUse {
	m := NewToolMerger()
	for _, f := range fragments {
		m.Merge(f, nil)
	}
	return m.Tools()
}

func TestMergerJoinsAnIndexlessContinuation(t *testing.T) {
	tools := merge(
		&llm.ToolUse{ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":`)},
		&llm.ToolUse{Args: json.RawMessage(`"go"}`)},
	)
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1: argument bytes arriving on their own continue the open call", len(tools))
	}
	if tools[0].Name != "search" || string(tools[0].Args) != `{"q":"go"}` {
		t.Fatalf("tool = %+v", tools[0])
	}
}

func TestMergerJoinsFragmentsThatRestateTheirID(t *testing.T) {
	tools := merge(
		&llm.ToolUse{ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":`)},
		&llm.ToolUse{ID: "call_a", Args: json.RawMessage(`"go"}`)},
	)
	if len(tools) != 1 {
		t.Fatalf("tools = %d, want 1: a source that repeats an id is still on the same call", len(tools))
	}
	if string(tools[0].Args) != `{"q":"go"}` {
		t.Fatalf("args = %q", tools[0].Args)
	}
}

func TestMergerKeepsSeparateCallsApart(t *testing.T) {
	tools := merge(
		&llm.ToolUse{ID: "call_a", Name: "one", Args: json.RawMessage(`{"a":1}`)},
		&llm.ToolUse{ID: "call_b", Name: "two", Args: json.RawMessage(`{"b":2}`)},
	)
	if len(tools) != 2 || string(tools[0].Args) != `{"a":1}` || string(tools[1].Args) != `{"b":2}` {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestMergerFollowsTheIndexWhereTheWireCarriesOne(t *testing.T) {
	tools := merge(
		&llm.ToolUse{Index: idx(0), ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":`)},
		&llm.ToolUse{Index: idx(1), ID: "call_b", Name: "fetch", Args: json.RawMessage(`{"url":`)},
		&llm.ToolUse{Index: idx(0), Args: json.RawMessage(`"go"}`)},
		&llm.ToolUse{Index: idx(1), Args: json.RawMessage(`"x"}`)},
	)
	if len(tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(tools))
	}
	if string(tools[0].Args) != `{"q":"go"}` || string(tools[1].Args) != `{"url":"x"}` {
		t.Fatalf("args = %q / %q", tools[0].Args, tools[1].Args)
	}
}

func TestMergerHonoursAnArgumentBudget(t *testing.T) {
	m := NewToolMerger()
	m.Merge(&llm.ToolUse{ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":`)}, func(int) bool { return true })
	m.Merge(&llm.ToolUse{Args: json.RawMessage(`"go"}`)}, func(int) bool { return false })
	tools := m.Tools()
	if len(tools) != 1 || string(tools[0].Args) != `{"q":` {
		t.Fatalf("tools = %+v: a refused budget drops the bytes, not the call", tools)
	}
}

func TestMergerResetOpensAFreshTurn(t *testing.T) {
	m := NewToolMerger()
	m.Merge(&llm.ToolUse{ID: "call_a", Name: "one", Args: json.RawMessage(`{}`)}, nil)
	m.Reset()
	if got := m.Tools(); len(got) != 0 {
		t.Fatalf("tools = %+v, want none", got)
	}
	m.Merge(&llm.ToolUse{Args: json.RawMessage(`{"b":2}`)}, nil)
	tools := m.Tools()
	if len(tools) != 1 || string(tools[0].Args) != `{"b":2}` {
		t.Fatalf("tools = %+v: the closed turn's call is not the one to continue", tools)
	}
}

func TestMergerStopsAtItsOwnBudget(t *testing.T) {
	m := NewToolMerger()
	m.Merge(&llm.ToolUse{ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":"`)}, nil)
	huge := make([]byte, DefaultToolArgsBudget+1)
	m.Merge(&llm.ToolUse{Args: huge}, nil)
	if !m.Overflowed() {
		t.Fatal("an upstream streaming fragments without end must be refused, and the refusal must be tellable")
	}
	if got := len(m.Tools()[0].Args); got != len(`{"q":"`) {
		t.Fatalf("args = %d bytes: what did not fit was not taken", got)
	}
}

func TestSharedMergersSpendOneBudget(t *testing.T) {
	first := NewToolMerger()
	second := NewToolMerger()
	second.Share(first)

	half := make([]byte, DefaultToolArgsBudget/2+1)
	first.Merge(&llm.ToolUse{ID: "a", Name: "one", Args: half}, nil)
	second.Merge(&llm.ToolUse{ID: "b", Name: "two", Args: half}, nil)

	if !second.Overflowed() || !first.Overflowed() {
		t.Fatal("one turn's worth is one turn's worth however many candidates the upstream names")
	}
}

func TestAZeroValueMergerStillTakesArguments(t *testing.T) {
	var m ToolMerger
	m.Merge(&llm.ToolUse{ID: "a", Name: "one", Args: json.RawMessage(`{"a":1}`)}, nil)
	tools := m.Tools()
	if len(tools) != 1 || string(tools[0].Args) != `{"a":1}` {
		t.Fatalf("tools = %+v: the lazy path exists to make the zero value usable", tools)
	}
}

func TestShadowedFoldsTheWayJSONMatches(t *testing.T) {
	modelled := map[string]bool{"stop": true, "num_predict": true}

	// encoding/json matches a long s to s, so this pair lands on one field.
	got := Shadowed(json.RawMessage(`{"stop":["a"],"ſtop":["b"]}`), modelled)
	if len(got) != 1 || got[0] != "stop" {
		t.Fatalf("shadowed = %v: the decoder folds this pair onto one field, so the caller must be told", got)
	}

	// A dotted capital I does not fold to i for encoding/json, so these are two
	// separate keys and neither shadows the other.
	if got := Shadowed(json.RawMessage(`{"num_predict":1,"num_predİct":2}`), modelled); len(got) != 0 {
		t.Fatalf("shadowed = %v: these are two keys, and saying otherwise drops one", got)
	}

	if got := Shadowed(json.RawMessage(`{"stop":["a"],"other":1}`), modelled); len(got) != 0 {
		t.Fatalf("shadowed = %v: one spelling each is not a clash", got)
	}
}

func TestOverflowIsReportedWhoeverKeepsTheBudget(t *testing.T) {
	var m ToolMerger
	m.Merge(&llm.ToolUse{ID: "a", Name: "one", Args: json.RawMessage(`{"a":1}`)}, func(int) bool { return false })
	if !m.Overflowed() {
		t.Fatal("a call that lost its tail is one the codec must be able to say so about")
	}
	if tools := m.Tools(); len(tools) != 1 || len(tools[0].Args) != 0 {
		t.Fatalf("tools = %+v: the refused bytes were kept anyway", tools)
	}
}

func TestOneCandidateEndingDoesNotRefillTheSharedBudget(t *testing.T) {
	first := NewToolMerger()
	second := NewToolMerger()
	second.Share(first)

	big := json.RawMessage(make([]byte, DefaultToolArgsBudget))
	first.Merge(&llm.ToolUse{ID: "a", Name: "one", Args: big}, nil)
	second.Reset()

	second.Merge(&llm.ToolUse{ID: "b", Name: "two", Args: json.RawMessage(`{"b":1}`)}, nil)
	if tools := second.Tools(); len(tools) != 1 || len(tools[0].Args) != 0 {
		t.Fatalf("tools = %+v: the turn's whole allowance was already spent", tools)
	}
}

func TestSharingNothingIsNotACrash(t *testing.T) {
	m := NewToolMerger()
	m.Share(nil)
	m.Merge(&llm.ToolUse{ID: "a", Name: "one", Args: json.RawMessage(`{"a":1}`)}, nil)
	if tools := m.Tools(); len(tools) != 1 {
		t.Fatalf("tools = %+v", tools)
	}
}

// Arguments that did not survive are not arguments that were never given, and
// a codec has to be able to say which it is looking at.
func TestBrokenArgumentsAreReportedAsBroken(t *testing.T) {
	for _, tc := range []struct {
		in     string
		broken bool
	}{
		{`{"a":1}`, false},
		{``, false},
		{`null`, false},
		{`"a string"`, false},
		{`{"a":`, true},
		{`{"a":1`, true},
	} {
		out, broken := ObjectArgsBroken(json.RawMessage(tc.in))
		if broken != tc.broken {
			t.Errorf("ObjectArgsBroken(%q) = %q, %v; want broken = %v", tc.in, out, broken, tc.broken)
		}
	}
}
