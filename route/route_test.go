// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package route

import (
	"net/http"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

func TestModelGlobMatchesQualifiedNames(t *testing.T) {
	cases := []struct {
		pattern, model string
		want           bool
	}{
		{"claude-*", "claude-sonnet-5", true},
		{"claude-*", "publishers/anthropic/models/claude-sonnet-5", true},
		{"gpt-5", "gpt-5", true},
		{"gpt-5", "gpt-5-mini", false},
		{"llama3.1:*", "llama3.1:8b", true},
		{"*", "anything", true},
		{"claude-*", "gpt-5", false},
	}
	for _, c := range cases {
		got := Model(c.pattern).Match(&Call{Model: c.model})
		if got != c.want {
			t.Fatalf("Model(%q).Match(%q) = %v, want %v", c.pattern, c.model, got, c.want)
		}
	}
}

func TestFirstMatchingRuleWins(t *testing.T) {
	tbl := &Table{Rules: []Rule{
		{Name: "claude", Match: Model("claude-*"), Chain: []Target{{Provider: "anthropic"}}},
		{Name: "all", Match: Always(), Chain: []Target{{Provider: "openai"}}},
	}}
	dec, err := tbl.Route(&Call{Model: "claude-sonnet-5"})
	if err != nil {
		t.Fatal(err)
	}
	if dec.Rule != "claude" || dec.Targets[0].Provider != "anthropic" {
		t.Fatalf("decision = %+v", dec)
	}
	dec, err = tbl.Route(&Call{Model: "gpt-5"})
	if err != nil || dec.Rule != "all" {
		t.Fatalf("decision = %+v err = %v", dec, err)
	}
}

func TestNoMatchIsATypedFailure(t *testing.T) {
	tbl := &Table{Rules: []Rule{{Match: Model("claude-*"), Chain: []Target{{Provider: "a"}}}}}
	_, err := tbl.Route(&Call{Model: "gpt-5", Format: "openai"})
	if llm.AsError(err).Kind != llm.KindNoRoute {
		t.Fatalf("err = %v", err)
	}
}

func TestTargetsInheritTheRuleModelThenTheRequest(t *testing.T) {
	tbl := &Table{Rules: []Rule{{
		Match: Always(), Model: "rule-model",
		Chain: []Target{{Provider: "a"}, {Provider: "b", Model: "target-model"}},
	}}}
	dec, err := tbl.Route(&Call{Model: "asked"})
	if err != nil {
		t.Fatal(err)
	}
	if dec.Targets[0].Model != "rule-model" || dec.Targets[1].Model != "target-model" {
		t.Fatalf("targets = %+v", dec.Targets)
	}
}

func TestMatchersCompose(t *testing.T) {
	m := All(Format("openai"), Any(Model("gpt-*"), Model("o3*")), Not(Header("X-Skip", "")))
	if !m.Match(&Call{Format: "openai", Model: "gpt-5", Header: http.Header{}}) {
		t.Fatal("composite matcher rejected a call it describes")
	}
	if m.Match(&Call{Format: "openai", Model: "gpt-5", Header: http.Header{"X-Skip": []string{"1"}}}) {
		t.Fatal("Not did not exclude")
	}
}

func TestSelectorRunsBeforeTheRules(t *testing.T) {
	tbl := &Table{
		Selector: func(c *Call) []Target {
			if c.Header.Get("X-Provider") != "" {
				return []Target{{Provider: c.Header.Get("X-Provider")}}
			}
			return nil
		},
		Rules: []Rule{{Match: Always(), Chain: []Target{{Provider: "default"}}}},
	}
	dec, err := tbl.Route(&Call{Header: http.Header{"X-Provider": []string{"chosen"}}})
	if err != nil || dec.Targets[0].Provider != "chosen" {
		t.Fatalf("decision = %+v err = %v", dec, err)
	}
	dec, _ = tbl.Route(&Call{Header: http.Header{}})
	if dec.Targets[0].Provider != "default" {
		t.Fatalf("decision = %+v", dec)
	}
}

func TestStickySessionCollapsesTheChain(t *testing.T) {
	tbl := &Table{Rules: []Rule{{
		Match: Always(), Sticky: true,
		Chain: []Target{{Provider: "a"}, {Provider: "b"}},
	}}}
	dec, err := tbl.Route(&Call{Session: "resp_1", Provider: "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(dec.Targets) != 1 || dec.Targets[0].Provider != "b" {
		t.Fatalf("targets = %+v: a session id belongs to the provider that issued it", dec.Targets)
	}
}

func TestSelectorChainRunsUnderTheTableDefaults(t *testing.T) {
	table := &Table{
		Selector: func(*Call) []Target { return []Target{{Provider: "chosen"}} },
		Defaults: Rule{
			Tools:  []string{"mcp"},
			Policy: llm.LossPolicy{Strict: true},
			Limits: llm.Limits{Timeout: time.Minute},
		},
	}
	dec, err := table.Route(&Call{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if len(dec.Tools) != 1 || dec.Tools[0] != "mcp" {
		t.Fatalf("tools = %v", dec.Tools)
	}
	if !dec.Policy.Strict {
		t.Fatal("a selector is not a way to drop the loss policy")
	}
	if dec.Limits.Timeout != time.Minute {
		t.Fatalf("limits = %+v", dec.Limits)
	}
}

func TestRuleOverridesTheDefaultsItSets(t *testing.T) {
	table := &Table{
		Rules: []Rule{{
			Name: "own", Match: Always(), Chain: []Target{{Provider: "p"}},
			Tools: []string{"own"},
		}},
		Defaults: Rule{Tools: []string{"mcp"}, Limits: llm.Limits{Timeout: time.Minute}},
	}
	dec, err := table.Route(&Call{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if len(dec.Tools) != 1 || dec.Tools[0] != "own" {
		t.Fatalf("tools = %v: a rule that names its own does not also get the defaults", dec.Tools)
	}
	if dec.Limits.Timeout != time.Minute {
		t.Fatalf("limits = %+v: what the rule leaves unset falls back", dec.Limits)
	}
}

func TestPathAndStreamingMatchers(t *testing.T) {
	long := PathPrefix("/v1beta/models/")
	if !long.Match(&Call{Path: "/v1beta/models/gemini:streamGenerateContent"}) {
		t.Fatal("the prefix matcher rejected a path under it")
	}
	if long.Match(&Call{Path: "/v1/chat/completions"}) {
		t.Fatal("the prefix matcher took a path outside it")
	}

	for _, want := range []bool{true, false} {
		m := Streaming(want)
		if !m.Match(&Call{Stream: want}) || m.Match(&Call{Stream: !want}) {
			t.Fatalf("Streaming(%v) matched the wrong call", want)
		}
	}

	tbl := &Table{Rules: []Rule{
		{Match: All(PathPrefix("/v1beta/"), Streaming(true)), Chain: []Target{{Provider: "streamer"}}},
		{Match: Always(), Chain: []Target{{Provider: "default"}}},
	}}
	dec, err := tbl.Route(&Call{Path: "/v1beta/models/x:streamGenerateContent", Stream: true})
	if err != nil || dec.Targets[0].Provider != "streamer" {
		t.Fatalf("decision = %+v err = %v", dec, err)
	}
	dec, err = tbl.Route(&Call{Path: "/v1beta/models/x:generateContent"})
	if err != nil || dec.Targets[0].Provider != "default" {
		t.Fatalf("decision = %+v err = %v", dec, err)
	}
}
