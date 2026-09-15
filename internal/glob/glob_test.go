// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package glob

import "testing"

func TestMatch(t *testing.T) {
	cases := []struct {
		pattern string
		name    string
		want    bool
		why     string
	}{
		{"*", "anything", true, "the bare star matches everything"},
		{"", "anything", false, "an empty pattern matches nothing, including an empty name"},
		{"claude-sonnet-5", "claude-sonnet-5", true, "a literal matches itself"},
		{"claude-sonnet-5", "claude-sonnet-4", false, "a literal matches nothing else"},
		{"CLAUDE-*", "claude-sonnet-5", true, "matching is case-insensitive"},
		{"claude-?", "claude-5", true, "a question mark is one character"},
		{"claude-?", "claude-50", false, "a question mark is not two"},

		{"claude-*", "anthropic.claude-sonnet-4-20250514-v1:0", true, "a Bedrock identifier is dot-qualified"},
		{"claude-*", "us.anthropic.claude-sonnet-4-v1:0", true, "a Bedrock identifier may carry a region too"},
		{"claude-*", "publishers/anthropic/models/claude-sonnet-4", true, "a Vertex identifier is path-qualified"},
		{"gemini-*", "models/gemini-2.5-pro", true, "a Gemini identifier is prefixed"},
		{"llama3.1*", "llama3.1:8b", true, "an Ollama tag is part of the name"},
		{"8b", "llama3.1:8b", true, "a tail after any separator is a segment"},

		{"models/gemini-*", "models/gemini-2.5-pro", true, "a qualified pattern matches a qualified name"},
		{"models/gemini-*", "publishers/x/models/gemini-2.5-pro", false, "and only from the start"},
		{"gpt-4.1", "gpt-4.1", true, "a dot in a pattern is literal"},
		{"gpt-4.1", "prefix.gpt-4.1", false, "a pattern that spells a separator is not tried against a tail"},

		{"claude-*", "anthropic-claude-sonnet-4", false, "a hyphen is not a separator"},
		{"sonnet-*", "anthropic.claude-sonnet-4", false, "a tail match is anchored, not a substring search"},
	}
	for _, c := range cases {
		if got := Match(c.pattern, c.name); got != c.want {
			t.Errorf("Match(%q, %q) = %v, want %v: %s", c.pattern, c.name, got, c.want, c.why)
		}
	}
}

func TestMatchDoesNotBacktrackForever(t *testing.T) {
	pattern := "*a*a*a*a*a*a*a*a*a*a*b"
	name := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaac"
	if Match(pattern, name) {
		t.Fatal("this must not match")
	}
}
