// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package route decides which provider serves a request, and in what order.
package route

import (
	"net/http"
	"strings"

	"github.com/leelsey/goose/internal/glob"
	"github.com/leelsey/goose/llm"
)

type Call struct {
	Model    string
	Format   string
	Path     string
	Header   http.Header
	Stream   bool
	Session  string
	Provider string
}

type Matcher interface {
	Match(*Call) bool
}

type MatchFunc func(*Call) bool

func (f MatchFunc) Match(c *Call) bool { return f(c) }

func Model(pattern string) Matcher {
	return MatchFunc(func(c *Call) bool { return glob.Match(pattern, c.Model) })
}

func Format(names ...string) Matcher {
	return MatchFunc(func(c *Call) bool {
		for _, n := range names {
			if strings.EqualFold(n, c.Format) {
				return true
			}
		}
		return false
	})
}

func Header(key, value string) Matcher {
	return MatchFunc(func(c *Call) bool {
		got := c.Header.Get(key)
		if value == "" {
			return got != ""
		}
		return strings.EqualFold(got, value)
	})
}

func PathPrefix(prefix string) Matcher {
	return MatchFunc(func(c *Call) bool { return strings.HasPrefix(c.Path, prefix) })
}

func Streaming(want bool) Matcher {
	return MatchFunc(func(c *Call) bool { return c.Stream == want })
}

func All(ms ...Matcher) Matcher {
	return MatchFunc(func(c *Call) bool {
		for _, m := range ms {
			if m == nil || !m.Match(c) {
				return false
			}
		}
		return true
	})
}

func Any(ms ...Matcher) Matcher {
	return MatchFunc(func(c *Call) bool {
		for _, m := range ms {
			if m != nil && m.Match(c) {
				return true
			}
		}
		return false
	})
}

func Not(m Matcher) Matcher {
	return MatchFunc(func(c *Call) bool { return m == nil || !m.Match(c) })
}

func Always() Matcher { return MatchFunc(func(*Call) bool { return true }) }

type Target struct {
	Provider string
	Model    string
}

type Rule struct {
	Name   string
	Match  Matcher
	Model  string
	Chain  []Target
	Tools  []string
	Sticky bool
	Policy llm.LossPolicy
	Limits llm.Limits
}

type Decision struct {
	Rule    string
	Targets []Target
	Tools   []string
	Policy  llm.LossPolicy
	Limits  llm.Limits
}

type Table struct {
	Rules    []Rule
	Selector func(*Call) []Target
	Defaults Rule
}

func (t *Table) Route(c *Call) (Decision, error) {
	if t == nil {
		return Decision{}, llm.Errorf(llm.KindNoRoute, "no routes are configured")
	}
	if t.Selector != nil {
		if targets := t.Selector(c); len(targets) > 0 {
			return Decision{
				Rule: "selector", Targets: targets, Tools: t.Defaults.Tools,
				Policy: t.Defaults.Policy, Limits: t.Defaults.Limits,
			}, nil
		}
	}
	for _, r := range t.Rules {
		if r.Match != nil && !r.Match.Match(c) {
			continue
		}
		chain := make([]Target, 0, len(r.Chain))
		for _, tgt := range r.Chain {
			if tgt.Model == "" {
				tgt.Model = r.Model
			}
			if tgt.Model == "" {
				tgt.Model = c.Model
			}
			chain = append(chain, tgt)
		}
		if len(chain) == 0 {
			continue
		}
		if c.Session != "" && r.Sticky && c.Provider != "" {
			for _, tgt := range chain {
				if tgt.Provider == c.Provider {
					chain = []Target{tgt}
					break
				}
			}
		}
		tools, policy, limits := r.Tools, r.Policy, r.Limits
		if len(tools) == 0 {
			tools = t.Defaults.Tools
		}
		if !policy.Strict && len(policy.Allow) == 0 && len(policy.Deny) == 0 {
			policy = t.Defaults.Policy
		}
		if limits == (llm.Limits{}) {
			limits = t.Defaults.Limits
		}
		return Decision{
			Rule: nameOr(r.Name, "rule"), Targets: chain, Tools: tools,
			Policy: policy, Limits: limits,
		}, nil
	}
	return Decision{}, &llm.Error{
		Kind:    llm.KindNoRoute,
		Message: "no route matches model " + quote(c.Model) + " on format " + quote(c.Format),
	}
}

func nameOr(s, fallback string) string {
	if s == "" {
		return fallback
	}
	return s
}

func quote(s string) string {
	if s == "" {
		return `""`
	}
	return `"` + s + `"`
}
