// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
)

func oneUser() *llm.Request {
	return &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}
}

func TestWithSelectorChoosesTheBackend(t *testing.T) {
	wrong := &capped{id: "wrong", caps: full()}
	right := &capped{id: "right", caps: full()}
	c, err := New(
		WithProvider(wrong), WithProvider(right),
		WithSelector(func(*route.Call) []route.Target {
			return []route.Target{{Provider: "right"}}
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), oneUser())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != "right" {
		t.Fatalf("served by %q", resp.Provider)
	}
	if wrong.seen != nil {
		t.Fatal("the selector was overruled")
	}
}

func TestWithDefaultToolsReachesARouteThatNamesNone(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, err := New(
		WithProvider(p),
		WithTools("mcp", []llm.Tool{{Name: "search", Schema: []byte(`{"type":"object"}`)}}),
		WithDefaultTools("mcp"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), oneUser()); err != nil {
		t.Fatal(err)
	}
	if len(p.seen.Tools) != 1 || p.seen.Tools[0].Name != "search" {
		t.Fatalf("tools = %+v", p.seen.Tools)
	}
}

func TestWithToolSourceIsReadPerRequest(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	tools := []llm.Tool{{Name: "first", Schema: []byte(`{"type":"object"}`)}}
	c, err := New(
		WithProvider(p),
		WithToolSource("mcp", func() []llm.Tool { return tools }),
		WithDefaultTools("mcp"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), oneUser()); err != nil {
		t.Fatal(err)
	}
	if len(p.seen.Tools) != 1 {
		t.Fatalf("tools = %+v", p.seen.Tools)
	}
	tools = append(tools, llm.Tool{Name: "second", Schema: []byte(`{"type":"object"}`)})
	if _, err := c.Chat(context.Background(), oneUser()); err != nil {
		t.Fatal(err)
	}
	if len(p.seen.Tools) != 2 {
		t.Fatalf("tools = %+v: a list captured once advertises what is no longer there", p.seen.Tools)
	}
}

func TestWithToolSourceRefusesNil(t *testing.T) {
	if _, err := New(WithProvider(&capped{id: "one", caps: full()}), WithToolSource("mcp", nil)); err == nil {
		t.Fatal("a source that is nil cannot be read per request")
	}
}

// A group nobody registered contributes nothing at call time and says nothing
// about it, so a route naming one is refused when the client is built.
func TestAnUnregisteredToolGroupIsRefused(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	source := WithToolSource("search", func() []llm.Tool { return nil })
	named := func(group string) Option {
		return WithRoutes(route.Rule{
			Name: "r", Match: route.Always(), Tools: []string{group},
			Chain: []route.Target{{Provider: "one"}},
		})
	}
	if _, err := New(WithProvider(p), source, named("Search")); err == nil {
		t.Fatal("a route naming a group that was never registered was accepted")
	}
	if _, err := New(WithProvider(p), source, WithDefaultTools("Search")); err == nil {
		t.Fatal("default tools naming a group that was never registered were accepted")
	}
	if _, err := New(WithProvider(p), source, named("search"),
		WithTools("fixed", nil), WithDefaultTools("fixed")); err != nil {
		t.Fatalf("registered groups were refused: %v", err)
	}
}

func TestWithHTTPClientReachesTheProviders(t *testing.T) {
	mine := &http.Client{Timeout: 17 * time.Second}
	p := &binding{capped: capped{id: "one", caps: full()}}
	if _, err := New(WithProvider(p), WithHTTPClient(mine)); err != nil {
		t.Fatal(err)
	}
	if p.env.HTTP != mine {
		t.Fatalf("the provider was bound to %v", p.env.HTTP)
	}
}

type binding struct {
	capped
	env llm.Env
}

func (b *binding) Bind(env llm.Env) error { b.env = env; return nil }

func TestWithCallPolicyAndLimitsOutrankTheClients(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, err := New(
		WithProvider(p),
		WithLossPolicy(llm.LossPolicy{Strict: true}),
		WithLimits(llm.Limits{Timeout: time.Hour, MaxFrameBytes: 1}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), oneUser(),
		WithCallPolicy(llm.LossPolicy{Allow: []string{llm.DropImage}}),
		WithCallLimits(llm.Limits{Timeout: time.Minute, MaxFrameBytes: 4096}),
	); err != nil {
		t.Fatal(err)
	}
	if p.seen.Policy.Strict {
		t.Fatal("the call's own policy was overruled by the client's")
	}
	if p.seen.Limits.MaxFrameBytes != 4096 || p.seen.Limits.Timeout != time.Minute {
		t.Fatalf("limits = %+v", p.seen.Limits)
	}
}

func TestACallWithoutOverridesTakesTheClients(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, err := New(
		WithProvider(p),
		WithLossPolicy(llm.LossPolicy{Strict: true}),
		WithLimits(llm.Limits{MaxFrameBytes: 99}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), oneUser()); err != nil {
		t.Fatal(err)
	}
	if !p.seen.Policy.Strict || p.seen.Limits.MaxFrameBytes != 99 {
		t.Fatalf("policy = %+v limits = %+v", p.seen.Policy, p.seen.Limits)
	}
}
