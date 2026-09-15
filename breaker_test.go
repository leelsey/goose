// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"slices"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

func TestABreakerSkipsALinkThatKeepsFailing(t *testing.T) {
	clock := time.Now()
	dead := &stub{id: "dead", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "dead", Message: "down", Sent: true, Retryable: true,
	}}
	alive := &stub{id: "alive"}
	mem := usage.NewMemory()
	c, err := New(
		WithProvider(dead), WithProvider(alive),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "dead"}, {Provider: "alive"},
		}}),
		WithBreaker(Breaker{Failures: 3, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
		WithUsage(mem),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	for range 3 {
		if _, err := c.Chat(context.Background(), req()); err != nil {
			t.Fatal(err)
		}
	}
	if dead.calls != 3 {
		t.Fatalf("dead link tried %d times before the breaker opened", dead.calls)
	}

	for range 5 {
		resp, err := c.Chat(context.Background(), req())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Provider != "alive" {
			t.Fatalf("served by %q", resp.Provider)
		}
	}
	if dead.calls != 3 {
		t.Fatalf("dead link tried %d times: an open breaker skips it", dead.calls)
	}

	for _, r := range mem.Records() {
		if r.Provider == "dead" && r.Index >= 0 && r.Sent && r.Outcome == llm.OutcomeOK {
			t.Fatalf("a skipped link was booked as served: %+v", r)
		}
	}

	clock = clock.Add(2 * time.Minute)
	if _, err := c.Chat(context.Background(), req()); err != nil {
		t.Fatal(err)
	}
	if dead.calls != 4 {
		t.Fatalf("dead link tried %d times: the cooldown lets one through", dead.calls)
	}
}

func TestABreakerClosesOnceALinkAnswers(t *testing.T) {
	clock := time.Now()
	flaky := &flakyStub{stub: stub{id: "flaky"}, failFor: 3}
	c, err := New(
		WithProvider(flaky),
		WithBreaker(Breaker{Failures: 3, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}
	for range 3 {
		_, _ = c.Chat(context.Background(), req)
	}
	if _, err := c.Chat(context.Background(), req); err == nil {
		t.Fatal("an open breaker skips the only link there is")
	}
	clock = clock.Add(2 * time.Minute)
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatalf("the link recovered and was still refused: %v", err)
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}

type flakyStub struct {
	stub
	failFor int
}

func (f *flakyStub) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	f.calls++
	if f.calls <= f.failFor {
		return nil, &llm.Error{Kind: llm.KindOverloaded, Provider: f.id, Sent: true, Retryable: true, Message: "down"}
	}
	return &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}},
		Stop:    llm.StopEnd,
	}}}, nil
}

func TestABreakerRefusesANegativeCount(t *testing.T) {
	if _, err := New(WithProvider(&stub{id: "p"}), WithBreaker(Breaker{Failures: -1})); err == nil {
		t.Fatal("accepted a negative failure count")
	}
}

func TestAProbeRefusedBeforeItIsSentGivesItsSlotBack(t *testing.T) {
	clock := time.Now()
	b := newBreakers(Breaker{Failures: 2, Cooldown: time.Minute}, func() time.Time { return clock })

	b.failed("p")
	b.failed("p")
	if b.allow("p") {
		t.Fatal("an open breaker let a request through inside the cooldown")
	}

	clock = clock.Add(2 * time.Minute)
	if !b.allow("p") {
		t.Fatal("the cooldown passed and no probe was let through")
	}
	b.abandoned("p")

	if !b.allow("p") {
		t.Fatal("the slot was never given back, so the link is skipped for good")
	}
}

func TestASkippedLinkIsReportedToTheCaller(t *testing.T) {
	clock := time.Now()
	dead := &stub{id: "dead", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "dead", Message: "down", Sent: true, Retryable: true,
	}}
	alive := &stub{id: "alive", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "ok"}}},
		{Stop: llm.StopEnd, TurnEnd: true},
	}}
	c, err := New(
		WithProvider(dead), WithProvider(alive),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "dead"}, {Provider: "alive"},
		}}),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	if _, err := c.Chat(context.Background(), req()); err != nil {
		t.Fatal(err)
	}

	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if codes := resp.Diagnostics.Codes(); !slices.Contains(codes, llm.PolicyBreakerOpen) {
		t.Fatalf("the response says nothing about the skipped link: %v", codes)
	}

	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	for range s.Frames() {
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if codes := s.Diagnostics().Codes(); !slices.Contains(codes, llm.PolicyBreakerOpen) {
		t.Fatalf("the stream says nothing about the skipped link: %v", codes)
	}
}

func TestAWhollySkippedChainSaysWhyOnTheError(t *testing.T) {
	clock := time.Now()
	dead := &stub{id: "dead", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "dead", Message: "down", Sent: true, Retryable: true,
	}}
	c, err := New(
		WithProvider(dead),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	if _, err := c.Chat(context.Background(), req()); err == nil {
		t.Fatal("the dead link answered")
	}
	_, err = c.Chat(context.Background(), req())
	e := llm.AsError(err)
	if e == nil || !slices.Contains(llm.Diagnostics(e.Diagnostics).Codes(), llm.PolicyBreakerOpen) {
		t.Fatalf("the refusal says nothing about the skipped link: %v", err)
	}
	if _, err := c.Stream(context.Background(), req()); !slices.Contains(
		llm.Diagnostics(llm.AsError(err).Diagnostics).Codes(), llm.PolicyBreakerOpen) {
		t.Fatalf("the streamed refusal says nothing about the skipped link: %v", err)
	}
}
