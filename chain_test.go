// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

func TestSaturationIsReportedAsAPolicyDecision(t *testing.T) {
	p := &gated{capped: capped{id: "one", caps: full()}, slots: make(chan struct{})}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err == nil {
		t.Fatal("a shed request is not a served one")
	}
	rows := mem.Records()
	if len(rows) != 1 || rows[0].Sent {
		t.Fatalf("records = %+v: nothing was sent, so nothing was billed", rows)
	}
	if !llm.Diagnostics(rows[0].Diags).Has(llm.PolicySaturated) {
		t.Fatalf("diagnostics = %+v", rows[0].Diags)
	}
}

func TestAbandonedStreamIsStillBooked(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	c.abandon = 20 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := c.Stream(ctx, &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err != nil {
		t.Fatal(err)
	}
	cancel()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rows := mem.Records(); len(rows) == 1 {
			if !rows[0].Sent || rows[0].Outcome != llm.OutcomeCancelled {
				t.Fatalf("row = %+v", rows[0])
			}
			if rows[0].Usage.Reported {
				t.Fatal("what an abandoned stream cost is not known, and a zero that says so is not a measurement")
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("records = %+v: a delivered call with an unknown cost is still a delivered call", mem.Records())
}

func TestAClosedStreamBooksTheFiguresRatherThanTheFallback(t *testing.T) {
	p := &stub{id: "one", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "one"}}},
		{Stop: llm.StopEnd},
		{Usage: &llm.Usage{Prompt: 7, Completion: 3, Reported: true}, TurnEnd: true},
	}}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	c.abandon = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	s, err := c.Stream(ctx, &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	for range s.Frames() {
	}
	cancel()
	_ = s.Close()

	time.Sleep(2 * time.Second)
	rows := mem.Records()
	if len(rows) != 1 {
		t.Fatalf("records = %d, want 1: the call is booked once", len(rows))
	}
	if rows[0].Usage.Prompt != 7 || rows[0].Usage.Completion != 3 {
		t.Fatalf("usage = %+v: the row that knows the figures is the one to keep", rows[0].Usage)
	}
}

func TestARefusedKeyFallsThroughToTheNextLink(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{
		Kind: llm.KindAuth, Provider: "first", Message: "invalid credentials", Sent: true, Status: 401,
	}}
	second := &stub{id: "second"}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if second.calls != 1 {
		t.Fatal("one backend's credentials say nothing about the next one's, so the chain had somewhere left to go")
	}
	if got := llm.TextOf(resp.Choices[0].Message.Content); got != "ok from second" {
		t.Fatalf("text = %q", got)
	}
}

func TestAMalformedRequestStopsTheChain(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{
		Kind: llm.KindInvalidRequest, Provider: "first", Message: "unreadable field",
		Sent: true, Status: 400, Retryable: true,
	}}
	second := &stub{id: "second"}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}); err == nil {
		t.Fatal("a request no backend can read is not a served one")
	}
	if second.calls != 0 {
		t.Fatal("the request itself is what is wrong, so the next link would refuse it the same way")
	}
}

func TestAStreamThatFailsAfterItsFirstFrameIsNotFailedOver(t *testing.T) {
	boom := &llm.Error{Kind: llm.KindUpstream, Provider: "first", Message: "cut off", Sent: true, Retryable: true}
	first := &stub{id: "first", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half"}}},
		{Err: boom},
	}}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if s.Err() == nil {
		t.Fatal("a stream cut off mid-answer is not a whole one")
	}
	if text != "half" {
		t.Fatalf("text = %q: the client reads the stream it was given", text)
	}
	if second.calls != 0 {
		t.Fatal("the second link was tried after the client had started reading the first")
	}
}

func TestAStreamThatFailsBeforeItsFirstFrameIsFailedOver(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "first", Message: "busy", Sent: true, Retryable: true,
	}}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if text != "whole" {
		t.Fatalf("text = %q: nothing had been written, so the next link could still serve", text)
	}
	if s.Meta().Provider != "second" {
		t.Fatalf("provider = %q", s.Meta().Provider)
	}
}

func TestADeadCallerStopsTheChainWhereverItGotTo(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "first", Message: "busy", Sent: true, Retryable: true,
	}}
	second := &stub{id: "second"}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Chat(ctx, &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}); err == nil {
		t.Fatal("a call whose caller has gone is not a served one")
	}
	if second.calls != 0 {
		t.Fatal("the caller's own deadline is blown, so every remaining link would fail the same way")
	}
}

type mute struct{ id string }

func (m *mute) ID() string                     { return m.id }
func (m *mute) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }
func (m *mute) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	return nil, retryable("nothing doing")
}

// Answers with neither a stream nor an error, which is the shape that used to
// keep the breaker's probe for good.
func (m *mute) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) { return nil, nil }

func TestAProviderThatAnswersWithNothingGivesTheProbeBack(t *testing.T) {
	now := time.Now()
	p := &mute{id: "quiet"}
	c, err := New(
		WithProvider(p),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { return now }),
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
		t.Fatal("the provider refused, so the call did not succeed")
	}
	now = now.Add(2 * time.Minute)
	if _, err := c.Stream(context.Background(), req()); err == nil {
		t.Fatal("a provider that returns no stream and no error is a failure")
	}
	now = now.Add(2 * time.Minute)
	if !c.breaker.allow("quiet") {
		t.Fatal("the probe was taken and never given back, so this backend is shut out for good")
	}
}

func TestRedactionKeepsTheUpstreamsWordsOutOfTheLedger(t *testing.T) {
	// An upstream refusal routinely quotes the request back, so this stands in
	// for a prompt arriving in a file operators treat as billing metadata.
	boom := &llm.Error{
		Kind: llm.KindInvalidRequest, Provider: "one", Status: 400, Sent: true,
		Message: "invalid request",
		Body:    `{"error":{"message":"messages[0].content: SECRET-PROMPT-BODY"}}`,
	}
	for _, redact := range []bool{false, true} {
		mem := usage.NewMemory()
		opts := []Option{WithProvider(&stub{id: "one", err: boom}), WithUsage(mem)}
		if redact {
			opts = append(opts, WithRedactedErrors())
		}
		c, err := New(opts...)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := c.Chat(context.Background(), &llm.Request{
			Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		}); err == nil {
			t.Fatal("the backend refused")
		}
		rows := mem.Records()
		if len(rows) != 1 {
			t.Fatalf("rows = %d", len(rows))
		}
		held := strings.Contains(rows[0].Detail, "SECRET-PROMPT-BODY")
		if redact && held {
			t.Fatal("the ledger kept the upstream's own words after being told not to")
		}
		if !redact && !held {
			t.Fatal("without redaction the body is what an operator is diagnosing from")
		}
	}
}

// halting is a backend that hands back a stream and a failure together, which
// the chain has to read as a failure while still saying what the stream had
// counted by then.
type halting struct{ id string }

func (h *halting) ID() string                     { return h.id }
func (h *halting) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (h *halting) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	return nil, llm.Errorf(llm.KindInternal, "not this way")
}

func (h *halting) Stream(_ context.Context, a *llm.Attempt) (*llm.Stream, error) {
	return llm.StaticStream(llm.StreamMeta{Provider: h.id}, nil),
		&llm.Error{Kind: llm.KindOverloaded, Message: "gone", Sent: true}
}

func (h *halting) Close(context.Context) error { return nil }

// hollow is a backend that answers with neither a stream nor a failure. The
// attempt was still made, and a vendor may still have billed it.
type hollow struct{ id string }

func (h *hollow) ID() string                     { return h.id }
func (h *hollow) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (h *hollow) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	return nil, llm.Errorf(llm.KindInternal, "not this way")
}

func (h *hollow) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) { return nil, nil }

func (h *hollow) Close(context.Context) error { return nil }

// Every attempt that was sent leaves a row, and a row says what it knows about
// the cost rather than implying there was none.
func TestAnAttemptThatFailedStillLeavesARowThatSaysWhatItKnows(t *testing.T) {
	for _, tc := range []struct {
		what string
		p    llm.Provider
	}{
		{"a stream handed back with a failure", &halting{id: "one"}},
		{"and neither a stream nor a failure", &hollow{id: "one"}},
	} {
		t.Run(tc.what, func(t *testing.T) {
			mem := usage.NewMemory()
			c, err := New(WithProvider(tc.p), WithUsage(mem))
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Stream(context.Background(), req()); err == nil {
				t.Fatal("the backend did not answer")
			}
			rows := mem.Records()
			if len(rows) != 1 {
				t.Fatalf("rows = %d, want the attempt that was made to leave one", len(rows))
			}
			if !hasAnomaly(rows[0].Usage, llm.AnomalyNotMeasurable) {
				t.Errorf("the row reads as having cost nothing rather than as uncounted: %+v", rows[0].Usage)
			}
		})
	}
}
