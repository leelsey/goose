// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

type stub struct {
	id      string
	err     error
	resp    *llm.Response
	frames  []*llm.Frame
	seenMod []string
	calls   int
	closed  bool
}

func (s *stub) ID() string                     { return s.id }
func (s *stub) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (s *stub) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	s.calls++
	s.seenMod = append(s.seenMod, a.Request.Model)
	a.Request.Model = s.id + "-rewritten"
	if s.err != nil {
		return s.resp, s.err
	}
	r := s.resp
	if r == nil {
		r = &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "ok from " + s.id}}}, Stop: llm.StopEnd}}}
	}
	return r, nil
}

func (s *stub) Stream(_ context.Context, a *llm.Attempt) (*llm.Stream, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return llm.StaticStream(llm.StreamMeta{Provider: s.id}, s.frames), nil
}

func (s *stub) Close(context.Context) error { s.closed = true; return nil }

func retryable(msg string) error {
	return &llm.Error{Kind: llm.KindOverloaded, Message: msg, Sent: true, Retryable: true}
}

func req() *llm.Request {
	return &llm.Request{Model: "claude-sonnet-5", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}
}

func TestSingleProviderNeedsNoTable(t *testing.T) {
	c, err := New(WithProvider(&stub{id: "one"}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(resp.Choices[0].Message.Content) != "ok from one" {
		t.Fatalf("response = %+v", resp)
	}
}

func TestSeveralProvidersWithoutRoutesFailsClosed(t *testing.T) {
	_, err := New(WithProvider(&stub{id: "a"}), WithProvider(&stub{id: "b"}))
	if err == nil {
		t.Fatal("routing by accident is how a request reaches a backend nobody chose")
	}
}

func TestChainFailsOverAndKeepsEveryAttempt(t *testing.T) {
	first := &stub{id: "first", err: retryable("overloaded")}
	second := &stub{id: "second"}
	mem := usage.NewMemory()
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{
			Name: "claude", Match: route.Model("claude-*"),
			Chain: []route.Target{{Provider: "first"}, {Provider: "second"}},
		}),
		WithUsage(mem),
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(resp.Choices[0].Message.Content) != "ok from second" {
		t.Fatalf("response = %+v", resp)
	}
	if len(resp.Attempts) != 2 {
		t.Fatalf("attempts = %+v: a failed link was still delivered and billed", resp.Attempts)
	}
	if resp.Attempts[0].Outcome != llm.OutcomeError || !resp.Attempts[0].Sent {
		t.Fatalf("first attempt = %+v", resp.Attempts[0])
	}
	if got, _ := mem.Stats(); got != 2 {
		t.Fatalf("booked %d attempts, want 2", got)
	}
}

func TestAttemptGetsItsOwnCopyOfTheRequest(t *testing.T) {
	first := &stub{id: "first", err: retryable("overloaded")}
	second := &stub{id: "second"}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Match: route.Always(), Chain: []route.Target{{Provider: "first"}, {Provider: "second"}}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	original := req()
	if _, err := c.Chat(context.Background(), original); err != nil {
		t.Fatal(err)
	}
	if second.seenMod[0] != "claude-sonnet-5" {
		t.Fatalf("second link saw %q: it inherited the first link's rewrite", second.seenMod[0])
	}
	if original.Model != "claude-sonnet-5" {
		t.Fatalf("caller's request was mutated to %q", original.Model)
	}
}

func TestUnretryableFailureStopsTheChain(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{Kind: llm.KindInvalidRequest, Message: "bad", Sent: true}}
	second := &stub{id: "second"}
	c, _ := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Match: route.Always(), Chain: []route.Target{{Provider: "first"}, {Provider: "second"}}}),
	)
	if _, err := c.Chat(context.Background(), req()); err == nil {
		t.Fatal("a rejected request must not be replayed against another backend")
	}
	if second.calls != 0 {
		t.Fatal("the second link was tried after a request-level rejection")
	}
}

func TestPartialResponseSurvivesTheError(t *testing.T) {
	partial := &llm.Response{
		Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "half"}}}}},
		Usage:   llm.Usage{Prompt: 40, Completion: 5, Reported: true},
	}
	p := &stub{id: "only", err: &llm.Error{Kind: llm.KindUpstream, Message: "reset", Sent: true}, resp: partial}
	c, _ := New(WithProvider(p))
	resp, err := c.Chat(context.Background(), req())
	if err == nil {
		t.Fatal("the failure must reach the caller")
	}
	if resp == nil || resp.Usage.Prompt != 40 {
		t.Fatalf("response = %+v: the generated work was still charged for", resp)
	}
}

func TestModelRoutingAndRewrite(t *testing.T) {
	a := &stub{id: "anthropic"}
	o := &stub{id: "openai"}
	c, err := New(
		WithProvider(a), WithProvider(o),
		WithRoutes(
			route.Rule{Name: "claude", Match: route.Model("claude-*"), Chain: []route.Target{{Provider: "anthropic", Model: "claude-sonnet-4-5"}}},
			route.Rule{Name: "rest", Match: route.Always(), Chain: []route.Target{{Provider: "openai"}}},
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req()); err != nil {
		t.Fatal(err)
	}
	if a.calls != 1 || o.calls != 0 {
		t.Fatalf("calls: anthropic %d, openai %d", a.calls, o.calls)
	}
	if a.seenMod[0] != "claude-sonnet-4-5" {
		t.Fatalf("resolved model = %q: the target's own name must reach the provider", a.seenMod[0])
	}
	other := req()
	other.Model = "gpt-5"
	if _, err := c.Chat(context.Background(), other); err != nil {
		t.Fatal(err)
	}
	if o.calls != 1 {
		t.Fatal("the catch-all rule did not take the second model")
	}
}

func TestNoRouteIsAClearFailure(t *testing.T) {
	c, _ := New(
		WithProvider(&stub{id: "a"}),
		WithRoutes(route.Rule{Match: route.Model("claude-*"), Chain: []route.Target{{Provider: "a"}}}),
	)
	other := req()
	other.Model = "gpt-5"
	_, err := c.Chat(context.Background(), other)
	if llm.AsError(err).Kind != llm.KindNoRoute {
		t.Fatalf("err = %v", err)
	}
}

func TestStreamBooksWhenItEnds(t *testing.T) {
	p := &stub{id: "only", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopEnd},
		{Usage: &llm.Usage{Prompt: 12, Completion: 3, Reported: true}, TurnEnd: true},
	}}
	mem := usage.NewMemory()
	c, _ := New(WithProvider(p), WithUsage(mem))
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := mem.Stats(); n != 0 {
		t.Fatal("a stream must not be booked before it has ended")
	}
	for range s.Frames() {
	}
	_ = s.Close()
	if n, _ := mem.Stats(); n != 1 {
		t.Fatalf("booked %d, want 1", n)
	}
	if total := mem.Total(); total.Prompt != 12 || total.Completion != 3 {
		t.Fatalf("total = %+v", total)
	}
	if len(s.Attempts()) != 1 {
		t.Fatalf("attempts = %+v", s.Attempts())
	}
}

func TestStreamFailsOverBeforeTheFirstFrame(t *testing.T) {
	first := &stub{id: "first", err: retryable("overloaded")}
	second := &stub{id: "second", frames: []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: "second"}}}, {Stop: llm.StopEnd}}}
	c, _ := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Match: route.Always(), Chain: []route.Target{{Provider: "first"}, {Provider: "second"}}}),
	)
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if text != "second" {
		t.Fatalf("text = %q", text)
	}
	if len(s.Attempts()) == 0 {
		t.Fatal("the failed attempt must travel with the stream that replaced it")
	}
}

func TestSessionPinsToItsIssuer(t *testing.T) {
	a := &stub{id: "a"}
	b := &stub{id: "b"}
	c, _ := New(
		WithProvider(a), WithProvider(b),
		WithRoutes(route.Rule{
			Match: route.Always(), Sticky: true,
			Chain: []route.Target{{Provider: "a"}, {Provider: "b"}},
		}),
	)
	if _, err := c.Chat(context.Background(), req(), WithSession("resp_1"), WithSessionProvider("b")); err != nil {
		t.Fatal(err)
	}
	if b.calls != 1 || a.calls != 0 {
		t.Fatalf("calls: a %d, b %d: a session id means nothing to another provider", a.calls, b.calls)
	}
}

func TestCloseReachesEveryProvider(t *testing.T) {
	a, b := &stub{id: "a"}, &stub{id: "b"}
	c, _ := New(WithProvider(a), WithProvider(b),
		WithRoutes(route.Rule{Match: route.Always(), Chain: []route.Target{{Provider: "a"}}}))
	if err := c.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !a.closed || !b.closed {
		t.Fatal("a provider that owns child processes must be closed even when it never served")
	}
}

func TestStrictPolicyRefusesALossyAnswer(t *testing.T) {
	p := &stub{id: "only", resp: &llm.Response{
		Choices:     []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "ok"}}}, Stop: llm.StopEnd}},
		Diagnostics: []llm.Diagnostic{{Code: llm.DropImage, Class: llm.ClassContent, Severity: llm.SeverityLoss}},
	}}
	c, _ := New(WithProvider(p), WithLossPolicy(llm.LossPolicy{Strict: true}))
	resp, err := c.Chat(context.Background(), req())
	if err == nil {
		t.Fatal("strict mode must refuse a translation that lost content")
	}
	if !errors.Is(err, &llm.Error{Kind: llm.KindLoss}) {
		t.Fatalf("err = %v", err)
	}
	if resp == nil {
		t.Fatal("the answer must still be available to a caller that decides to take it")
	}
}

func TestDiscardedRetriesReachTheLedger(t *testing.T) {
	p := &retrying{id: "only"}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Attempts) != 2 {
		t.Fatalf("attempts = %+v: a retry the adapter threw away was still billed", resp.Attempts)
	}
	if resp.Attempts[0].Outcome != llm.OutcomeDiscarded {
		t.Fatalf("first row = %+v", resp.Attempts[0])
	}
	if n, _ := mem.Stats(); n != 2 {
		t.Fatalf("booked %d rows, want 2", n)
	}
}

type retrying struct{ id string }

func (r *retrying) ID() string                     { return r.id }
func (r *retrying) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (r *retrying) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	a.BookExtra(llm.Attempted{Sent: true, Outcome: llm.OutcomeDiscarded, Status: 503, Err: "overloaded"})
	return &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "ok"}}}, Stop: llm.StopEnd}}}, nil
}

func (r *retrying) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return llm.StaticStream(llm.StreamMeta{}, nil), nil
}

type slow struct {
	id    string
	delay time.Duration
}

func (s *slow) ID() string                     { return s.id }
func (s *slow) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }
func (s *slow) Complete(ctx context.Context, _ *llm.Attempt) (*llm.Response, error) {
	select {
	case <-ctx.Done():
		return nil, &llm.Error{Kind: llm.KindTimeout, Provider: s.id, Message: ctx.Err().Error(), Sent: true}
	case <-time.After(s.delay):
		return &llm.Response{Choices: []llm.Choice{{Stop: llm.StopEnd}}}, nil
	}
}

func (s *slow) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	resp, err := s.Complete(ctx, a)
	if err != nil {
		return nil, err
	}
	return wirecore.Synthesise(resp), nil
}

func TestLimitsTimeoutBoundsAnAttempt(t *testing.T) {
	c, err := New(
		WithProvider(&slow{id: "slow", delay: time.Minute}),
		WithLimits(llm.Limits{Timeout: 20 * time.Millisecond}),
	)
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if _, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err == nil {
		t.Fatal("a call past the route's deadline must not be served")
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("took %s: the deadline was not applied", elapsed)
	}
}

func TestLimitsTimeoutDoesNotStrangleAStreamedBody(t *testing.T) {
	c, err := New(
		WithProvider(&stub{id: "s", frames: []*llm.Frame{
			{Delta: []llm.Part{&llm.Text{Text: "hi"}}}, {Stop: llm.StopEnd},
		}}),
		WithLimits(llm.Limits{Timeout: 30 * time.Millisecond}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	time.Sleep(80 * time.Millisecond)
	var n int
	for range s.Frames() {
		n++
	}
	if s.Err() != nil {
		t.Fatalf("err = %v: a primed stream outlives the deadline that started it", s.Err())
	}
	if n != 2 {
		t.Fatalf("frames = %d, want 2", n)
	}
}

func TestARoutesToolsJoinTheCallersRatherThanReplaceThem(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, err := New(
		WithProvider(p),
		WithTools("group", []llm.Tool{{Name: "shared"}, {Name: "route_only"}}),
		WithRoutes(route.Rule{
			Name: "r", Match: route.Always(), Tools: []string{"group"},
			Chain: []route.Target{{Provider: "one"}},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	r := req()
	r.Tools = []llm.Tool{{Name: "shared", Description: "the caller's"}, {Name: "caller_only"}}
	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}

	got := map[string]string{}
	for _, tool := range p.seen.Request.Tools {
		got[tool.Name] = tool.Description
	}
	for _, want := range []string{"shared", "caller_only", "route_only"} {
		if _, ok := got[want]; !ok {
			t.Errorf("%s did not reach the backend: %v", want, got)
		}
	}
	if got["shared"] != "the caller's" {
		t.Errorf("the route's tool shadowed the caller's: %q", got["shared"])
	}
	if !resp.Diagnostics.Has(llm.DropToolDefinition) {
		t.Errorf("the collision went unreported: %v", resp.Diagnostics.Codes())
	}
	if len(p.seen.Tools) != len(p.seen.Request.Tools) {
		t.Errorf("attempt tools %d, request tools %d", len(p.seen.Tools), len(p.seen.Request.Tools))
	}
}

func TestAnUnboundedModelNameIsRefused(t *testing.T) {
	c, err := New(WithProvider(&stub{id: "one"}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), &llm.Request{
		Model:    strings.Repeat("a.", 200000),
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err == nil {
		t.Fatal("matching a megabyte of separators against every route is work a caller must not be able to ask for")
	}
	if e := llm.AsError(err); e.Kind != llm.KindInvalidRequest {
		t.Fatalf("kind = %q", e.Kind)
	}
}
