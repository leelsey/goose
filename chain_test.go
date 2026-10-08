// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
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

// A backend that answers and then fails on its first event has delivered
// nothing, so the next link can still serve. The failure is the backend's like
// any other, and what that first event counted stays on its row.
func TestAStreamWhoseFirstFrameIsAFailureIsFailedOver(t *testing.T) {
	first := &stub{id: "first", frames: []*llm.Frame{{
		Err: &llm.Error{
			Kind: llm.KindOverloaded, Provider: "first", Status: 200, Sent: true, Retryable: true,
			Message: "overloaded",
		},
		Usage: &llm.Usage{Prompt: 9, Reported: true},
	}}}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	mem := usage.NewMemory()
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithUsage(mem),
	)
	if err != nil {
		t.Fatal(err)
	}
	for round := range 2 {
		s, err := c.Stream(context.Background(), req())
		if err != nil {
			t.Fatal(err)
		}
		var text string
		for f := range s.Frames() {
			text += llm.TextOf(f.Delta)
		}
		_ = s.Close()
		if text != "whole" || s.Meta().Provider != "second" {
			t.Fatalf("round %d: text = %q from %q: nothing had been written, so the next link could still serve",
				round, text, s.Meta().Provider)
		}
	}
	if first.calls != 1 {
		t.Fatalf("first link tried %d times: its failure was booked as a success and the breaker never opened", first.calls)
	}
	var row *usage.Record
	for _, r := range mem.Records() {
		if r.Provider == "first" {
			row = &r
			break
		}
	}
	if row == nil || row.Outcome == llm.OutcomeOK || !row.Sent || row.Usage.Prompt != 9 {
		t.Fatalf("row = %+v: a failed attempt is booked as one, with what it counted", row)
	}
}

// Anthropic opens every stream with message_start, so an overload reported
// after it is the shape that matters: the opening named and counted the answer,
// but nothing of it was delivered, and the next link can still serve.
func TestAnAnthropicStreamOverloadedAfterItsOpeningIsFailedOver(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\n"+
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`+"\n\n"+
			"event: ping\n"+`data: {"type":"ping"}`+"\n\n"+
			"event: error\n"+`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n")
	}))
	defer srv.Close()
	claude, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	mem := usage.NewMemory()
	c, err := New(
		WithProvider(claude), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "claude"}, {Provider: "second"},
		}}),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithUsage(mem),
	)
	if err != nil {
		t.Fatal(err)
	}
	for round := range 2 {
		s, err := c.Stream(context.Background(), req())
		if err != nil {
			t.Fatal(err)
		}
		var text string
		for f := range s.Frames() {
			text += llm.TextOf(f.Delta)
		}
		_ = s.Close()
		if text != "whole" || s.Meta().Provider != "second" || s.Err() != nil {
			t.Fatalf("round %d: text = %q from %q, err = %v: nothing had been written, so the next link could still serve",
				round, text, s.Meta().Provider, s.Err())
		}
	}
	if n := hits.Load(); n != 1 {
		t.Fatalf("claude was tried %d times: its failure was booked as a success and the breaker never opened", n)
	}
	rows := mem.Records()
	if len(rows) == 0 || rows[0].Provider != "claude" || rows[0].Outcome == llm.OutcomeOK || rows[0].Usage.Prompt != 12 {
		t.Fatalf("rows = %+v: the failed attempt is booked as one, with what its opening counted", rows)
	}
}

// A capability one backend lacks is that backend's own: what none of them has
// was refused while planning, so the next link may well serve.
func TestACapabilityOneLinkLacksDoesNotEndTheChain(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{
		Kind: llm.KindUnsupported, Provider: "first", Message: "this deployment reads no images",
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
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatalf("err = %v: the next link had not been asked", err)
	}
	if resp.Provider != "second" {
		t.Fatalf("served by %q", resp.Provider)
	}
}

// A loss the caller would not take is refused on a stream as it is on an
// unstreamed answer. What went before it cannot be taken back, but the frame
// carrying it is not delivered, and the stream ends in a failure rather than
// reading as a whole answer.
func TestAStreamMeetingALossTheCallerRefusesEndsInAFailure(t *testing.T) {
	lost := llm.Diagnostic{
		Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
		Severity: llm.SeverityLoss, Detail: "a block this wire cannot carry",
	}
	p := &stub{id: "one", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "before"}}},
		{Delta: []llm.Part{&llm.Text{Text: "lossy"}}, Diagnostics: llm.Diagnostics{lost}},
		{Delta: []llm.Part{&llm.Text{Text: "after"}}},
		{Stop: llm.StopEnd, TurnEnd: true},
	}}
	c, err := New(WithProvider(p), WithLossPolicy(llm.LossPolicy{Strict: true}))
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if text != "before" {
		t.Fatalf("text = %q: the stream went on past a loss the caller refused", text)
	}
	if e := llm.AsError(s.Err()); e == nil || e.Kind != llm.KindLoss {
		t.Fatalf("err = %v: a refused stream that ends cleanly reads as a whole answer", s.Err())
	}
}

// noting drops a turn it has no place for while building its request, and says
// so, as an adapter does.
type noting struct{ capped }

func (n *noting) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	a.Note(llm.DroppedMessage(llm.RoleUser, n.id, "this backend has no place for the turn"))
	return n.capped.Complete(ctx, a)
}

func (n *noting) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	a.Note(llm.DroppedMessage(llm.RoleUser, n.id, "this backend has no place for the turn"))
	return n.capped.Stream(ctx, a)
}

// A stream answers to everything an unstreamed answer is checked against,
// including what the backend noted while it built the request.
func TestALossNotedWhileSendingIsRefusedOnAStreamToo(t *testing.T) {
	c, err := New(WithProvider(&noting{capped{id: "one", caps: full()}}), WithLossPolicy(llm.LossPolicy{Strict: true}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req()); llm.AsError(err) == nil || llm.AsError(err).Kind != llm.KindLoss {
		t.Fatalf("err = %v", err)
	}
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var n int
	for range s.Frames() {
		n++
	}
	if e := llm.AsError(s.Err()); n != 0 || e == nil || e.Kind != llm.KindLoss {
		t.Fatalf("frames = %d, err = %v: the stream delivered what the unstreamed call refused", n, s.Err())
	}
}

// A loss the caller refuses is known once planning has noted it, so it is
// refused before anything is sent, and the next link, which may carry the
// request whole, is tried.
func TestALossTheCallerRefusesIsRefusedBeforeItIsSent(t *testing.T) {
	for _, tc := range []struct {
		what   string
		stream bool
	}{
		{"an unstreamed call", false},
		{"and a streamed one", true},
	} {
		stream := tc.stream
		t.Run(tc.what, func(t *testing.T) {
			caps := full()
			caps.ThinkingSignatures = false
			narrow := &capped{id: "narrow", caps: caps}
			wide := &capped{id: "wide", caps: full()}
			mem := usage.NewMemory()
			c, err := New(
				WithProvider(narrow), WithProvider(wide),
				WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
					{Provider: "narrow"}, {Provider: "wide"},
				}}),
				WithLossPolicy(llm.LossPolicy{Strict: true}),
				WithUsage(mem),
			)
			if err != nil {
				t.Fatal(err)
			}
			r := &llm.Request{Model: "m", Messages: []llm.Message{
				{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "because", Signature: "sig"}}},
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
			}}
			if stream {
				s, err := c.Stream(context.Background(), r)
				if err != nil {
					t.Fatal(err)
				}
				for range s.Frames() {
				}
				_ = s.Close()
			} else if _, err := c.Chat(context.Background(), r); err != nil {
				t.Fatal(err)
			}
			if narrow.seen != nil {
				t.Fatal("the link that would have lost the signature was sent the request and paid for it")
			}
			if wide.seen == nil {
				t.Fatal("the next link, which carries the request whole, was never tried")
			}
			rows := mem.Records()
			if len(rows) == 0 || rows[0].Provider != "narrow" || rows[0].Sent || rows[0].Outcome != llm.OutcomeRejected {
				t.Fatalf("rows = %+v: the refusal is on the record and cost nothing", rows)
			}
		})
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
	if _, ok := c.breaker.allow("quiet"); !ok {
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

// empty is a backend that answers Chat with neither a response nor a failure.
type empty struct{ id string }

func (e *empty) ID() string                     { return e.id }
func (e *empty) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (e *empty) Complete(context.Context, *llm.Attempt) (*llm.Response, error) { return nil, nil }

func (e *empty) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return nil, llm.Errorf(llm.KindInternal, "not this way")
}

func (e *empty) Close(context.Context) error { return nil }

// A backend that answers with nothing has not answered, whichever call it was.
// Reading it as a success bills a row as though it worked and tells the
// breaker this backend is healthy while the caller is handed a failure.
func TestABackendThatAnswersWithNothingIsNotASuccess(t *testing.T) {
	mem := usage.NewMemory()
	c, err := New(WithProvider(&empty{id: "one"}), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req()); err == nil {
		t.Fatal("the backend did not answer")
	}
	rows := mem.Records()
	if len(rows) != 1 {
		t.Fatalf("rows = %d", len(rows))
	}
	if rows[0].Outcome == llm.OutcomeOK {
		t.Errorf("a call that returned nothing was booked as one that worked: %+v", rows[0])
	}
	if !hasAnomaly(rows[0].Usage, llm.AnomalyNotMeasurable) {
		t.Errorf("the row reads as having cost nothing rather than as uncounted: %+v", rows[0].Usage)
	}
}

// Every row for an attempt that was sent says the same thing about what it
// cost: nobody counted it, which is not the same as it having cost nothing.
func TestARowForASentAttemptSaysItWasNotCounted(t *testing.T) {
	boom := &llm.Error{Kind: llm.KindInternal, Status: 500, Sent: true, Message: "upstream fell over"}
	for _, tc := range []struct {
		what   string
		stream bool
	}{
		{"an unstreamed call", false},
		{"and a streamed one", true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			mem := usage.NewMemory()
			c, err := New(WithProvider(&stub{id: "one", err: boom}), WithUsage(mem))
			if err != nil {
				t.Fatal(err)
			}
			if tc.stream {
				_, err = c.Stream(context.Background(), req())
			} else {
				_, err = c.Chat(context.Background(), req())
			}
			if err == nil {
				t.Fatal("the backend refused")
			}
			rows := mem.Records()
			if len(rows) != 1 {
				t.Fatalf("rows = %d", len(rows))
			}
			if !hasAnomaly(rows[0].Usage, llm.AnomalyNotMeasurable) {
				t.Errorf("the row reads as having cost nothing rather than as uncounted: %+v", rows[0].Usage)
			}
		})
	}
}

// selfRefusing simulates a provider whose own pre-send check refuses once it
// has built its request, as agentcli's a.Check does under strict: unlike a
// prepare-stage refusal, this one comes back from Complete or Stream itself.
type selfRefusing struct {
	capped
}

func (s *selfRefusing) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	s.seen = a
	return nil, &llm.Error{Kind: llm.KindLoss, Code: llm.DropToolCall, Message: "a tool call did not survive translation", Sent: false}
}

func (s *selfRefusing) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	_, err := s.Complete(ctx, a)
	return nil, err
}

// partialOverload behaves like failing, but hands back a partial response
// alongside its error, as a backend that had begun answering before it
// failed might.
type partialOverload struct {
	capped
}

func (p *partialOverload) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	p.seen = a
	return &llm.Response{Usage: llm.Usage{Prompt: 7, Reported: true}}, retryable("busy")
}

// A deferred link's own provider can refuse to send what it built, after
// prepare already let it through, as agentcli's a.Check does under strict.
// That refusal must not replace the tool-carrying link's own failure, which a
// retry may cure where a loss reads as final.
func TestADeferredLinksOwnRefusalLeavesTheCarriersFailureStanding(t *testing.T) {
	for _, stream := range []bool{false, true} {
		api := &failing{capped{id: "api", caps: full()}}
		cli := &selfRefusing{capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}}
		c := pair(t, api, cli)
		var err error
		if stream {
			_, err = c.Stream(context.Background(), withTool(req()))
		} else {
			_, err = c.Chat(context.Background(), withTool(req()))
		}
		e := llm.AsError(err)
		if e == nil || e.Kind != llm.KindOverloaded || !e.Sent {
			t.Fatalf("stream %v: err = %v: the deferred link's own refusal hid why the carrier failed", stream, err)
		}
		if cli.seen == nil {
			t.Fatalf("stream %v: the deferred link was never reached", stream)
		}
	}
}

// The failure a deferred link's own refusal must not hide includes what the
// carrier had already answered: a partial response and its usage, not only
// its error.
func TestADeferredLinksOwnRefusalDoesNotDropTheCarriersPartialAnswer(t *testing.T) {
	api := &partialOverload{capped{id: "api", caps: full()}}
	cli := &selfRefusing{capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}}
	c := pair(t, api, cli)
	resp, err := c.Chat(context.Background(), withTool(req()))
	e := llm.AsError(err)
	if e == nil || e.Kind != llm.KindOverloaded || !e.Sent {
		t.Fatalf("err = %v: the deferred link's own refusal hid why the carrier failed", err)
	}
	if resp == nil || resp.Usage.Prompt != 7 {
		t.Fatalf("resp = %+v: the carrier's own partial answer was dropped", resp)
	}
	if e.Usage == nil || e.Usage.Prompt != 7 {
		t.Fatalf("err.Usage = %+v: the carrier's usage did not travel with the failure", e.Usage)
	}
}

// prechecking simulates a Gated provider whose Precheck finds a loss the
// policy would refuse once it actually built its request, as agentcli's
// flatten does under strict, or a request it would refuse outright: acquired
// records whether the gate was ever asked for a slot. Like agentcli it notes
// what it found again once it builds the request.
type prechecking struct {
	capped
	diags    llm.Diagnostics
	err      error
	acquired atomic.Bool
}

func (p *prechecking) Precheck(*llm.Attempt) (llm.Diagnostics, error) { return p.diags, p.err }

func (p *prechecking) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	for _, d := range p.diags {
		a.Note(d)
	}
	return p.capped.Complete(ctx, a)
}

func (p *prechecking) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	for _, d := range p.diags {
		a.Note(d)
	}
	return p.capped.Stream(ctx, a)
}

func (p *prechecking) Acquire(context.Context) (func(), error) {
	p.acquired.Store(true)
	return func() {}, nil
}

// A provider that can say what its own request will lose is checked against
// the policy before its gate, so a call it was always going to refuse never
// queues for a slot it could not have used, and the chain still walks on to
// a healthy link.
func TestAProvidersPrecheckRunsBeforeItsGateAndTheChainWalksOn(t *testing.T) {
	loss := llm.Diagnostics{{Code: llm.DropToolCall, Class: llm.ClassContent, Stage: llm.StageRequest, Severity: llm.SeverityLoss}}
	for _, stream := range []bool{false, true} {
		cli := &prechecking{capped: capped{id: "cli", caps: full()}, diags: loss}
		api := &capped{id: "api", caps: full()}
		c := pair(t, cli, api, WithLossPolicy(llm.LossPolicy{Strict: true}))
		var provider string
		if stream {
			s, err := c.Stream(context.Background(), req())
			if err != nil {
				t.Fatalf("stream %v: err = %v: the healthy link was never tried", stream, err)
			}
			provider = s.Meta().Provider
			_ = s.Close()
		} else {
			resp, err := c.Chat(context.Background(), req())
			if err != nil {
				t.Fatalf("stream %v: err = %v: the healthy link was never tried", stream, err)
			}
			provider = resp.Provider
		}
		if provider != "api" {
			t.Fatalf("stream %v: served by %q: the precheck's own refusal did not walk the chain on", stream, provider)
		}
		if cli.acquired.Load() {
			t.Fatalf("stream %v: a call precheck was always going to refuse still queued for a slot", stream)
		}
	}
}

// A Precheck's own refusal is a prepare-stage rejection like any other, and
// its row must say why, the same as every other rejected row does.
func TestAProvidersPrecheckRefusalIsBookedWithItsDiagnostics(t *testing.T) {
	loss := llm.Diagnostics{{Code: llm.DropToolCall, Class: llm.ClassContent, Stage: llm.StageRequest, Severity: llm.SeverityLoss}}
	cli := &prechecking{capped: capped{id: "cli", caps: full()}, diags: loss}
	api := &capped{id: "api", caps: full()}
	mem := usage.NewMemory()
	c := pair(t, cli, api, WithLossPolicy(llm.LossPolicy{Strict: true}), WithUsage(mem))
	if _, err := c.Chat(context.Background(), req()); err != nil {
		t.Fatalf("err = %v: the healthy link was never tried", err)
	}
	rows := mem.Records()
	if len(rows) != 2 || rows[0].Outcome != llm.OutcomeRejected {
		t.Fatalf("rows = %+v", rows)
	}
	if !llm.Diagnostics(rows[0].Diags).Has(llm.DropToolCall) {
		t.Fatalf("rows[0].Diags = %+v: the precheck's own refusal left no trace on its row", rows[0].Diags)
	}
}

// A KindLoss refused once the answer is in is goose's own decision: the
// request went and was billed either way, so the ledger row reads the same
// whether the caller streamed or not, and the failure names the backend that
// served it just as an unstreamed refusal already does.
func TestAPostSendRefusalIsBookedTheSameWayStreamedOrNot(t *testing.T) {
	chatWire := WithRenders(llm.Renders{ToolCalls: true, Candidates: 128})
	for _, stream := range []bool{false, true} {
		mem := usage.NewMemory()
		c, err := New(WithProvider(carrying{}), WithLossPolicy(llm.LossPolicy{Strict: true}), WithUsage(mem))
		if err != nil {
			t.Fatal(err)
		}
		if !stream {
			if _, err := c.Chat(context.Background(), req(), chatWire); llm.AsError(err) == nil {
				t.Fatal("the unstreamed call did not refuse the carried block")
			}
		} else {
			s, err := c.Stream(context.Background(), req(), chatWire)
			if err != nil {
				t.Fatal(err)
			}
			for range s.Frames() {
			}
			e := llm.AsError(s.Err())
			_ = s.Close()
			if e == nil || e.Kind != llm.KindLoss || e.Provider != "carrying" {
				t.Fatalf("err = %+v: a stream's own refusal must name the backend that served it, as the unstreamed one does", e)
			}
		}
		rows := mem.Records()
		if len(rows) != 1 || !rows[0].Sent || rows[0].Outcome != llm.OutcomeOK {
			t.Fatalf("stream %v: row = %+v: a loss goose refused after the answer was served and billed reads as one, not as the backend's own failure", stream, rows[0])
		}
	}
}
