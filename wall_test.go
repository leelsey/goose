// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
)

// A backend that accepts the request and then says nothing, which is what the
// attempt deadline exists for. headers decides whether it answers before going
// quiet, which is the difference between the wall landing on the dial and the
// wall landing on the read.
func stalled(t *testing.T, headers bool) string {
	t.Helper()
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if headers {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		<-done
	}))
	t.Cleanup(func() { close(done); srv.Close() })
	return srv.URL
}

// Our own wall is not the caller leaving: a client still on the line must be
// told the backend ran out of time, not that it hung up on itself.
func TestTheAttemptWallReadsAsATimeoutOnBothPaths(t *testing.T) {
	for _, tc := range []struct {
		name    string
		stream  bool
		headers bool
	}{
		{"unary, quiet from the dial", false, false},
		{"unary, quiet after the headers", false, true},
		{"stream, quiet from the dial", true, false},
		{"stream, quiet after the headers", true, true},
	} {
		stream := tc.stream
		t.Run(tc.name, func(t *testing.T) {
			p, err := openai.New(openai.Options{ID: "stall", BaseURL: stalled(t, tc.headers), APIKey: "k", MaxRetries: 0})
			if err != nil {
				t.Fatal(err)
			}
			mem := usage.NewMemory()
			c, err := New(WithProvider(p), WithUsage(mem),
				WithLimits(llm.Limits{Timeout: 50 * time.Millisecond}))
			if err != nil {
				t.Fatal(err)
			}
			req := &llm.Request{Model: "m", Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			}}
			if stream {
				_, err = c.Stream(context.Background(), req)
			} else {
				_, err = c.Chat(context.Background(), req)
			}
			if err == nil {
				t.Fatal("a backend that never answered did not serve the call")
			}
			e := llm.AsError(err)
			if e.Kind != llm.KindTimeout {
				t.Fatalf("kind = %q, want %q: the caller is still here, the backend ran out of time",
					e.Kind, llm.KindTimeout)
			}
			if got := wire.StatusFor(e.Kind); got != http.StatusGatewayTimeout {
				t.Fatalf("status = %d, want 504: a caller still on the line is not a caller that left", got)
			}
			rows := mem.Records()
			if len(rows) == 0 {
				t.Fatal("the attempt went unbilled")
			}
			if got := rows[0].Err; !strings.Contains(got, "deadline") {
				t.Fatalf("ledger says %q, which an operator cannot tell from a caller hanging up", got)
			}
		})
	}
}

// A caller that hangs up says nothing about the backend, so counting it as a
// fault would take a healthy backend out of the chain for the cooldown.
func TestACallerLeavingIsNotABackendFault(t *testing.T) {
	for _, tc := range []struct {
		name    string
		stream  bool
		headers bool
	}{
		{"unary", false, false},
		{"unary, quiet after the headers", false, true},
		{"stream", true, false},
		{"stream, quiet after the headers", true, true},
	} {
		stream := tc.stream
		t.Run(tc.name, func(t *testing.T) {
			now := time.Now()
			_ = now
			p, err := openai.New(openai.Options{ID: "one", BaseURL: stalled(t, tc.headers), APIKey: "k", MaxRetries: 0})
			if err != nil {
				t.Fatal(err)
			}
			c, err := New(WithProvider(p),
				WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
				WithClock(func() time.Time { return now }))
			if err != nil {
				t.Fatal(err)
			}
			req := &llm.Request{Model: "m", Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			}}
			ctx, cancel := context.WithCancel(context.Background())
			time.AfterFunc(30*time.Millisecond, cancel)
			if stream {
				_, err = c.Stream(ctx, req)
			} else {
				_, err = c.Chat(ctx, req)
			}
			cancel()
			if err == nil {
				t.Fatal("a call whose caller has gone is not a served one")
			}
			if got := llm.AsError(err).Kind; got != llm.KindCancelled {
				t.Fatalf("kind = %q, want %q", got, llm.KindCancelled)
			}
			if _, ok := c.breaker.allow("one"); !ok {
				t.Fatal("the backend was taken out of the chain because a client hung up on it")
			}
		})
	}
}

// A caller's own deadline is the caller leaving too, and the backend it left
// must keep the probe slot it was holding or it is never tried again.
func TestACallersDeadlineIsNotABackendFaultEither(t *testing.T) {
	for _, tc := range []struct {
		name string
		gone func() (context.Context, context.CancelFunc)
	}{
		{"cancelled", func() (context.Context, context.CancelFunc) {
			ctx, cancel := context.WithCancel(context.Background())
			time.AfterFunc(30*time.Millisecond, cancel)
			return ctx, cancel
		}},
		{"out of time", func() (context.Context, context.CancelFunc) {
			return context.WithTimeout(context.Background(), 30*time.Millisecond)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Now()
			p, err := openai.New(openai.Options{ID: "one", BaseURL: stalled(t, false), APIKey: "k", MaxRetries: 0})
			if err != nil {
				t.Fatal(err)
			}
			c, err := New(WithProvider(p),
				WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
				WithClock(func() time.Time { return now }))
			if err != nil {
				t.Fatal(err)
			}
			// The backend is already out and its cooldown has passed, so the
			// call below is the one probe it gets.
			c.breaker.failed("one", 0)
			now = now.Add(2 * time.Minute)

			ctx, cancel := tc.gone()
			defer cancel()
			_, err = c.Chat(ctx, &llm.Request{Model: "m", Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			}})
			if err == nil {
				t.Fatal("a call whose caller has gone is not a served one")
			}
			if _, ok := c.breaker.allow("one"); !ok {
				t.Fatal("the probe slot was never given back, so this backend is out for good")
			}
			if got := c.breaker.state["one"].failures; got != 1 {
				t.Fatalf("failures = %d: a client that went away is not a fault of the backend", got)
			}
		})
	}
}

// A row that says a call cost nothing and a row that says nobody counted it
// are different rows, and an attempt that was actually sent is the second.
func TestEveryBookedAttemptSaysWhetherItWasCounted(t *testing.T) {
	var answers int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		answers++
		if answers == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p, err := openai.New(openai.Options{ID: "up", BaseURL: srv.URL, APIKey: "k", MaxRetries: 2})
	if err != nil {
		t.Fatal(err)
	}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}); err != nil {
		t.Fatal(err)
	}
	rows := mem.Records()
	if len(rows) < 2 {
		t.Fatalf("rows = %d: the discarded attempt went unbooked", len(rows))
	}
	for _, r := range rows {
		if r.Outcome != llm.OutcomeDiscarded {
			continue
		}
		if r.Time.IsZero() {
			t.Error("the discarded attempt has no time of its own")
		}
		if len(r.Usage.Anomalies) == 0 {
			t.Errorf("usage = %+v: an attempt that was sent and never counted is not a free one", r.Usage)
		}
	}
	last := rows[len(rows)-1]
	if last.Outcome != llm.OutcomeOK {
		t.Fatalf("last row = %+v", last)
	}
	if len(last.Usage.Anomalies) == 0 {
		t.Errorf("usage = %+v: this backend reported no counts at all", last.Usage)
	}
}

// queue is a gate that holds a call until a slot frees or its time runs out, as
// an agent CLI's does. No slot ever frees here.
type queue struct{ capped }

func (q *queue) Acquire(ctx context.Context) (func(), error) {
	<-ctx.Done()
	cause, kind := context.Cause(ctx), llm.KindCancelled
	if errors.Is(cause, context.DeadlineExceeded) {
		kind = llm.KindTimeout
	}
	return func() {}, &llm.Error{Kind: kind, Message: cause.Error(), Err: cause}
}

// A wait for a slot is part of the attempt and ends at its deadline. A call
// that waited and ran out of time was not shed, and its row says what it was:
// a timeout, and how long it waited. Its budget is spent, so the chain stops.
func TestAWaitAtTheGateIsHeldToTheAttemptDeadline(t *testing.T) {
	for _, stream := range []bool{false, true} {
		busy := &queue{capped{id: "busy", caps: full()}}
		next := &capped{id: "next", caps: full()}
		mem := usage.NewMemory()
		c := pair(t, busy, next, WithUsage(mem), WithLimits(llm.Limits{Timeout: 50 * time.Millisecond}))
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		start := time.Now()
		var err error
		if stream {
			_, err = c.Stream(ctx, req())
		} else {
			_, err = c.Chat(ctx, req())
		}
		waited := time.Since(start)
		cancel()
		if e := llm.AsError(err); e == nil || e.Kind != llm.KindTimeout {
			t.Fatalf("stream %v: err = %v: the attempt ran out of time at the gate", stream, err)
		}
		if waited > time.Second {
			t.Fatalf("stream %v: waited %v: the attempt's deadline did not bound the wait for a slot", stream, waited)
		}
		if next.seen != nil {
			t.Fatalf("stream %v: the budget was spent at the gate, and the next link was given another", stream)
		}
		rows := mem.Records()
		if len(rows) != 1 || rows[0].Sent || rows[0].Outcome != llm.OutcomeCancelled || rows[0].Elapsed <= 0 {
			t.Fatalf("stream %v: rows = %+v: a call that waited out its time is booked as that", stream, rows)
		}
		for _, d := range rows[0].Diags {
			if d.Code == llm.PolicySaturated && strings.Contains(d.Detail, "shed") {
				t.Fatalf("stream %v: detail = %q: nothing was shed", stream, d.Detail)
			}
		}
	}
}

// freeing is a gate whose slot frees a while after the call arrives, as an
// agent CLI's does once the call ahead of it ends, or is free at once with no
// while given, in front of a backend that then says nothing until its wall.
type freeing struct {
	capped
	after time.Duration
}

func (f *freeing) Acquire(ctx context.Context) (func(), error) {
	if f.after <= 0 {
		return func() {}, nil
	}
	select {
	case <-time.After(f.after):
		return func() {}, nil
	case <-ctx.Done():
		return func() {}, &llm.Error{Kind: llm.KindTimeout, Message: "no slot freed", Err: ctx.Err()}
	}
}

func (f *freeing) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	f.seen = a
	<-ctx.Done()
	return nil, &llm.Error{Kind: llm.KindTimeout, Message: "no answer", Sent: true, Err: ctx.Err()}
}

func (f *freeing) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	_, err := f.Complete(ctx, a)
	return nil, err
}

// The wait for a slot and the send share one attempt budget. A call that spent
// most of it waiting is sent with what is left, not given the whole budget
// again, and its row covers the wait as well as the send.
func TestAWaitAtTheGateAndTheSendShareOneBudget(t *testing.T) {
	const budget, wait = 400 * time.Millisecond, 300 * time.Millisecond
	for _, stream := range []bool{false, true} {
		p := &freeing{capped: capped{id: "one", caps: full()}, after: wait}
		mem := usage.NewMemory()
		c, err := New(WithProvider(p), WithUsage(mem), WithLimits(llm.Limits{Timeout: budget}))
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if stream {
			_, err = c.Stream(context.Background(), req())
		} else {
			_, err = c.Chat(context.Background(), req())
		}
		took := time.Since(start)
		if e := llm.AsError(err); e == nil || e.Kind != llm.KindTimeout || p.seen == nil {
			t.Fatalf("stream %v: err = %v: the backend was to be reached and run to its wall", stream, err)
		}
		if took > budget+wait/2 {
			t.Fatalf("stream %v: took %v: an attempt of %v was given its budget once for the wait and again for the send",
				stream, took, budget)
		}
		rows := mem.Records()
		if len(rows) != 1 {
			t.Fatalf("stream %v: rows = %+v", stream, rows)
		}
		if at, spent := rows[0].Time.Sub(start), time.Duration(rows[0].Elapsed)*time.Millisecond; at > wait/2 || spent < wait {
			t.Fatalf("stream %v: row starts %v in and runs %v: the wait for a slot is part of the attempt",
				stream, at, spent)
		}
	}
}

// The wait for a slot and the send share one budget, so a call queued behind
// others is sent with what the queue left of it. Its wall falling on the send
// says the queue was long, not that the backend is failing, and counting it
// would take a healthy backend out of the chain for being busy. A call given
// its slot at once had the whole budget, and its timeout still counts.
func TestATimeoutAfterAWaitForASlotIsNotTheBackendsFault(t *testing.T) {
	for _, tc := range []struct {
		what    string
		wait    time.Duration
		counted bool
	}{
		{"sent after waiting for a slot", 100 * time.Millisecond, false},
		{"sent at once", 0, true},
	} {
		for _, stream := range []bool{false, true} {
			p := &freeing{capped: capped{id: "one", caps: full()}, after: tc.wait}
			c, err := New(WithProvider(p), WithLimits(llm.Limits{Timeout: 200 * time.Millisecond}),
				WithBreaker(Breaker{Failures: 2, Cooldown: time.Minute}))
			if err != nil {
				t.Fatal(err)
			}
			for range 2 {
				p.seen = nil
				if stream {
					_, err = c.Stream(context.Background(), req())
				} else {
					_, err = c.Chat(context.Background(), req())
				}
				if e := llm.AsError(err); e == nil || e.Kind != llm.KindTimeout || p.seen == nil {
					t.Fatalf("%s, stream %v: err = %v: the backend was to be reached and run to its wall", tc.what, stream, err)
				}
			}
			if _, ok := c.breaker.allow("one"); ok == tc.counted {
				t.Fatalf("%s, stream %v: still in the chain = %v, want %v", tc.what, stream, ok, !tc.counted)
			}
		}
	}
}
