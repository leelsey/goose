// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
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
			if !c.breaker.allow("one") {
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
			c.breaker.failed("one")
			now = now.Add(2 * time.Minute)

			ctx, cancel := tc.gone()
			defer cancel()
			_, err = c.Chat(ctx, &llm.Request{Model: "m", Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			}})
			if err == nil {
				t.Fatal("a call whose caller has gone is not a served one")
			}
			if !c.breaker.allow("one") {
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
