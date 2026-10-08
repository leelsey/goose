// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

func TestDiscardedRetriesAreBooked(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("overloaded"))
			return
		}
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	var booked []llm.Attempted
	a := &llm.Attempt{
		Provider: "stub", Model: "m", Request: &llm.Request{Model: "asked"},
		Book: func(rec llm.Attempted) { booked = append(booked, rec) },
	}
	c := &Caller{Provider: "stub", HTTP: srv.Client(), MaxRetries: 3, Backoff: time.Millisecond}
	resp, err := c.Do(context.Background(), Request{URL: srv.URL, OnDiscard: BookDiscards(a)})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	if calls != 3 {
		t.Fatalf("calls = %d", calls)
	}
	if len(booked) != 2 {
		t.Fatalf("booked %d rows, want one per delivered attempt that was thrown away", len(booked))
	}
	for _, rec := range booked {
		if !rec.Sent || rec.Outcome != llm.OutcomeDiscarded || rec.Status != 503 {
			t.Fatalf("row = %+v", rec)
		}
		if rec.Provider != "stub" || rec.Requested != "asked" {
			t.Fatalf("row = %+v: a row nothing can attribute is no use", rec)
		}
	}
}

func TestUnsentFailuresAreNotBooked(t *testing.T) {
	var booked int
	a := &llm.Attempt{Provider: "stub", Book: func(llm.Attempted) { booked++ }}
	c := &Caller{Provider: "stub", MaxRetries: 2, Backoff: time.Millisecond}
	_, err := c.Do(context.Background(), Request{
		URL: "http://127.0.0.1:1", OnDiscard: BookDiscards(a),
		Sign: func(context.Context, *http.Request, []byte) error { return context.Canceled },
	})
	if err == nil {
		t.Fatal("signing failed, so nothing should have been sent")
	}
	if booked != 0 {
		t.Fatal("a request that never left the process was booked as billed")
	}
}

func TestADeadlineThatEndsWhileSigningIsATimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	c := &Caller{Provider: "stub"}
	_, err := c.Do(ctx, Request{
		URL: "http://127.0.0.1:1",
		Sign: func(ctx context.Context, _ *http.Request, _ []byte) error {
			<-ctx.Done()
			return ctx.Err()
		},
	})
	e := llm.AsError(err)
	if e == nil || e.Kind != llm.KindTimeout {
		t.Fatalf("err = %+v: a credential still on its way when time ran out is not a refused key", e)
	}
	if e.Sent {
		t.Fatal("nothing left the process, so nothing was billed")
	}
}

func TestRequestLimitRefusesBeforeAnythingIsSent(t *testing.T) {
	var reached bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = true
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	c := &Caller{Provider: "p", HTTP: srv.Client()}
	_, err := c.Do(context.Background(), Request{
		URL: srv.URL, Body: make([]byte, 64), Limits: llm.Limits{MaxRequestBytes: 16},
	})
	e := llm.AsError(err)
	if e == nil || e.Code != llm.PolicyBodyTooLarge {
		t.Fatalf("err = %v, want %s", err, llm.PolicyBodyTooLarge)
	}
	if e.Sent {
		t.Fatal("nothing left the process, so nothing was billed")
	}
	if reached {
		t.Fatal("the request must not reach the backend")
	}
}

func TestResponseLimitStopsAnUpstreamThatKeepsSending(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, 1<<16))
	}))
	defer srv.Close()

	c := &Caller{Provider: "p", HTTP: srv.Client()}
	resp, err := c.Do(context.Background(), Request{URL: srv.URL, Limits: llm.Limits{MaxResponseBytes: 128}})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, err = io.ReadAll(resp.Body)
	if e := llm.AsError(err); e == nil || e.Code != llm.PolicyBodyTooLarge {
		t.Fatalf("err = %v, want %s", err, llm.PolicyBodyTooLarge)
	}
}

func TestResponseLimitPassesABodyThatExactlyFitsIt(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(make([]byte, 128))
	}))
	defer srv.Close()

	c := &Caller{Provider: "p", HTTP: srv.Client()}
	resp, err := c.Do(context.Background(), Request{URL: srv.URL, Limits: llm.Limits{MaxResponseBytes: 128}})
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read = %v: a body that fills the allowance is not over it", err)
	}
	if len(b) != 128 {
		t.Fatalf("read %d bytes", len(b))
	}
}

func TestBackoffIsDithered(t *testing.T) {
	seen := map[time.Duration]bool{}
	for range 64 {
		seen[wait(time.Second, 2, 0)] = true
	}
	if len(seen) < 8 {
		t.Fatalf("%d distinct waits: a convoy retries in lockstep", len(seen))
	}
	for d := range seen {
		if d > 4*time.Second {
			t.Fatalf("wait = %s: the jitter must stay inside the window", d)
		}
	}
}

func TestRetryAfterIsHonouredAndBounded(t *testing.T) {
	if d := wait(time.Second, 0, 2*time.Second); d < 2*time.Second || d > 2*time.Second+200*time.Millisecond {
		t.Fatalf("wait = %s: the upstream named two seconds", d)
	}
	if d := wait(time.Second, 0, time.Hour); d > maxBackoff+maxBackoff/10 {
		t.Fatalf("wait = %s: an hour is not a wait a proxy takes", d)
	}
}

func TestClientKeepsConnectionsToOneUpstream(t *testing.T) {
	tr, ok := NewClient(time.Minute).Transport.(*http.Transport)
	if !ok {
		t.Fatal("transport is not an *http.Transport")
	}
	if tr.MaxIdleConnsPerHost < IdleConnsPerHost {
		t.Fatalf("MaxIdleConnsPerHost = %d: a proxy talks to few hosts and many callers", tr.MaxIdleConnsPerHost)
	}
}

func TestAPassthroughCarriesNoCredentialItWasGiven(t *testing.T) {
	src := http.Header{
		"Authorization":       {"Bearer client"},
		"X-Api-Key":           {"anthropic-key"},
		"X-Goog-Api-Key":      {"google-key"},
		"Api-Key":             {"azure-key"},
		"X-Goog-User-Project": {"someone"},
		"Openai-Organization": {"org-caller"},
		"Openai-Project":      {"proj-caller"},
		"Cookie":              {"session=1"},
		"Host":                {"proxy.example"},
		"Content-Length":      {"12"},
		"Content-Type":        {"application/json"},
		"Anthropic-Version":   {"2023-06-01"},
		"Connection":          {"keep-alive"},
	}
	dst := http.Header{}
	ForwardPassthrough(dst, src)

	for _, gone := range []string{
		"Authorization", "X-Api-Key", "X-Goog-Api-Key", "Api-Key",
		"X-Goog-User-Project", "OpenAI-Organization", "OpenAI-Project",
		"Cookie", "Host", "Content-Length", "Connection",
	} {
		if dst.Get(gone) != "" {
			t.Errorf("%s reached the backend", gone)
		}
	}
	for _, kept := range []string{"Content-Type", "Anthropic-Version"} {
		if dst.Get(kept) == "" {
			t.Errorf("%s belongs to the message and was dropped", kept)
		}
	}
}

// The gate takes a key in the query, and a backend handed one there keeps it in
// its logs with the rest of the address.
func TestAPassthroughQueryCarriesNoCredential(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"", ""},
		{"alt=sse&n=1", "alt=sse&n=1"},
		{"key=GATE&alt=sse", "alt=sse"},
		{"KEY=GATE", ""},
		{"api_key=a&api-key=b&access_token=c&limit=2", "limit=2"},
		{"n=1&key=a&key=b", "n=1"},
	} {
		got, err := PassQuery("p", tc.in)
		if err != nil {
			t.Fatalf("PassQuery(%q) = %v", tc.in, err)
		}
		if got != tc.want {
			t.Errorf("PassQuery(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if _, err := PassQuery("p", "key=GATE;alt=sse"); llm.AsError(err).Kind != llm.KindInvalidRequest {
		t.Fatalf("err = %v: a query nobody can read may carry a key nobody can find", err)
	}
}

func TestAConfiguredHeaderIsTheOperators(t *testing.T) {
	dst := http.Header{"X-Gateway-Auth": {"from-the-caller"}, "X-Trace": {"t"}}
	Overlay(dst, http.Header{"X-Gateway-Auth": {"gw-secret"}, "X-Extra": {"a", "b"}})
	if got := dst.Values("X-Gateway-Auth"); len(got) != 1 || got[0] != "gw-secret" {
		t.Fatalf("X-Gateway-Auth = %q: the operator set it for this backend", got)
	}
	if len(dst.Values("X-Extra")) != 2 || dst.Get("X-Trace") != "t" {
		t.Fatalf("header = %+v", dst)
	}
}

func TestARefusalIsReadForWhatItsStatusDoesNotSay(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"key not valid"}}`))
	}))
	defer srv.Close()

	c := &Caller{Provider: "p", HTTP: srv.Client(), Refine: func(e *llm.Error) {
		if strings.Contains(e.Body, "key not valid") {
			e.Kind = llm.KindAuth
		}
	}}
	_, err := c.Do(context.Background(), Request{URL: srv.URL})
	if e := llm.AsError(err); e.Kind != llm.KindAuth || e.Status != http.StatusBadRequest {
		t.Fatalf("err = %+v", e)
	}
}

func TestAnErrorKeepsNoMoreOfAnInBandBodyThanOfARefusal(t *testing.T) {
	if got := ErrorBody(make([]byte, errBodyCap*2)); len(got) != errBodyCap {
		t.Fatalf("kept %d bytes, want %d", len(got), errBodyCap)
	}
}
