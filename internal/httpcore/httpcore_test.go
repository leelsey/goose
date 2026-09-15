// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
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
		"X-Goog-User-Project", "Cookie", "Host", "Content-Length", "Connection",
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
