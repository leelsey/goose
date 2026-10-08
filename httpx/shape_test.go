// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/openaichat"
)

type sessioned struct{ stub }

func (s *sessioned) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	r, err := s.stub.Complete(ctx, a)
	if r != nil {
		r.Session = "sess-1"
	}
	return r, err
}

func (s *sessioned) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "served", Session: "sess-1"}, []*llm.Frame{
		{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopEnd, TurnEnd: true},
	}), nil
}

// Clients join a repeated header into one value, and "sess-1, sess-1" is not
// an id anyone can send back.
func TestTheSessionIsSentOnce(t *testing.T) {
	srv := server(t, &sessioned{}, Options{})
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"/api/chat", `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`},
		{"/api/chat", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
	} {
		resp := post(t, srv, c.path, c.body, nil)
		_ = read(t, resp)
		if got := resp.Header.Values(SessionHeader); len(got) != 1 || got[0] != "sess-1" {
			t.Errorf("%s %s: %s = %q", c.path, c.body, SessionHeader, got)
		}
	}
}

// The allowance comes back one request at a time, so a hint shorter than that
// sends a client that honours it straight into the next refusal.
func TestTheRetryHintIsWhenTheAllowanceRefills(t *testing.T) {
	now := time.Now()
	l := newLimiter(6, func() time.Time { return now })
	for range 6 {
		if ok, _ := l.allow("a"); !ok {
			t.Fatal("the allowance is spendable")
		}
	}
	ok, wait := l.allow("a")
	if ok || wait != 10*time.Second {
		t.Fatalf("allow = %v, wait = %v: six a minute is one every ten seconds", ok, wait)
	}
	now = now.Add(4 * time.Second)
	if _, wait = l.allow("a"); wait.Round(time.Millisecond) != 6*time.Second {
		t.Fatalf("wait = %v: part of the token has come back", wait)
	}

	c, err := goose.New(goose.WithProvider(&stub{}), goose.WithClock(func() time.Time { return now }))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		MaxPerMinute: 6,
		Codecs:       []wire.Codec{openaichat.New(openaichat.Options{}), anthropicmsg.New(anthropicmsg.Options{})},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	for _, path := range []string{"/v1/chat/completions", "/v1/messages", "/v1/nothing"} {
		var last *http.Response
		for range 7 {
			last = post(t, srv, path, `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`, nil)
			_ = read(t, last)
		}
		if last.StatusCode != http.StatusTooManyRequests {
			t.Fatalf("%s: status = %d, want 429", path, last.StatusCode)
		}
		if got := last.Header.Values("Retry-After"); len(got) != 1 {
			t.Fatalf("%s: Retry-After = %q", path, got)
		}
		if secs, _ := strconv.Atoi(last.Header.Get("Retry-After")); secs != 10 {
			t.Fatalf("%s: Retry-After = %q, want 10", path, last.Header.Get("Retry-After"))
		}
		now = now.Add(time.Minute)
	}
}

// Gemini's routes are POST prefixes under /v1/models/ and /models/, which is
// neither the model list those paths name without the slash nor the model the
// OpenAI and Anthropic SDKs GET beneath it.
func TestTheModelListIsRefusedInItsOwnWire(t *testing.T) {
	m := NewMetrics()
	c, err := goose.New(goose.WithProvider(&probing{}), goose.WithUsage(m))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{APIKey: "secret", Metrics: m, Codecs: []wire.Codec{
		openaichat.New(openaichat.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
		geminiapi.New(geminiapi.Options{}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	get := func(path, key string) (int, string) {
		req, _ := http.NewRequest(http.MethodGet, srv.URL+path, nil)
		req.Header.Set("Authorization", "Bearer "+key)
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp.StatusCode, read(t, resp)
	}
	notGemini := func(path, body string) {
		t.Helper()
		var out struct {
			Error map[string]any `json:"error"`
		}
		if err := json.Unmarshal([]byte(body), &out); err != nil || out.Error["message"] == nil {
			t.Fatalf("%s: refusal = %s", path, body)
		}
		if out.Error["status"] != nil {
			t.Fatalf("%s: refusal = %s: this is Gemini's shape, and no Gemini client asks here", path, body)
		}
	}
	for _, path := range []string{"/v1/models", "/models", "/v1/models/gpt-4o", "/models/gpt-4o"} {
		status, body := get(path, "wrong")
		if status != http.StatusUnauthorized {
			t.Fatalf("%s: status = %d", path, status)
		}
		notGemini(path, body)
	}

	if status, _ := get("/v1/models", "secret"); status != http.StatusOK {
		t.Fatalf("status = %d", status)
	}
	status, body := get("/v1/models/gpt-4o", "secret")
	if status != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d", status)
	}
	notGemini("/v1/models/gpt-4o", body)

	status, counters := get("/metrics", "secret")
	if status != http.StatusOK || !strings.Contains(counters, "goose_requests_total") || strings.Contains(counters, `format="google"`) {
		t.Fatalf("metrics = %s: nothing here was a Gemini request", counters)
	}
}
