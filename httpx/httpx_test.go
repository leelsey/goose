// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openaichat"
)

type stub struct {
	frames []*llm.Frame
	resp   *llm.Response
	err    error
	tools  bool
	seen   *llm.Attempt
}

func (s *stub) ID() string { return "stub" }
func (s *stub) Capabilities() llm.Capabilities {
	return llm.Capabilities{Streaming: true, Tools: s.tools}
}

func (s *stub) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	s.seen = a
	if s.err != nil {
		return nil, s.err
	}
	r := s.resp
	if r == nil {
		r = &llm.Response{
			ID: "x", Model: "served", Provider: "stub",
			Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hello"}}}, Stop: llm.StopEnd}},
			Usage:   llm.Usage{Prompt: 11, Completion: 2, Reported: true},
		}
	}
	return r, nil
}

func (s *stub) Stream(_ context.Context, a *llm.Attempt) (*llm.Stream, error) {
	s.seen = a
	if s.err != nil {
		return nil, s.err
	}
	frames := s.frames
	if frames == nil {
		frames = []*llm.Frame{
			{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "hi"}}},
			{Stop: llm.StopEnd},
			{Usage: &llm.Usage{Prompt: 11, Completion: 2, Reported: true}, TurnEnd: true},
		}
	}
	return llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "served"}, frames), nil
}

func server(t *testing.T, p llm.Provider, o Options) *httptest.Server {
	t.Helper()
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	if o.Codecs == nil {
		o.Codecs = []wire.Codec{
			openaichat.New(openaichat.Options{}),
			anthropicmsg.New(anthropicmsg.Options{}),
			ollamaapi.NewChat(ollamaapi.Options{}),
		}
	}
	s, err := New(c, o)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, srv *httptest.Server, path, body string, header http.Header) *http.Response {
	t.Helper()
	req, err := http.NewRequest("POST", srv.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, vs := range header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func TestEveryFormatReachesTheSameProvider(t *testing.T) {
	p := &stub{}
	srv := server(t, p, Options{})
	cases := []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`},
		{"/api/chat", `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`},
	}
	for _, c := range cases {
		resp := post(t, srv, c.path, c.body, nil)
		if resp.StatusCode != 200 {
			t.Fatalf("%s: status %d", c.path, resp.StatusCode)
		}
		if resp.Header.Get(UsageHeader) == "" {
			t.Fatalf("%s: usage header missing", c.path)
		}
		var u llm.Usage
		if err := json.Unmarshal([]byte(resp.Header.Get(UsageHeader)), &u); err != nil {
			t.Fatal(err)
		}
		if u.Prompt != 11 || !u.Reported {
			t.Fatalf("%s: usage = %+v", c.path, u)
		}
		if resp.Header.Get(ProviderHeader) != "stub" {
			t.Fatalf("%s: provider header = %q", c.path, resp.Header.Get(ProviderHeader))
		}
		_ = resp.Body.Close()
	}
}

func TestStreamingServesSSEAndNDJSON(t *testing.T) {
	srv := server(t, &stub{}, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type = %q", ct)
	}
	body := read(t, resp)
	if !strings.Contains(body, "data: ") || !strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Fatalf("body = %s", body)
	}

	resp = post(t, srv, "/api/chat", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	if ct := resp.Header.Get("Content-Type"); ct != "application/x-ndjson" {
		t.Fatalf("ollama streams by default: content type = %q", ct)
	}
	body = read(t, resp)
	if !strings.Contains(body, `"done":true`) {
		t.Fatalf("body = %s", body)
	}
}

func TestUpstreamFailureBeforeAnyByteIsAStatus(t *testing.T) {
	p := &stub{err: &llm.Error{Kind: llm.KindRateLimit, Status: 429, Message: "slow down", Sent: true}}
	srv := server(t, p, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 429 {
		t.Fatalf("status = %d: nothing was written yet, so the real status still fits", resp.StatusCode)
	}
	if !strings.Contains(read(t, resp), "slow down") {
		t.Fatal("the upstream explanation was thrown away")
	}
}

func TestAuthGatesEverythingButHealth(t *testing.T) {
	srv := server(t, &stub{}, Options{APIKey: "secret"})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	if resp.StatusCode != 401 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	_ = resp.Body.Close()

	h := http.Header{"Authorization": []string{"Bearer secret"}}
	resp = post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, h)
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	_ = resp.Body.Close()

	hr, err := srv.Client().Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	if hr.StatusCode != 200 {
		t.Fatalf("health status = %d", hr.StatusCode)
	}
	_ = hr.Body.Close()
}

func TestProxyKeyIsNotForwardedAsAnUpstreamCredential(t *testing.T) {
	p := &stub{}
	srv := server(t, p, Options{APIKey: "secret"})
	h := http.Header{"Authorization": []string{"Bearer secret"}}
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, h)
	_ = resp.Body.Close()
	if p.seen.Credential.Key != "" {
		t.Fatalf("credential = %q: the proxy's own key is not a backend key", p.seen.Credential.Key)
	}
}

func TestWildcardCORSNeedsAKey(t *testing.T) {
	c, err := goose.New(goose.WithProvider(&stub{}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = New(c, Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}, CORS: true, CORSOrigin: "*"})
	if err == nil {
		t.Fatal("an open proxy with a wildcard origin lets any page spend the operator's credit")
	}
}

func TestAnUnsetCORSOriginIsAWildcardAndNeedsAKey(t *testing.T) {
	c, err := goose.New(goose.WithProvider(&stub{}))
	if err != nil {
		t.Fatal(err)
	}
	_, err = New(c, Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}, CORS: true})
	if err == nil {
		t.Fatal("an open proxy served a wildcard origin because no origin was named")
	}
}

func TestBodyLimitIsEnforced(t *testing.T) {
	srv := server(t, &stub{}, Options{MaxBody: 64})
	resp := post(t, srv, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"`+strings.Repeat("x", 500)+`"}]}`, nil)
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	_ = resp.Body.Close()
}

func TestDiagnosticsReachTheClientAsAHeader(t *testing.T) {
	p := &stub{resp: &llm.Response{
		Provider: "stub",
		Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{
			&llm.Thinking{Text: "reasoned"}, &llm.Text{Text: "answer"},
		}}, Stop: llm.StopEnd}},
	}}
	srv := server(t, p, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	if !strings.Contains(resp.Header.Get(DiagnosticsHeader), llm.DropThinking) {
		t.Fatalf("diagnostics = %q: a drop must be visible to the client that paid for it", resp.Header.Get(DiagnosticsHeader))
	}
}

func read(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

type probing struct {
	stub
	err    error
	probes int
}

func (p *probing) Probe(context.Context) error {
	p.probes++
	return p.err
}

func (p *probing) Models(context.Context) ([]llm.Model, error) {
	return []llm.Model{{ID: "served-model", Owner: "stub"}}, nil
}

func TestModelsListsWhatTheProvidersServe(t *testing.T) {
	srv := server(t, &probing{}, Options{})
	resp, err := srv.Client().Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatal(err)
	}
	if len(body.Data) != 1 || body.Data[0].ID != "served-model" {
		t.Fatalf("models = %+v: an empty list is a lie when a provider can answer", body.Data)
	}
}

func TestReadyReflectsTheUpstreams(t *testing.T) {
	srv := server(t, &probing{}, Options{})
	resp, err := srv.Client().Get(srv.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	_ = resp.Body.Close()

	broken := server(t, &probing{err: errors.New("credential refused")}, Options{})
	resp, err = broken.Client().Get(broken.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d: a proxy that cannot reach its upstream is not ready", resp.StatusCode)
	}
	if !strings.Contains(read(t, resp), "credential refused") {
		t.Fatal("the reason must reach whoever asked")
	}
}

func TestReadyIsNotGatedByTheProxyKey(t *testing.T) {
	srv := server(t, &probing{}, Options{APIKey: "secret"})
	resp, err := srv.Client().Get(srv.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		t.Fatal("an orchestrator checking readiness does not hold the proxy key")
	}
}

func TestVerboseLoggingRecordsCostAndLoss(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
	p := &stub{resp: &llm.Response{
		Provider: "stub",
		Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{
			&llm.Thinking{Text: "reasoned"}, &llm.Text{Text: "answer"},
		}}, Stop: llm.StopEnd}},
		Usage: llm.Usage{Prompt: 11, Completion: 2, Reported: true},
	}}
	srv := server(t, p, Options{Logger: logger, LogRequests: true})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	_ = resp.Body.Close()

	line := buf.String()
	if !strings.Contains(line, "status=200") || !strings.Contains(line, "provider=stub") {
		t.Fatalf("log = %q", line)
	}
	if !strings.Contains(line, "prompt") {
		t.Fatalf("log = %q: what the answer cost is half of what an operator is watching for", line)
	}
	if !strings.Contains(line, llm.DropThinking) {
		t.Fatalf("log = %q: what the translation lost is the other half", line)
	}
}

func TestLoggingDoesNotStopStreaming(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	srv := server(t, &stub{}, Options{Logger: logger, LogRequests: true})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	body := read(t, resp)
	if !strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Fatalf("body = %q: the wrapper swallowed the flusher", body)
	}
}

func TestAFailedCallIsNotServedAsSuccess(t *testing.T) {
	p := &stub{err: &llm.Error{Kind: llm.KindUpstream, Message: "connection reset", Sent: true}}
	srv := server(t, p, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode == 200 {
		t.Fatal("a truncated answer was delivered as a complete one")
	}
	if !strings.Contains(read(t, resp), "connection reset") {
		t.Fatal("the reason did not reach the client")
	}
}

func TestFailedCallStillReportsWhatItCost(t *testing.T) {
	p := &stub{err: &llm.Error{
		Kind: llm.KindUpstream, Message: "reset", Sent: true,
		Usage: &llm.Usage{Prompt: 40, Completion: 5, Reported: true},
	}}
	srv := server(t, p, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	header := resp.Header.Get(UsageHeader)
	if header == "" {
		t.Fatal("no error body in any served format has a field for the cost, so the header must carry it")
	}
	var u llm.Usage
	if err := json.Unmarshal([]byte(header), &u); err != nil {
		t.Fatal(err)
	}
	if u.Prompt != 40 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestFormatThatCarriesOneCandidateRefusesMore(t *testing.T) {
	srv := server(t, &stub{}, Options{})
	resp := post(t, srv, "/v1/messages",
		`{"model":"m","max_tokens":10,"n":2,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	resp2 := post(t, srv, "/v1/chat/completions",
		`{"model":"m","n":2,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp2.Body.Close()
	if resp2.StatusCode == 200 {
		t.Fatal("a single-candidate backend answered a request for two")
	}
}

func TestProxyKeyIsNotForwardedOnPassthrough(t *testing.T) {
	p := &passing{}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:      []wire.Codec{openaichat.New(openaichat.Options{})},
		APIKey:      "secret",
		Passthrough: "stub",
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	req, _ := http.NewRequest("GET", srv.URL+"/v1/anything", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if p.key == "secret" {
		t.Fatal("the proxy's own key was handed to the backend")
	}
}

type passing struct {
	stub
	key  string
	path string
	body string
}

func (p *passing) Pass(_ context.Context, r *llm.PassRequest) (*llm.PassResponse, error) {
	p.key = r.Key
	p.path = r.Path
	if r.Body != nil {
		b, _ := io.ReadAll(r.Body)
		p.body = string(b)
	}
	return &llm.PassResponse{Status: 200, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("{}"))}, nil
}

func TestAVerbTheFormatDoesNotServeGoesToThePassthrough(t *testing.T) {
	p := &passing{}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:      []wire.Codec{geminiapi.New(geminiapi.Options{})},
		Passthrough: "stub",
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/v1beta/models/gemini-2.5-pro:countTokens",
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	if p.path != "/v1beta/models/gemini-2.5-pro:countTokens" {
		t.Fatalf("path = %q", p.path)
	}
	if !strings.Contains(p.body, "hi") {
		t.Fatalf("body = %q: the request goose already read must reach the backend", p.body)
	}
}

func TestAVerbTheFormatDoesNotServeIsRefused(t *testing.T) {
	srv := server(t, &stub{}, Options{Codecs: []wire.Codec{geminiapi.New(geminiapi.Options{})}})
	resp := post(t, srv, "/v1beta/models/gemini-2.5-pro:countTokens",
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestReadyTellsAnAnonymousCallerOnlyTheVerdict(t *testing.T) {
	p := &probing{err: errors.New("invalid x-api-key for account acct_12345")}
	srv := server(t, p, Options{APIKey: "secret"})

	resp, err := srv.Client().Get(srv.URL + "/ready")
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d: the verdict itself is public", resp.StatusCode)
	}
	body := read(t, resp)
	if strings.Contains(body, "acct_12345") || strings.Contains(body, "stub") {
		t.Fatalf("body = %s: an anonymous caller learned what the upstream said", body)
	}

	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/ready", nil)
	req.Header.Set("Authorization", "Bearer secret")
	resp, err = srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	if body := read(t, resp); !strings.Contains(body, "acct_12345") {
		t.Fatalf("body = %s: whoever holds the key needs the reason", body)
	}
}

func TestHealthDoesNotNameTheProvidersToAnAnonymousCaller(t *testing.T) {
	srv := server(t, &probing{}, Options{APIKey: "secret"})
	resp, err := srv.Client().Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	if body := read(t, resp); strings.Contains(body, "stub") {
		t.Fatalf("body = %s: which backends are registered is not a liveness fact", body)
	}
}

func TestReadinessIsNotAWayToSpendTheOperatorsQuota(t *testing.T) {
	p := &probing{}
	srv := server(t, p, Options{})
	for range 5 {
		resp, err := srv.Client().Get(srv.URL + "/ready")
		if err != nil {
			t.Fatal(err)
		}
		_ = resp.Body.Close()
	}
	if p.probes != 1 {
		t.Fatalf("probes = %d: five anonymous requests became five rounds of upstream calls", p.probes)
	}
}

func TestCORSVariesByOrigin(t *testing.T) {
	srv := server(t, &probing{}, Options{CORS: true, CORSOrigin: "https://leelsey.com"})
	resp, err := srv.Client().Get(srv.URL + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get("Vary"); !strings.Contains(got, "Origin") {
		t.Fatalf("Vary = %q: a cache would serve one origin's response to another", got)
	}
}

func TestConcurrencyCapRefusesRatherThanQueues(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	defer close(release)
	srv := server(t, &blocking{entered: entered, release: release}, Options{MaxConcurrent: 1})

	go func() {
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
		resp.Body.Close()
	}()
	<-entered

	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Fatal("a client told to come back needs to know when")
	}
}

type blocking struct {
	stub
	entered chan struct{}
	release chan struct{}
}

func (b *blocking) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	b.entered <- struct{}{}
	<-b.release
	return b.stub.Complete(ctx, a)
}

func TestRateLimitMetersOneCallerAtATime(t *testing.T) {
	now := time.Now()
	l := newLimiter(2, func() time.Time { return now })
	if !l.allow("a") || !l.allow("a") {
		t.Fatal("the allowance is spendable")
	}
	if l.allow("a") {
		t.Fatal("past the allowance the caller waits")
	}
	if !l.allow("b") {
		t.Fatal("one caller's spending is not another's")
	}
	now = now.Add(30 * time.Second)
	if !l.allow("a") {
		t.Fatal("half a minute is half the allowance back")
	}
}

func TestReadinessIsNotMetered(t *testing.T) {
	srv := server(t, &stub{}, Options{MaxPerMinute: 1})
	for range 4 {
		resp, err := srv.Client().Get(srv.URL + "/ready")
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode == http.StatusTooManyRequests {
			t.Fatal("an orchestrator must always get an answer")
		}
	}
}

func TestRefusedCredentialsAreMeteredAndLogged(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	srv := server(t, &stub{}, Options{
		APIKey: "secret", MaxPerMinute: 1, Logger: logger, LogRequests: true,
	})

	var last int
	for _, guess := range []string{"Bearer one", "Bearer two", "Bearer three", "Bearer four"} {
		header := http.Header{"Authorization": []string{guess}}
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, header)
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: a caller working through forged keys is what the allowance is for", last)
	}
	if !strings.Contains(buf.String(), "status=401") {
		t.Fatalf("log = %q: a refused call is the one an operator most needs to see", buf.String())
	}
}

func TestMetricsCountWhatWasServedAndSpent(t *testing.T) {
	m := NewMetrics()
	c, err := goose.New(goose.WithProvider(&stub{}), goose.WithUsage(m))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:  []wire.Codec{openaichat.New(openaichat.Options{})},
		Metrics: m,
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	resp.Body.Close()

	page, err := srv.Client().Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer page.Body.Close()
	body, _ := io.ReadAll(page.Body)
	for _, want := range []string{
		`goose_requests_total{format="openai",status="2xx"} 1`,
		`goose_upstream_attempts_total{provider="stub",model="m",outcome="ok"} 1`,
		`goose_tokens_total{provider="stub",model="m",kind="prompt"} 11`,
		"goose_inflight_requests 0",
	} {
		if !strings.Contains(string(body), want) {
			t.Fatalf("metrics missing %q:\n%s", want, body)
		}
	}
}

func TestMetricsQuoteALabelChosenByTheCaller(t *testing.T) {
	m := NewMetrics()
	_ = m.Record(context.Background(), usage.Record{
		Provider: "p", Model: `a" ,evil="1`, Outcome: llm.OutcomeOK,
	})
	w := httptest.NewRecorder()
	m.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	if strings.Contains(w.Body.String(), `,evil="1"`) {
		t.Fatalf("a caller wrote a label of their own:\n%s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `model="a\" ,evil=\"1"`) {
		t.Fatalf("body = %s", w.Body.String())
	}
}

func TestHostileRequestsProduceAStatus(t *testing.T) {
	srv := server(t, &stub{}, Options{APIKey: "secret", MaxBody: 1 << 10})
	big := strings.Repeat("x", 4<<10)

	cases := []struct {
		name, method, path, body string
		header                   http.Header
	}{
		{name: "no credential", method: "POST", path: "/v1/chat/completions", body: `{"model":"m","messages":[]}`},
		{name: "wrong credential", method: "POST", path: "/v1/chat/completions", body: `{}`,
			header: http.Header{"Authorization": {"Bearer wrong"}}},
		{name: "body over the limit", method: "POST", path: "/v1/chat/completions",
			body:   `{"model":"m","messages":[{"role":"user","content":"` + big + `"}]}`,
			header: http.Header{"Authorization": {"Bearer secret"}}},
		{name: "not json", method: "POST", path: "/v1/chat/completions", body: "\x00\xff not json",
			header: http.Header{"Authorization": {"Bearer secret"}}},
		{name: "deeply nested", method: "POST", path: "/v1/chat/completions",
			body:   strings.Repeat(`{"a":`, 200) + "1" + strings.Repeat("}", 200),
			header: http.Header{"Authorization": {"Bearer secret"}}},
		{name: "unknown path", method: "POST", path: "/nope", body: "{}",
			header: http.Header{"Authorization": {"Bearer secret"}}},
		{name: "wrong method", method: "GET", path: "/v1/chat/completions", body: "",
			header: http.Header{"Authorization": {"Bearer secret"}}},
		{name: "header repeated", method: "POST", path: "/v1/chat/completions", body: `{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
			header: http.Header{"Authorization": {"Bearer secret", "Bearer other"}, "X-Api-Key": {"a", "b"}}},
		{name: "credential in the query", method: "POST", path: "/v1/chat/completions?key=secret",
			body: `{"model":"m","messages":[{"role":"user","content":"hi"}]}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req, err := http.NewRequest(c.method, srv.URL+c.path, strings.NewReader(c.body))
			if err != nil {
				t.Fatal(err)
			}
			req.Header.Set("Content-Type", "application/json")
			for k, vs := range c.header {
				for _, v := range vs {
					req.Header.Add(k, v)
				}
			}
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatalf("no status at all: %v", err)
			}
			defer resp.Body.Close()
			io.Copy(io.Discard, resp.Body)
			if resp.StatusCode < 200 || resp.StatusCode > 599 {
				t.Fatalf("status %d", resp.StatusCode)
			}
		})
	}
}

func TestMetricsLabelAPrefixRouteByItsFormat(t *testing.T) {
	m := NewMetrics()
	c, err := goose.New(goose.WithProvider(&stub{}), goose.WithUsage(m))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:  []wire.Codec{geminiapi.New(geminiapi.Options{}), openaichat.New(openaichat.Options{})},
		Metrics: m,
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/v1beta/models/m:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, nil)
	resp.Body.Close()

	page, err := srv.Client().Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer page.Body.Close()
	body, _ := io.ReadAll(page.Body)
	if !strings.Contains(string(body), `goose_requests_total{format="google",status="2xx"} 1`) {
		t.Fatalf("metrics = %s", body)
	}
}

func TestAToolTheFormatCannotReportBackIsRefused(t *testing.T) {
	p := &stub{tools: true}
	c, err := goose.New(
		goose.WithProvider(p),
		goose.WithTools("mcp", []llm.Tool{{Name: "search", Schema: []byte(`{"type":"object"}`)}}),
		goose.WithRoutes(route.Rule{
			Name: "tooled", Match: route.Always(),
			Chain: []route.Target{{Provider: "stub"}}, Tools: []string{"mcp"},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{Codecs: []wire.Codec{ollamaapi.NewGenerate(ollamaapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/api/generate", `{"model":"m","prompt":"hi","stream":false}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("served a request whose answer it could not report: %s", body)
	}
	if p.seen != nil {
		t.Fatal("the request reached the backend")
	}
}

func TestAFormatServesWhatItCanRender(t *testing.T) {
	c, err := goose.New(
		goose.WithProvider(&stub{}),
		goose.WithTools("mcp", []llm.Tool{{Name: "search", Schema: []byte(`{"type":"object"}`)}}),
		goose.WithRoutes(route.Rule{
			Name: "tooled", Match: route.Always(),
			Chain: []route.Target{{Provider: "stub"}}, Tools: []string{"mcp"},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{Codecs: []wire.Codec{ollamaapi.NewChat(ollamaapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()
	resp := post(t, srv, "/api/chat", `{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
}

func TestModelListKeepsADisplayName(t *testing.T) {
	srv := server(t, &listing{}, Options{})
	resp, err := srv.Client().Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	for _, want := range []string{`"id":"m-1"`, `"display_name":"Model One"`, `"owned_by":"up"`} {
		if !strings.Contains(string(body), want) {
			t.Fatalf("models = %s, want %s", body, want)
		}
	}
}

type listing struct{ stub }

func (l *listing) Models(context.Context) ([]llm.Model, error) {
	return []llm.Model{{ID: "m-1", Display: "Model One", Owner: "up"}}, nil
}

func TestMeteringReadsAForwardedAddressOnlyWhenNamed(t *testing.T) {
	cases := []struct {
		name, header, value, want string
	}{
		{"not named", "", "203.0.113.7", "addr:192.0.2.1"},
		{"named", "X-Forwarded-For", "203.0.113.7", "addr:203.0.113.7"},
		// The nearest hop is what the proxy in front saw; the leftmost is what
		// the caller wrote, and a caller that can write its own bucket key has
		// no allowance at all.
		{"nearest hop, not what the caller wrote", "X-Forwarded-For", "203.0.113.7, 10.0.0.1, 10.0.0.2", "addr:10.0.0.2"},
		{"with a port", "X-Forwarded-For", "203.0.113.7:4001", "addr:203.0.113.7"},
		{"forwarded syntax", "Forwarded", `for="203.0.113.7"`, "addr:203.0.113.7"},
		{"ipv6", "X-Forwarded-For", "[2001:db8::1]:4001", "addr:2001:db8::1"},
		{"empty header", "X-Forwarded-For", "", "addr:192.0.2.1"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			r.RemoteAddr = "192.0.2.1:1234"
			if c.value != "" {
				r.Header.Set(cmp.Or(c.header, "X-Forwarded-For"), c.value)
			}
			if got := caller(r, c.header); got != c.want {
				t.Fatalf("caller = %q, want %q", got, c.want)
			}
		})
	}
}

func TestMeteringPrefersTheCredential(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.RemoteAddr = "192.0.2.1:1234"
	r.Header.Set("Authorization", "Bearer tenant-key")
	r.Header.Set("X-Forwarded-For", "203.0.113.7")
	if got := caller(r, "X-Forwarded-For"); got != "key:tenant-key" {
		t.Fatalf("caller = %q", got)
	}
}

func TestAStreamedCallReportsItsCostInATrailer(t *testing.T) {
	srv := server(t, &stub{}, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()

	if _, announced := resp.Trailer[http.CanonicalHeaderKey(UsageHeader)]; !announced {
		t.Fatalf("trailers = %+v: a client learns what to wait for from the headers", resp.Trailer)
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatal(err)
	}
	raw := resp.Trailer.Get(UsageHeader)
	if raw == "" {
		t.Fatal("the stream ended without saying what it cost")
	}
	var u llm.Usage
	if err := json.Unmarshal([]byte(raw), &u); err != nil {
		t.Fatalf("trailer = %q: %v", raw, err)
	}
	if u.Prompt != 11 || u.Completion != 2 {
		t.Fatalf("usage = %+v, want the figures the backend reported", u)
	}
}

func TestARefusedCallerIsCounted(t *testing.T) {
	m := NewMetrics()
	c, err := goose.New(goose.WithProvider(&stub{}), goose.WithUsage(m))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:       []wire.Codec{openaichat.New(openaichat.Options{})},
		Metrics:      m,
		MaxPerMinute: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	throttled := false
	for range 4 {
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
		resp.Body.Close()
		if resp.StatusCode == http.StatusTooManyRequests {
			throttled = true
		}
	}
	if !throttled {
		t.Fatal("the limit never bit")
	}
	resp, err := srv.Client().Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "goose_throttled_total") ||
		strings.Contains(string(body), "goose_throttled_total 0\n") {
		t.Fatalf("counters:\n%s", body)
	}
}

func TestAWriterThatCannotFlushIsReported(t *testing.T) {
	var out strings.Builder
	logger := slog.New(slog.NewTextHandler(&out, &slog.HandlerOptions{Level: slog.LevelWarn}))
	c, err := goose.New(goose.WithProvider(&stub{}))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs: []wire.Codec{openaichat.New(openaichat.Options{})},
		Logger: logger,
	})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	s.Handler().ServeHTTP(&unflushable{header: http.Header{}}, req)

	if !strings.Contains(out.String(), "cannot flush") {
		t.Fatalf("nothing was said about a buffered stream: %s", out.String())
	}
}

type unflushable struct {
	header http.Header
	body   strings.Builder
	status int
}

func (u *unflushable) Header() http.Header         { return u.header }
func (u *unflushable) Write(p []byte) (int, error) { return u.body.Write(p) }
func (u *unflushable) WriteHeader(status int)      { u.status = status }

func TestAPassthroughCallIsOnTheRecord(t *testing.T) {
	mem := usage.NewMemory()
	c, err := goose.New(goose.WithProvider(&passing{}), goose.WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:      []wire.Codec{openaichat.New(openaichat.Options{})},
		Passthrough: "stub",
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp, err := srv.Client().Get(srv.URL + "/v1/files")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	rows := mem.Records()
	if len(rows) != 1 {
		t.Fatalf("rows = %d: a delivered call left no record", len(rows))
	}
	if !rows[0].Sent || rows[0].Provider != "stub" || rows[0].Format != "passthrough" {
		t.Fatalf("row = %+v", rows[0])
	}
	if rows[0].Usage.Reported {
		t.Errorf("usage = %+v", rows[0].Usage)
	}
	if len(rows[0].Usage.Anomalies) == 0 {
		t.Errorf("the row does not say the figure is unknown: %+v", rows[0].Usage)
	}
}

func TestAClientMayHoldTheProxyKeyAndItsOwn(t *testing.T) {
	p := &stub{}
	srv := server(t, p, Options{APIKey: "proxy-secret"})

	req, _ := http.NewRequest("POST", srv.URL+"/v1/messages",
		strings.NewReader(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "callers-own")
	req.Header.Set("Authorization", "Bearer proxy-secret")

	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusUnauthorized {
		t.Fatal("a client holding the proxy's key was refused for also holding its own")
	}
	if p.seen == nil {
		t.Fatal("the call never reached a provider")
	}
	if p.seen.Credential.Key != "callers-own" {
		t.Fatalf("credential = %q: the caller's own key is the one that is not the proxy's", p.seen.Credential.Key)
	}
}

func TestAClientWithNoKeyIsStillRefused(t *testing.T) {
	srv := server(t, &stub{}, Options{APIKey: "proxy-secret"})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestEachWireListsModelsInItsOwnShape(t *testing.T) {
	srv := server(t, &probing{}, Options{Codecs: []wire.Codec{
		openaichat.New(openaichat.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
		ollamaapi.NewGenerate(ollamaapi.Options{}),
		geminiapi.New(geminiapi.Options{}),
	}})

	get := func(path string) map[string]any {
		resp, err := srv.Client().Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("%s answered %d", path, resp.StatusCode)
		}
		var body map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		return body
	}

	shared := get("/v1/models")
	entries, _ := shared["data"].([]any)
	if len(entries) != 1 {
		t.Fatalf("data = %+v", shared["data"])
	}
	first, _ := entries[0].(map[string]any)
	for _, key := range []string{"id", "object", "type"} {
		if first[key] == nil {
			t.Fatalf("entry = %+v: missing %q", first, key)
		}
	}
	if shared["has_more"] != false {
		t.Fatalf("has_more = %v: an Anthropic client reads the page flag", shared["has_more"])
	}

	tags := get("/api/tags")
	listed, _ := tags["models"].([]any)
	if len(listed) != 1 {
		t.Fatalf("tags = %+v", tags)
	}
	entry, _ := listed[0].(map[string]any)
	if entry["name"] != "served-model" || entry["model"] != "served-model" {
		t.Fatalf("tag = %+v: this wire names a model in both keys", entry)
	}

	gemini := get("/v1beta/models")
	listed, _ = gemini["models"].([]any)
	if len(listed) != 1 {
		t.Fatalf("models = %+v", gemini)
	}
	entry, _ = listed[0].(map[string]any)
	if entry["name"] != "models/served-model" {
		t.Fatalf("model = %+v: this wire names a model models/<id> everywhere", entry)
	}
	methods, _ := entry["supportedGenerationMethods"].([]any)
	if len(methods) != 2 {
		t.Fatalf("methods = %+v: a client checks what it may ask for", entry)
	}
}

func TestAListThatLeftAProviderOutSaysSo(t *testing.T) {
	srv := server(t, &stub{}, Options{})
	resp, err := srv.Client().Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if got := resp.Header.Get(DiagnosticsHeader); got != llm.CapNoModelList {
		t.Fatalf("diagnostics = %q: a backend with no list is not an empty one", got)
	}
}

func TestTheProxysOwnRefusalsSpeakTheClientsWire(t *testing.T) {
	codecs := []wire.Codec{
		openaichat.New(openaichat.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
	}
	refusing := server(t, &stub{}, Options{APIKey: "secret", Codecs: codecs})

	body := func(srv *httptest.Server, path string, header map[string]string) (int, map[string]any) {
		req, err := http.NewRequest(http.MethodPost, srv.URL+path,
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1}`))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		for k, v := range header {
			req.Header.Set(k, v)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		return resp.StatusCode, out
	}

	status, out := body(refusing, "/v1/messages", nil)
	if status != http.StatusUnauthorized {
		t.Fatalf("status = %d", status)
	}
	if out["type"] != "error" {
		t.Fatalf("refusal = %v: an Anthropic client reads its own shape", out)
	}
	inner, _ := out["error"].(map[string]any)
	if inner["type"] != "authentication_error" {
		t.Fatalf("refusal = %v", out)
	}

	status, out = body(refusing, "/v1/chat/completions", nil)
	if status != http.StatusUnauthorized || out["type"] == "error" {
		t.Fatalf("status = %d refusal = %v", status, out)
	}
	if inner, _ = out["error"].(map[string]any); inner == nil {
		t.Fatalf("refusal = %v", out)
	}

	throttling := server(t, &stub{}, Options{APIKey: "secret", MaxPerMinute: 1, Codecs: codecs})
	auth := map[string]string{"Authorization": "Bearer secret"}
	if status, _ := body(throttling, "/v1/messages", auth); status == http.StatusTooManyRequests {
		t.Fatal("the first request was throttled")
	}
	status, out = body(throttling, "/v1/messages", auth)
	if status != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", status)
	}
	if out["type"] != "error" {
		t.Fatalf("throttled = %v: an Anthropic client reads its own shape", out)
	}
}

func TestAPathThatIsNotServedIsRefusedInJSON(t *testing.T) {
	srv := server(t, &stub{}, Options{})
	resp, err := srv.Client().Get(srv.URL + "/v1/embeddings")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("body: %v", err)
	}
	if out["error"] == nil {
		t.Fatalf("body = %v", out)
	}

	resp, err = srv.Client().Get(srv.URL + "/v1/chat/completions")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d: a wrong method on a served path is 405", resp.StatusCode)
	}
	if resp.Header.Get("Allow") == "" {
		t.Fatal("a refused method must say which ones are served")
	}
	out = nil
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil || out["error"] == nil {
		t.Fatalf("body = %v: %v", out, err)
	}
}

func TestAPassthroughThatCannotForwardIsRefusedAtTheStart(t *testing.T) {
	c, err := goose.New(goose.WithProvider(&noPass{}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := New(c, Options{Passthrough: "nopass"}); err == nil {
		t.Fatal("a provider that cannot forward opaque requests must fail the start")
	}
	if _, err := New(c, Options{Passthrough: "absent"}); err == nil {
		t.Fatal("a passthrough naming no registered provider must fail the start")
	}
}

type noPass struct{ stub }

func (n *noPass) ID() string { return "nopass" }

func TestAPassthroughObeysTheClientKeyRule(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Saw", r.Header.Get("Authorization"))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer backend.Close()

	ask := func(allow bool) string {
		p, err := openai.New(openai.Options{ID: "up", BaseURL: backend.URL, AllowClientKey: allow})
		if err != nil {
			t.Fatal(err)
		}
		c, err := goose.New(goose.WithProvider(p))
		if err != nil {
			t.Fatal(err)
		}
		s, err := New(c, Options{
			Codecs:      []wire.Codec{openaichat.New(openaichat.Options{})},
			Passthrough: "up",
		})
		if err != nil {
			t.Fatal(err)
		}
		srv := httptest.NewServer(s.Handler())
		defer srv.Close()
		req, _ := http.NewRequest(http.MethodGet, srv.URL+"/v1/anything", nil)
		req.Header.Set("Authorization", "Bearer the-callers-own")
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		io.Copy(io.Discard, resp.Body)
		return resp.Header.Get("X-Saw")
	}

	if got := ask(false); got != "" {
		t.Fatalf("backend saw %q: the operator did not allow a caller's key", got)
	}
	if got := ask(true); got != "Bearer the-callers-own" {
		t.Fatalf("backend saw %q: the operator allowed it", got)
	}
}

func TestTheRetryHintIsReadableFromABrowser(t *testing.T) {
	srv := server(t, &stub{}, Options{CORS: true, CORSOrigin: "https://example.test"})
	req, _ := http.NewRequest(http.MethodGet, srv.URL+"/health", nil)
	req.Header.Set("Origin", "https://example.test")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if !strings.Contains(resp.Header.Get("Access-Control-Expose-Headers"), "Retry-After") {
		t.Fatalf("exposed = %q", resp.Header.Get("Access-Control-Expose-Headers"))
	}
}

func TestAThrottledRequestIsStillLogged(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, nil))
	srv := server(t, &stub{}, Options{
		APIKey: "secret", MaxPerMinute: 1, Logger: logger, LogRequests: true,
	})
	header := http.Header{"Authorization": []string{"Bearer secret"}}
	var last int
	for range 2 {
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, header)
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", last)
	}
	if !strings.Contains(buf.String(), "status=429") {
		t.Fatalf("log = %q: a refusal the proxy made itself is one an operator needs to see", buf.String())
	}
}

func TestAForgedForwardedAddressBuysNoFreshAllowance(t *testing.T) {
	srv := server(t, &stub{}, Options{
		APIKey: "secret", MaxPerMinute: 1, ForwardedHeader: "X-Forwarded-For",
	})
	var last int
	for _, claimed := range []string{"10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"} {
		header := http.Header{
			"Authorization":   []string{"Bearer guess"},
			"X-Forwarded-For": []string{claimed},
		}
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, header)
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: an unproven caller does not get to name its own bucket", last)
	}
}

func TestMetricsLabelsAreBounded(t *testing.T) {
	m := NewMetrics()
	long := strings.Repeat("m", maxLabel+1)
	_ = m.Record(context.Background(), usage.Record{Provider: "p", Model: long, Outcome: llm.OutcomeOK})
	for k := range m.attempts {
		if k.model == long {
			t.Fatal("a model name longer than any real one is a label a client chose, not a model")
		}
	}

	for i := range maxSeries * 2 {
		_ = m.Record(context.Background(), usage.Record{
			Provider: "p", Model: "model-" + strconv.Itoa(i), Outcome: llm.OutcomeOK,
		})
	}
	if len(m.attempts) > maxSeries+2 {
		t.Fatalf("series = %d: every distinct model is kept for the life of the process", len(m.attempts))
	}
}

func TestANonceBesideTheRealKeyBuysNoFreshAllowance(t *testing.T) {
	srv := server(t, &stub{}, Options{APIKey: "secret", MaxPerMinute: 1, ForwardedHeader: "X-Forwarded-For"})
	var last int
	for i, nonce := range []string{"n1", "n2", "n3", "n4"} {
		header := http.Header{
			"Authorization":   []string{"Bearer secret"},
			"X-Api-Key":       []string{nonce},
			"X-Forwarded-For": []string{"10.0.0." + strconv.Itoa(i)},
		}
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, header)
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: neither the credential nor the address a caller writes is what it proved", last)
	}
}

func TestAForgedForwardedChainDoesNotMintBuckets(t *testing.T) {
	srv := server(t, &stub{}, Options{MaxPerMinute: 2, ForwardedHeader: "X-Forwarded-For"})
	var last int
	for i := range 5 {
		// What a caller writes ends up left of what the proxy appends, so
		// varying it must not buy anything.
		header := http.Header{"X-Forwarded-For": []string{"10.9.9." + strconv.Itoa(i) + ", 203.0.113.7"}}
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, header)
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: the caller wrote the left of the chain, not the right", last)
	}
}

// The trailer is the caller's copy of the bill, so it says what the ledger
// says. A usage nobody counted is not a usage of zero.
func TestTheStreamTrailerSaysWhatTheLedgerSays(t *testing.T) {
	p := &stub{frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopEnd, TurnEnd: true},
	}}
	mem := usage.NewMemory()
	c, err := goose.New(goose.WithProvider(p), goose.WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`, nil)
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)

	var trailer llm.Usage
	if err := json.Unmarshal([]byte(resp.Trailer.Get(UsageHeader)), &trailer); err != nil {
		t.Fatalf("trailer = %q: %v", resp.Trailer.Get(UsageHeader), err)
	}
	rows := mem.Records()
	if len(rows) == 0 {
		t.Fatal("nothing was booked")
	}
	booked := rows[len(rows)-1].Usage
	if len(trailer.Anomalies) != len(booked.Anomalies) {
		t.Fatalf("trailer = %+v, booked = %+v: the caller's copy is missing what the row says",
			trailer.Anomalies, booked.Anomalies)
	}
}
