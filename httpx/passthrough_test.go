// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/openaichat"
)

type seen struct {
	method, path, query string
	header              http.Header
}

type backend struct {
	*httptest.Server
	mu   sync.Mutex
	hits []seen
}

func newBackend(t *testing.T) *backend {
	t.Helper()
	b := &backend{}
	b.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.mu.Lock()
		b.hits = append(b.hits, seen{r.Method, r.URL.Path, r.URL.RawQuery, r.Header.Clone()})
		b.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	t.Cleanup(b.Close)
	return b
}

func (b *backend) seen() []seen {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]seen(nil), b.hits...)
}

func send(t *testing.T, srv *httptest.Server, method, path string, header http.Header) int {
	t.Helper()
	req, err := http.NewRequest(method, srv.URL+path, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k, vs := range header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}

// The catch-all takes every method on every path, including the paths the
// proxy's own endpoints answer GET on, so what those endpoints are excused
// from must not reach it.
func TestPassthroughIsBehindTheKeyCheckWhateverTheMethod(t *testing.T) {
	up := newBackend(t)
	p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL, APIKey: "operator-key"})
	if err != nil {
		t.Fatal(err)
	}
	srv := server(t, p, Options{APIKey: "proxy-secret", Passthrough: "up"})

	for _, c := range []struct{ method, path string }{
		{http.MethodOptions, "/v1/files"},
		{http.MethodPost, "/health"},
		{http.MethodDelete, "/ready"},
		{http.MethodPost, "/metrics"},
	} {
		if status := send(t, srv, c.method, c.path, nil); status != http.StatusUnauthorized {
			t.Errorf("%s %s = %d without the proxy's key, want 401", c.method, c.path, status)
		}
	}
	if hits := up.seen(); len(hits) != 0 {
		t.Fatalf("the backend was sent %d requests on the operator's key by a caller holding no key", len(hits))
	}
	if status := send(t, srv, http.MethodGet, "/health", nil); status != http.StatusOK {
		t.Fatalf("GET /health = %d: liveness stays public", status)
	}
	if status := send(t, srv, http.MethodPost, "/health", http.Header{"Authorization": {"Bearer proxy-secret"}}); status != http.StatusOK {
		t.Fatalf("POST /health = %d with the key: it is the backend's to answer", status)
	}
}

func TestPassthroughIsMeteredWhateverTheMethod(t *testing.T) {
	up := newBackend(t)
	p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL, APIKey: "operator-key"})
	if err != nil {
		t.Fatal(err)
	}
	srv := server(t, p, Options{Passthrough: "up", MaxPerMinute: 1})

	var throttled int
	for _, c := range []struct{ method, path string }{
		{http.MethodOptions, "/v1/files"},
		{http.MethodPost, "/health"},
		{http.MethodOptions, "/v1/files"},
		{http.MethodPut, "/ready"},
	} {
		if send(t, srv, c.method, c.path, nil) == http.StatusTooManyRequests {
			throttled++
		}
	}
	if hits := len(up.seen()); hits != 1 || throttled != 3 {
		t.Fatalf("backend reached %d times, %d throttled: an allowance of one is one", hits, throttled)
	}
}

func TestPassthroughHoldsAConcurrencySlotWhateverTheMethod(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	var taken atomic.Bool
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if taken.CompareAndSwap(false, true) {
			entered <- struct{}{}
			<-release
		}
	}))
	defer up.Close()
	p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL, APIKey: "operator-key"})
	if err != nil {
		t.Fatal(err)
	}
	srv := server(t, p, Options{Passthrough: "up", MaxConcurrent: 1})
	defer close(release)

	go func() {
		req, _ := http.NewRequest(http.MethodOptions, srv.URL+"/v1/files", nil)
		if resp, err := srv.Client().Do(req); err == nil {
			resp.Body.Close()
		}
	}()
	<-entered
	if status := send(t, srv, http.MethodPost, "/health", nil); status != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503: one slot was taken", status)
	}
}

func TestAQueryKeyIsACredentialOnPassthrough(t *testing.T) {
	cases := []struct {
		name  string
		opts  google.Options
		proxy string
		key   string
		want  string
	}{
		{"the proxy's own", google.Options{APIKey: "operator-key"}, "proxy-secret", "proxy-secret", "operator-key"},
		{"a caller's the operator did not allow", google.Options{APIKey: "operator-key"}, "", "callers-own", "operator-key"},
		{"a caller's with no key of the operator's", google.Options{}, "", "callers-own", ""},
		{"a caller's the operator allowed", google.Options{AllowClientKey: true}, "", "callers-own", "callers-own"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			up := newBackend(t)
			o := c.opts
			o.ID, o.BaseURL = "g", up.URL
			p, err := google.New(o)
			if err != nil {
				t.Fatal(err)
			}
			srv := server(t, p, Options{
				APIKey: c.proxy, Passthrough: "g",
				Codecs: []wire.Codec{geminiapi.New(geminiapi.Options{})},
			})
			send(t, srv, http.MethodGet, "/v1beta/files?pageSize=2&key="+c.key+"&access_token=tok", nil)

			hits := up.seen()
			if len(hits) != 1 {
				t.Fatalf("hits = %d", len(hits))
			}
			if hits[0].query != "pageSize=2" {
				t.Fatalf("query = %q: a credential went upstream in the URL, or the rest of the query did not", hits[0].query)
			}
			if got := hits[0].header.Get("x-goog-api-key"); got != c.want {
				t.Fatalf("x-goog-api-key = %q, want %q", got, c.want)
			}
		})
	}
}

// A key's header says which vendor minted it, and the passthrough holds it to
// that as the routed path does. A Gemini client's countTokens is forwarded as
// it came, and its own key must not reach Anthropic with it.
func TestAPassthroughHandsACallersKeyOnlyToItsOwnVendor(t *testing.T) {
	count := "/v1beta/models/gemini-2.5-flash:countTokens"
	for _, c := range []struct {
		name, path string
		header     http.Header
		want       string
	}{
		{"a Google key in the query", count + "?key=google-key", nil, ""},
		{"a Google key in its header", count, http.Header{"X-Goog-Api-Key": {"google-key"}}, ""},
		{"an Anthropic key", "/v1/messages/count_tokens", http.Header{"X-Api-Key": {"anthropic-key"}}, "anthropic-key"},
		{"a bearer, which names no vendor", "/v1/files", http.Header{"Authorization": {"Bearer own-key"}}, "own-key"},
	} {
		t.Run(c.name, func(t *testing.T) {
			up := newBackend(t)
			p, err := anthropic.New(anthropic.Options{
				ID: "a", BaseURL: up.URL, AllowClientKey: true, KeyScheme: llm.SchemeAnthropic,
			})
			if err != nil {
				t.Fatal(err)
			}
			srv := server(t, p, Options{Passthrough: "a", Codecs: []wire.Codec{
				anthropicmsg.New(anthropicmsg.Options{}), geminiapi.New(geminiapi.Options{}),
			}})
			resp := post(t, srv, c.path, `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, c.header)
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("status = %d: %s", resp.StatusCode, read(t, resp))
			}
			hits := up.seen()
			if len(hits) != 1 {
				t.Fatalf("hits = %d", len(hits))
			}
			if got := hits[0].header.Get("x-api-key"); got != c.want {
				t.Fatalf("x-api-key = %q, want %q: a key reaches the backend it was minted for and no other", got, c.want)
			}
			withheld := strings.Contains(resp.Header.Get(DiagnosticsHeader), llm.PolicyKeyWithheld)
			if withheld != (c.want == "") {
				t.Fatalf("diagnostics = %q: the caller is told its key was held back, and only then",
					resp.Header.Get(DiagnosticsHeader))
			}
		})
	}
}

// A query that url.ParseQuery cannot read is refused before anything is sent.
// That is the proxy refusing the caller's request, in the caller's wire, and
// not an upstream failure a client retries or an attempt on the record.
func TestAQueryTheProxyCannotReadIsTheCallersMistake(t *testing.T) {
	for _, c := range []struct {
		name, method, path, field, want string
	}{
		{"a semicolon", http.MethodGet, "/v1/files?q=a;b", "type", "invalid_request"},
		{"a bad escape", http.MethodGet, "/v1/files?q=%zz", "type", "invalid_request"},
		{"on a verb the Gemini wire forwards", http.MethodPost, "/v1beta/models/m:countTokens?q=%zz", "status", "INVALID_ARGUMENT"},
	} {
		t.Run(c.name, func(t *testing.T) {
			up := newBackend(t)
			p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL, APIKey: "operator-key"})
			if err != nil {
				t.Fatal(err)
			}
			mem := usage.NewMemory()
			client, err := goose.New(goose.WithProvider(p), goose.WithUsage(mem))
			if err != nil {
				t.Fatal(err)
			}
			s, err := New(client, Options{Passthrough: "up", RedactUpstream: true, Codecs: []wire.Codec{
				openaichat.New(openaichat.Options{}), geminiapi.New(geminiapi.Options{}),
			}})
			if err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(s.Handler())
			defer srv.Close()

			req, err := http.NewRequest(c.method, srv.URL+c.path,
				strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
			if err != nil {
				t.Fatal(err)
			}
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			var out struct{ Error map[string]any }
			_ = json.NewDecoder(resp.Body).Decode(&out)
			if resp.StatusCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %v", resp.StatusCode, out)
			}
			if out.Error[c.field] != c.want {
				t.Fatalf("refusal = %v: the caller's own wire", out)
			}
			if message, _ := out.Error["message"].(string); strings.Contains(message, "upstream") {
				t.Fatalf("message = %q: no upstream saw the request", message)
			}
			if hits := up.seen(); len(hits) != 0 {
				t.Fatalf("the backend was sent %d requests", len(hits))
			}
			if rows := mem.Records(); len(rows) != 0 {
				t.Fatalf("rows = %+v: nothing was attempted", rows)
			}
		})
	}
}

type failing struct {
	stub
	err *llm.Error
}

func (f *failing) Pass(context.Context, *llm.PassRequest) (*llm.PassResponse, error) {
	return nil, f.err
}

// A call that never left cost nothing, and its row says so; one that went and
// failed may well have been billed, and nobody counted it.
func TestAPassthroughThatFailedBeforeSendingIsBookedAsUnsent(t *testing.T) {
	for _, c := range []struct {
		name    string
		err     *llm.Error
		outcome llm.Outcome
	}{
		{"never sent", &llm.Error{Kind: llm.KindAuth, Provider: "stub", Message: "the token could not be minted"}, llm.OutcomeError},
		// A timeout waiting on an answer already sent is the attempt ending, not
		// the backend's own failure, so it is booked cancelled, as the routed
		// path books the same kind of failure.
		{"sent", &llm.Error{Kind: llm.KindTimeout, Provider: "stub", Message: "the backend did not answer", Sent: true}, llm.OutcomeCancelled},
	} {
		t.Run(c.name, func(t *testing.T) {
			mem := usage.NewMemory()
			client, err := goose.New(goose.WithProvider(&failing{err: c.err}), goose.WithUsage(mem))
			if err != nil {
				t.Fatal(err)
			}
			s, err := New(client, Options{Passthrough: "stub", Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
			if err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(s.Handler())
			defer srv.Close()

			if status := send(t, srv, http.MethodGet, "/v1/files", nil); status != http.StatusBadGateway {
				t.Fatalf("status = %d", status)
			}
			rows := mem.Records()
			if len(rows) != 1 || rows[0].Sent != c.err.Sent || rows[0].Outcome != c.outcome {
				t.Fatalf("rows = %+v: the attempt is on the record, as sent only if it left", rows)
			}
			if unknown := len(rows[0].Usage.Anomalies) > 0; unknown != c.err.Sent {
				t.Fatalf("usage = %+v: only a call that left has a figure nobody counted", rows[0].Usage)
			}
		})
	}
}

// A body over the limit is the caller's mistake on the passthrough as on the
// routed path. One that announces its length is refused before anything is
// sent; one that does not is cut off by the listener partway through being
// forwarded, and is answered the same way rather than as an upstream failure a
// client retries and the ledger blames on the backend.
func TestAPassthroughBodyOverTheLimitIsTheCallersMistake(t *testing.T) {
	for _, c := range []struct {
		name  string
		sized bool
	}{
		{"of announced length", true},
		{"of unannounced length", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			var hits atomic.Int32
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
			}))
			defer up.Close()
			p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL, APIKey: "operator-key"})
			if err != nil {
				t.Fatal(err)
			}
			mem := usage.NewMemory()
			client, err := goose.New(goose.WithProvider(p), goose.WithUsage(mem))
			if err != nil {
				t.Fatal(err)
			}
			s, err := New(client, Options{
				Passthrough: "up", MaxBody: 64, RedactUpstream: true,
				Codecs: []wire.Codec{openaichat.New(openaichat.Options{})},
			})
			if err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(s.Handler())
			defer srv.Close()

			var body io.Reader = strings.NewReader(strings.Repeat("x", 4096))
			if !c.sized {
				body = io.MultiReader(body)
			}
			req, err := http.NewRequest(http.MethodPost, srv.URL+"/v1/files", body)
			if err != nil {
				t.Fatal(err)
			}
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			var out struct{ Error map[string]any }
			_ = json.NewDecoder(resp.Body).Decode(&out)
			if resp.StatusCode != http.StatusRequestEntityTooLarge {
				t.Fatalf("status = %d, want 413: %v", resp.StatusCode, out)
			}
			if out.Error["type"] != string(llm.KindInvalidRequest) {
				t.Fatalf("refusal = %v: the mistake is the caller's", out)
			}
			if message, _ := out.Error["message"].(string); strings.Contains(message, "upstream") {
				t.Fatalf("message = %q: the backend did not fail", message)
			}
			rows := mem.Records()
			if c.sized {
				if hits.Load() != 0 || len(rows) != 0 {
					t.Fatalf("backend reached %d times, rows = %+v: nothing was to be sent", hits.Load(), rows)
				}
				return
			}
			if len(rows) != 1 || rows[0].Outcome != llm.OutcomeRejected || rows[0].Status != http.StatusRequestEntityTooLarge {
				t.Fatalf("rows = %+v: the row books goose's refusal, not the backend's error", rows)
			}
		})
	}
}

// TRACE has a server echo back the request it was sent, and the passthrough
// attaches the operator's credentials to what it sends, so a backend or
// anything between that answers TRACE would hand them to the caller. CONNECT
// asks for a tunnel, which no vendor's API is. Neither is sent on; the methods
// the APIs use still are.
func TestPassthroughForwardsOnlyTheMethodsAnAPIUses(t *testing.T) {
	up := newBackend(t)
	p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL, APIKey: "operator-key"})
	if err != nil {
		t.Fatal(err)
	}
	srv := server(t, p, Options{Passthrough: "up"})

	for _, c := range []struct{ method, path string }{
		{http.MethodTrace, "/v1/files"},
		{http.MethodTrace, "/v1/chat/completions"},
		{http.MethodConnect, "/v1/files"},
		{"TRACK", "/v1/files"},
	} {
		req, err := http.NewRequest(c.method, srv.URL+c.path, nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusMethodNotAllowed {
			t.Errorf("%s %s = %d, want 405", c.method, c.path, resp.StatusCode)
		}
		if allow := resp.Header.Get("Allow"); allow == "" || strings.Contains(allow, c.method) {
			t.Errorf("%s %s: Allow = %q", c.method, c.path, allow)
		}
	}
	if hits := up.seen(); len(hits) != 0 {
		t.Fatalf("the backend was sent %+v on the operator's key", hits)
	}

	methods := []string{http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut,
		http.MethodPatch, http.MethodDelete, http.MethodOptions}
	for _, method := range methods {
		if status := send(t, srv, method, "/v1/files", nil); status != http.StatusOK {
			t.Errorf("%s /v1/files = %d, want the backend's 200", method, status)
		}
	}
	if hits := up.seen(); len(hits) != len(methods) {
		t.Fatalf("the backend was reached %d times, want %d", len(hits), len(methods))
	}
}

func TestTheProxyKeyAndTheCallersMayBeOnEitherHeader(t *testing.T) {
	routed := func(t *testing.T, path, body string, header http.Header, query string) string {
		t.Helper()
		p := &stub{}
		srv := server(t, p, Options{APIKey: "proxy-secret", Codecs: []wire.Codec{
			anthropicmsg.New(anthropicmsg.Options{}), openaichat.New(openaichat.Options{}), geminiapi.New(geminiapi.Options{}),
		}})
		resp := post(t, srv, path+query, body, header)
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("%s: status %d: %s", path, resp.StatusCode, read(t, resp))
		}
		return p.seen.Credential.Key
	}
	anthropicBody := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`
	openaiBody := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	geminiBody := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`
	for _, c := range []struct {
		name, path, body, query string
		header                  http.Header
	}{
		{"anthropic, proxy key on x-api-key", "/v1/messages", anthropicBody, "",
			http.Header{"X-Api-Key": {"proxy-secret"}, "Authorization": {"Bearer callers-own"}}},
		{"anthropic, proxy key on the bearer", "/v1/messages", anthropicBody, "",
			http.Header{"X-Api-Key": {"callers-own"}, "Authorization": {"Bearer proxy-secret"}}},
		{"openai, proxy key on the bearer", "/v1/chat/completions", openaiBody, "",
			http.Header{"X-Api-Key": {"callers-own"}, "Authorization": {"Bearer proxy-secret"}}},
		{"google, proxy key on the header", "/v1beta/models/m:generateContent", geminiBody, "?key=callers-own",
			http.Header{"X-Goog-Api-Key": {"proxy-secret"}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := routed(t, c.path, c.body, c.header, c.query); got != "callers-own" {
				t.Fatalf("credential = %q: the one that is not the proxy's is the one handed on", got)
			}
		})
	}

	for _, c := range []struct {
		name   string
		header http.Header
	}{
		{"passthrough, proxy key on x-api-key", http.Header{"X-Api-Key": {"proxy-secret"}, "Authorization": {"Bearer callers-own"}}},
		{"passthrough, proxy key on the bearer", http.Header{"X-Api-Key": {"callers-own"}, "Authorization": {"Bearer proxy-secret"}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			up := newBackend(t)
			p, err := anthropic.New(anthropic.Options{ID: "a", BaseURL: up.URL, AllowClientKey: true})
			if err != nil {
				t.Fatal(err)
			}
			srv := server(t, p, Options{APIKey: "proxy-secret", Passthrough: "a"})
			if status := send(t, srv, http.MethodGet, "/v1/models/xyz", c.header); status != http.StatusOK {
				t.Fatalf("status = %d", status)
			}
			hits := up.seen()
			if len(hits) != 1 || hits[0].header.Get("x-api-key") != "callers-own" {
				t.Fatalf("hits = %+v: the caller's own key did not reach the backend", hits)
			}
			if strings.Contains(hits[0].header.Get("Authorization"), "proxy-secret") {
				t.Fatal("the proxy's own key was handed to the backend")
			}
		})
	}
}
