// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/openaichat"
)

func TestUpstreamErrorTextCanBeKeptOffTheReply(t *testing.T) {
	fail := &stub{err: &llm.Error{
		Kind: llm.KindRateLimit, Status: 429, Provider: "up", Code: "quota",
		Message: "organisation org-9f3a has exceeded its quota, contact billing@acme.example",
		Body:    `{"error":{"message":"organisation org-9f3a has exceeded its quota"}}`,
		Sent:    true,
	}}
	srv := server(t, fail, Options{RedactUpstream: true})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(body), "org-9f3a") || strings.Contains(string(body), "acme.example") {
		t.Fatalf("body = %s", body)
	}
	if !strings.Contains(string(body), "rate limiting") {
		t.Fatalf("body = %s: the client still needs to know what kind of failure it was", body)
	}
	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status = %d: the status is not the operator's secret", resp.StatusCode)
	}
}

func TestUpstreamErrorTextTravelsByDefault(t *testing.T) {
	fail := &stub{err: &llm.Error{
		Kind: llm.KindRateLimit, Status: 429, Provider: "up",
		Message: "you exceeded your current quota", Sent: true,
	}}
	srv := server(t, fail, Options{})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "you exceeded your current quota") {
		t.Fatalf("body = %s: on a proxy you run for yourself the vendor's sentence is the useful one", body)
	}
}

// A failure that arrives after the stream is primed is written into the body
// rather than through the error path, so an operator who asked for redaction
// gets none of it on the shape they actually serve.
func TestUpstreamErrorTextIsKeptOffAStreamedReplyToo(t *testing.T) {
	secret := "organisation org-9f3a has exceeded its quota, contact billing@acme.example"
	for _, tc := range []struct{ name, path, body string }{
		{"openai", "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`},
		{"anthropic", "/v1/messages", `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"stream":true}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := &stub{frames: []*llm.Frame{
				{Err: &llm.Error{
					Kind: llm.KindRateLimit, Status: 429, Provider: "up", Code: "org_9f3a_quota_exhausted",
					Message: secret,
					Body:    `{"error":{"message":"` + secret + `"}}`,
					Sent:    true,
				}},
			}}
			srv := server(t, p, Options{RedactUpstream: true})
			resp := post(t, srv, tc.path, tc.body, nil)
			defer resp.Body.Close()
			body, _ := io.ReadAll(resp.Body)
			for _, leak := range []string{"org-9f3a", "acme.example", "org_9f3a_quota_exhausted"} {
				if strings.Contains(string(body), leak) {
					t.Fatalf("%q reached the client: %s", leak, body)
				}
			}
			if !strings.Contains(string(body), "rate limiting") {
				t.Fatalf("body = %s: the client still needs to know what kind of failure it was", body)
			}
		})
	}
}

// The code is the vendor's own string, parsed out of its body, so redaction
// that leaves it behind leaves the operator's account named.
func TestRedactionCoversTheVendorCode(t *testing.T) {
	fail := &stub{err: &llm.Error{
		Kind: llm.KindRateLimit, Status: 429, Provider: "up", Code: "org_9f3a_quota_exhausted",
		Message: "organisation org-9f3a has exceeded its quota",
		Body:    `{"error":{"code":"org_9f3a_quota_exhausted"}}`,
		Sent:    true,
	}}
	srv := server(t, fail, Options{RedactUpstream: true})
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(body), "org_9f3a") {
		t.Fatalf("the vendor's own code reached the client: %s", body)
	}
}

// The row an operator keeps is not the sentence a stranger is shown. Redacting
// the reply must not redact the record of why it failed.
func TestRedactingTheReplyDoesNotRedactTheLedger(t *testing.T) {
	secret := "organisation org-9f3a has exceeded its quota"
	p := &stub{frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hel"}}},
		{Err: &llm.Error{
			Kind: llm.KindRateLimit, Status: 429, Provider: "up", Code: "org_9f3a",
			Message: secret, Body: `{"error":{"message":"` + secret + `"}}`, Sent: true,
		}},
	}}
	mem := usage.NewMemory()
	c, err := goose.New(goose.WithProvider(p), goose.WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{RedactUpstream: true, Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/v1/chat/completions",
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if strings.Contains(string(body), "org-9f3a") {
		t.Fatalf("the reply quoted the backend: %s", body)
	}
	rows := mem.Records()
	if len(rows) == 0 {
		t.Fatal("nothing was booked")
	}
	last := rows[len(rows)-1]
	if !strings.Contains(last.Err+last.Detail, "org-9f3a") {
		t.Fatalf("the operator's own record says %q: the ledger has its own switch", last.Err)
	}
}

// A passthrough failure is the proxy's to write, and a transport error's own
// sentence names the upstream's address, path and query in full.
func TestAPassthroughFailureIsRedactedToo(t *testing.T) {
	gone := httptest.NewServer(http.NotFoundHandler())
	base := gone.URL + "/internal/tenant-42"
	gone.Close()
	p, err := openai.New(openai.Options{ID: "up", BaseURL: base, APIKey: "operator-key"})
	if err != nil {
		t.Fatal(err)
	}
	srv := server(t, p, Options{Passthrough: "up", RedactUpstream: true})
	resp, err := srv.Client().Get(srv.URL + "/v1/files?purpose=x")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusBadGateway {
		t.Fatalf("status = %d", resp.StatusCode)
	}
	for _, leak := range []string{"tenant-42", strings.TrimPrefix(gone.URL, "http://"), "purpose=x"} {
		if strings.Contains(string(body), leak) {
			t.Fatalf("%q reached the client: %s", leak, body)
		}
	}
}
