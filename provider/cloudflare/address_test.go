// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudflare

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

type recorder struct {
	seen  []*http.Request
	reply string
}

func (r *recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	r.seen = append(r.seen, req)
	reply := r.reply
	if reply == "" {
		reply = `{"id":"c","model":"m","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	}
	return &http.Response{
		StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(reply)), Request: req,
	}, nil
}

func built(t *testing.T, reply string) (*Provider, *recorder) {
	t.Helper()
	p, err := New(Options{AccountID: "acct-1", APIKey: "token"})
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{reply: reply}
	if err := p.Bind(llm.Env{HTTP: &http.Client{Transport: rec}}); err != nil {
		t.Fatal(err)
	}
	return p, rec
}

// An operator names an account, not a URL. Workers AI serves the OpenAI shapes
// on one path and its own catalogue on another, and only the absolute URL says
// whether either is right.
func TestWhereAnAccountActuallySends(t *testing.T) {
	const base = "https://api.cloudflare.com/client/v4/accounts/acct-1"

	t.Run("a call", func(t *testing.T) {
		p, rec := built(t, "")
		_, err := p.Complete(context.Background(), &llm.Attempt{
			Model: "@cf/meta/llama-3.1-8b-instruct", Recorder: llm.NopRecorder{},
			Request: &llm.Request{Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		want := base + "/ai/v1/chat/completions"
		if got := rec.seen[0].URL.String(); got != want {
			t.Fatalf("asked %q, want %q", got, want)
		}
		if rec.seen[0].Header.Get("Authorization") != "Bearer token" {
			t.Errorf("authorization = %q", rec.seen[0].Header.Get("Authorization"))
		}
	})

	t.Run("the catalogue", func(t *testing.T) {
		p, rec := built(t, `{"success":true,"result":[]}`)
		if _, err := p.Models(context.Background()); err != nil {
			t.Fatal(err)
		}
		want := base + "/ai/models/search?per_page=50&page=1"
		if got := rec.seen[0].URL.String(); got != want {
			t.Fatalf("asked %q, want %q", got, want)
		}
	})

	t.Run("readiness reads the catalogue, not an endpoint nobody serves", func(t *testing.T) {
		p, rec := built(t, `{"success":true,"result":[]}`)
		if err := p.Probe(context.Background()); err != nil {
			t.Fatal(err)
		}
		if got := rec.seen[0].URL.Path; !strings.HasSuffix(got, "/ai/models/search") {
			t.Fatalf("asked %q", got)
		}
	})

	t.Run("a base that is not the OpenAI surface keeps the inherited listing", func(t *testing.T) {
		p, err := New(Options{AccountID: "acct-1", APIKey: "token", BaseURL: "https://gateway.example/v1"})
		if err != nil {
			t.Fatal(err)
		}
		rec := &recorder{reply: `{"data":[{"id":"m"}]}`}
		if err := p.Bind(llm.Env{HTTP: &http.Client{Transport: rec}}); err != nil {
			t.Fatal(err)
		}
		if _, err := p.Models(context.Background()); err != nil {
			t.Fatal(err)
		}
		want := "https://gateway.example/v1/models"
		if got := rec.seen[0].URL.String(); got != want {
			t.Fatalf("asked %q, want %q: a gateway is not an account", got, want)
		}
	})
}
