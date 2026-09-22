// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudflare

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func TestAccountScopedURLAndBearer(t *testing.T) {
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		seen = r
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p, err := New(Options{AccountID: "acct", APIKey: "token", BaseURL: srv.URL + "/client/v4/accounts/acct/ai/v1"})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Complete(context.Background(), &llm.Attempt{
		Request:  &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
		Model:    "@cf/meta/llama-3.1-8b-instruct",
		Recorder: llm.NopRecorder{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(seen.URL.Path, "/ai/v1/chat/completions") {
		t.Fatalf("path = %q", seen.URL.Path)
	}
	if seen.Header.Get("Authorization") != "Bearer token" {
		t.Fatalf("authorization = %q", seen.Header.Get("Authorization"))
	}
}

func TestAccountIsRequired(t *testing.T) {
	if _, err := New(Options{APIKey: "k"}); err == nil {
		t.Fatal("an account-scoped URL cannot be built without an account")
	}
}

// Workers AI serves chat completions on the OpenAI shape but keeps its
// catalogue on its own account-scoped endpoint, so the inherited /models call
// reaches nothing and readiness never turns green.
func TestTheCatalogueIsReadFromTheAccountEndpoint(t *testing.T) {
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = append(asked, r.URL.Path)
		if !strings.HasSuffix(r.URL.Path, "/ai/models/search") {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"success":false,"errors":[{"message":"not found"}]}`))
			return
		}
		if r.Header.Get("Authorization") != "Bearer token" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"success":true,"result":[
			{"name":"@cf/meta/llama-3.1-8b-instruct","description":"Llama"},
			{"name":"@cf/openai/gpt-oss-120b"}]}`))
	}))
	defer srv.Close()

	p, err := New(Options{AccountID: "acct", APIKey: "token",
		BaseURL: srv.URL + "/client/v4/accounts/acct/ai/v1"})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "@cf/meta/llama-3.1-8b-instruct" ||
		models[1].ID != "@cf/openai/gpt-oss-120b" {
		t.Fatalf("models = %+v", models)
	}
	if models[0].Owner != "cloudflare" {
		t.Fatalf("owner = %q", models[0].Owner)
	}
	if err := p.Probe(context.Background()); err != nil {
		t.Fatalf("a reachable account reads as unready: %v", err)
	}
	for _, path := range asked {
		if !strings.HasSuffix(path, "/ai/models/search") {
			t.Fatalf("asked %q, which this backend does not serve", path)
		}
	}
}

// This host answers 200 and puts the refusal in the body, so a reader that only
// looks at the status reports a broken account as healthy.
func TestARefusalInTheBodyIsStillARefusal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"success":false,"errors":[{"code":10000,"message":"Authentication error"}],"result":null}`))
	}))
	defer srv.Close()

	p, err := New(Options{AccountID: "acct", APIKey: "token", MaxRetries: 0,
		BaseURL: srv.URL + "/client/v4/accounts/acct/ai/v1"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Models(context.Background()); err == nil {
		t.Fatal("a catalogue that was refused is not an empty catalogue")
	}
	if err := p.Probe(context.Background()); err == nil {
		t.Fatal("an account that cannot be read is not a ready one")
	}
}

// The page size is the server's to honour, so the end of the list is what the
// server says it is rather than what we asked for.
func TestTheCatalogueIsReadToTheEndWhateverThePageSize(t *testing.T) {
	const total = 80
	var requests int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		page := 1
		if v := r.URL.Query().Get("page"); v != "" {
			_, _ = fmt.Sscanf(v, "%d", &page)
		}
		const per = 20 // this host caps the page size below what was asked for
		var names []string
		for i := (page - 1) * per; i < page*per && i < total; i++ {
			names = append(names, fmt.Sprintf(`{"name":"@cf/m/%d"}`, i))
		}
		fmt.Fprintf(w, `{"success":true,"result":[%s],"result_info":{"page":%d,"per_page":%d,"count":%d,"total_count":%d}}`,
			strings.Join(names, ","), page, per, len(names), total)
	}))
	defer srv.Close()

	p, err := New(Options{AccountID: "acct", APIKey: "token",
		BaseURL: srv.URL + "/client/v4/accounts/acct/ai/v1"})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != total {
		t.Fatalf("listed %d of %d models in %d requests", len(models), total, requests)
	}
}

func TestAZeroValueProviderDoesNotCrashOnBind(t *testing.T) {
	var p Provider
	if err := p.Bind(llm.Env{HTTP: http.DefaultClient}); err == nil {
		t.Fatal("a provider that was never built cannot be bound")
	}
}
