// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package azure

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func serve(t *testing.T, o Options, h http.HandlerFunc) (llm.Provider, *[]*http.Request) {
	t.Helper()
	var reqs []*http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		reqs = append(reqs, r)
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	o.BaseURL = srv.URL
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	return p, &reqs
}

func attempt() *llm.Attempt {
	return &llm.Attempt{
		Request:  &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
		Model:    "gpt-4o",
		Recorder: llm.NopRecorder{},
	}
}

func ok(w http.ResponseWriter, r *http.Request) {
	_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
}

func TestRollingSurfaceUsesStandardPaths(t *testing.T) {
	p, reqs := serve(t, Options{APIKey: "k"}, ok)
	if _, err := p.Complete(context.Background(), attempt()); err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	if r.URL.Path != "/openai/v1/chat/completions" {
		t.Fatalf("path = %q", r.URL.Path)
	}
	if r.Header.Get("api-key") != "k" {
		t.Fatalf("this host takes its own header, not a bearer: %v", r.Header)
	}
	if r.Header.Get("Authorization") != "" {
		t.Fatal("a bearer was sent as well")
	}
}

func TestDatedSurfaceIsDeploymentScoped(t *testing.T) {
	p, reqs := serve(t, Options{APIKey: "k", Deployment: "my-deploy", APIVersion: "2024-10-21"}, ok)
	if _, err := p.Complete(context.Background(), attempt()); err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	if r.URL.Path != "/openai/deployments/my-deploy/chat/completions" {
		t.Fatalf("path = %q", r.URL.Path)
	}
	if r.URL.Query().Get("api-version") != "2024-10-21" {
		t.Fatalf("query = %q", r.URL.RawQuery)
	}
}

func TestDatedSurfaceNeedsADeployment(t *testing.T) {
	if _, err := New(Options{BaseURL: "https://x", APIVersion: "2024-10-21"}); err == nil {
		t.Fatal("the deployment is in the path on that surface, so it cannot be omitted")
	}
}

func TestEntraTokenReplacesTheKeyHeader(t *testing.T) {
	p, reqs := serve(t, Options{
		APIKey: "k",
		Token:  func(context.Context) (string, error) { return "entra-token", nil },
	}, ok)
	if _, err := p.Complete(context.Background(), attempt()); err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	if r.Header.Get("Authorization") != "Bearer entra-token" {
		t.Fatalf("authorization = %q", r.Header.Get("Authorization"))
	}
	if r.Header.Get("api-key") != "" {
		t.Fatal("both credentials were sent; the token replaces the key")
	}
}

func TestResourceOrBaseURLIsRequired(t *testing.T) {
	if _, err := New(Options{APIKey: "k"}); err == nil || !strings.Contains(err.Error(), "resource") {
		t.Fatalf("err = %v", err)
	}
}
