// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package azure

import (
	"bytes"
	"context"
	"encoding/json"
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
		body, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(body))
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

// The Responses API names its deployment in the body, so it is not scoped
// under /deployments/ the way chat completions is.
func TestDatedSurfaceKeepsResponsesOffTheDeploymentPath(t *testing.T) {
	p, reqs := serve(t, Options{
		APIKey: "k", Deployment: "my-deploy", APIVersion: "2025-03-01-preview",
	}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[]}`))
	})
	a := attempt()
	a.Format = "openai-responses"
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	if r.URL.Path != "/openai/responses" {
		t.Fatalf("path = %q, want the deployment out of the path", r.URL.Path)
	}
	if r.URL.Query().Get("api-version") != "2025-03-01-preview" {
		t.Fatalf("query = %q", r.URL.RawQuery)
	}
	// With the deployment out of the path, the body is the only thing left
	// that says which one to serve.
	var sent struct {
		Model string `json:"model"`
	}
	raw, _ := io.ReadAll(r.Body)
	if err := json.Unmarshal(raw, &sent); err != nil {
		t.Fatal(err)
	}
	if sent.Model != "my-deploy" {
		t.Fatalf("body model = %q, want the deployment: nothing else names it", sent.Model)
	}
}

// The deployment belongs in the Responses body, not in every body: a chat call
// names its own model in the path, and rewriting the body one changes what a
// gateway in front of this host routes on.
func TestTheDeploymentDoesNotRewriteTheChatModel(t *testing.T) {
	p, reqs := serve(t, Options{
		APIKey: "k", Deployment: "my-deploy", APIVersion: "2024-10-21",
	}, ok)
	a := attempt()
	a.Model = "gpt-4o-mini"
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Model string `json:"model"`
	}
	raw, _ := io.ReadAll((*reqs)[0].Body)
	if err := json.Unmarshal(raw, &sent); err != nil {
		t.Fatal(err)
	}
	if sent.Model != "gpt-4o-mini" {
		t.Fatalf("body model = %q, want what was asked for", sent.Model)
	}
}

// The resource becomes the host the key is sent to, so anything that could
// end the host early or name another one is refused before a request is built.
func TestAResourceThatCouldMoveTheHostIsRefused(t *testing.T) {
	for _, resource := range []string{"x@127.0.0.1:18409/#", "evil.example.com/#", "a.b", "a b"} {
		if _, err := New(Options{Resource: resource, APIKey: "k"}); err == nil {
			t.Fatalf("resource %q was spliced into the host", resource)
		}
	}
	if _, err := New(Options{Resource: "my-Resource-01", APIKey: "k"}); err != nil {
		t.Fatalf("a resource's name was refused: %v", err)
	}
}
