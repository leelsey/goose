// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudflare

import (
	"context"
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
