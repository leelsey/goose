// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openrouter

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func TestIdentityHeadersTravelWhenSet(t *testing.T) {
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		seen = r
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1}}`))
	}))
	defer srv.Close()

	p, err := New(Options{BaseURL: srv.URL, APIKey: "sk", Referer: "https://leelsey.com", Title: "goose"})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Complete(context.Background(), &llm.Attempt{
		Request:  &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
		Model:    "anthropic/claude-sonnet-4.5",
		Recorder: llm.NopRecorder{},
	})
	if err != nil {
		t.Fatal(err)
	}
	if seen.Header.Get("HTTP-Referer") != "https://leelsey.com" || seen.Header.Get("X-Title") != "goose" {
		t.Fatalf("headers = %v", seen.Header)
	}
	if resp.Usage.Prompt != 4 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestPreferencesPinTheUpstreamOrder(t *testing.T) {
	pinned := Options{Providers: []string{"anthropic", "google"}}
	pref := pinned.Preferences()
	if pref == nil {
		t.Fatal("pinned providers must reach the request")
	}
	if pref["allow_fallbacks"] != false {
		t.Fatal("a silent fallback inside the gateway hides which vendor served the call")
	}
	open := Options{AllowFallbacks: true}
	if open.Preferences() != nil {
		t.Fatal("the default needs no block at all")
	}
}

func TestUnsetHeadersAreNotSent(t *testing.T) {
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		seen = r
		_, _ = w.Write([]byte(`{"id":"c","choices":[]}`))
	}))
	defer srv.Close()
	p, _ := New(Options{BaseURL: srv.URL, APIKey: "sk"})
	_, _ = p.Complete(context.Background(), &llm.Attempt{
		Request:  &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
		Recorder: llm.NopRecorder{},
	})
	if _, ok := seen.Header["Http-Referer"]; ok {
		t.Fatal("an empty identity header must not be sent")
	}
}

// OpenRouter refuses a model id it does not have with the 400 a malformed
// request gets. Only a malformed request fails the same on every backend, so
// the chain must be able to walk on past the rest.
func TestAModelOpenRouterDoesNotHaveIsNotAMalformedRequest(t *testing.T) {
	for body, want := range map[string]llm.ErrorKind{
		`{"error":{"message":"claude-sonnet-5 is not a valid model ID","code":400}}`: llm.KindNotFound,
		`{"error":{"message":"'model' must be a string","code":400}}`:                llm.KindInvalidRequest,
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(body))
		}))
		p, err := New(Options{BaseURL: srv.URL, APIKey: "sk"})
		if err != nil {
			t.Fatal(err)
		}
		a := &llm.Attempt{
			Model:    "claude-sonnet-5",
			Request:  &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
			Recorder: llm.NopRecorder{},
		}
		_, err = p.Complete(context.Background(), a)
		if e := llm.AsError(err); e.Kind != want {
			t.Errorf("unary %s: err = %+v, want %s", body, e, want)
		}
		_, err = p.Stream(context.Background(), a)
		if e := llm.AsError(err); e.Kind != want {
			t.Errorf("stream %s: err = %+v, want %s", body, e, want)
		}
		srv.Close()
	}
}

func TestPinnedVendorsTravelOnEveryRequest(t *testing.T) {
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p, err := New(Options{BaseURL: srv.URL, APIKey: "k", Providers: []string{"anthropic", "google"}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"provider"`, `"order":["anthropic","google"]`, `"allow_fallbacks":false`} {
		if !strings.Contains(body, want) {
			t.Errorf("body is missing %s:\n%s", want, body)
		}
	}
}
