// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	neturl "net/url"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// Azure names its own auth header and pins an api-version, and Entra mints a
// token instead; a passthrough that hard-codes the OpenAI shape reaches none
// of the three.
func TestPassthroughCarriesTheConfiguredCredential(t *testing.T) {
	var (
		auth     string
		apiKey   string
		rawQuery string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, apiKey, rawQuery = r.Header.Get("Authorization"), r.Header.Get("api-key"), r.URL.RawQuery
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	t.Run("named header and pinned query", func(t *testing.T) {
		p, err := New(Options{ID: "az", BaseURL: srv.URL, APIKey: "k",
			AuthHeader: "api-key", Query: "api-version=2025-01-01-preview"})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.Pass(context.Background(), &llm.PassRequest{
			Method: http.MethodPost, Path: "/deployments/d/chat/completions",
			Header: http.Header{}, Body: strings.NewReader("{}"),
		})
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if apiKey != "k" {
			t.Errorf("api-key = %q, want the configured key", apiKey)
		}
		if auth != "" {
			t.Errorf("Authorization = %q, want nothing: this backend names another header", auth)
		}
		if rawQuery != "api-version=2025-01-01-preview" {
			t.Errorf("query = %q, want the pinned api-version", rawQuery)
		}
	})

	t.Run("both are carried", func(t *testing.T) {
		p, err := New(Options{ID: "az", BaseURL: srv.URL, APIKey: "k",
			AuthHeader: "api-key", Query: "api-version=2025-01-01-preview"})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.Pass(context.Background(), &llm.PassRequest{
			Method: http.MethodPost, Path: "/x", Query: "include=usage",
			Header: http.Header{}, Body: strings.NewReader("{}"),
		})
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		got, err := neturl.ParseQuery(rawQuery)
		if err != nil {
			t.Fatal(err)
		}
		if got.Get("include") != "usage" {
			t.Errorf("query = %q: the caller's own parameter was dropped", rawQuery)
		}
		if got.Get("api-version") != "2025-01-01-preview" {
			t.Errorf("query = %q: the pinned api-version was dropped", rawQuery)
		}
		if len(got["api-version"]) != 1 {
			t.Errorf("query = %q: this backend refuses the same parameter twice", rawQuery)
		}
	})

	t.Run("a query that cannot be read is refused", func(t *testing.T) {
		p, err := New(Options{ID: "az", BaseURL: srv.URL, APIKey: "k",
			AuthHeader: "api-key", Query: "api-version=2025-01-01-preview"})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.Pass(context.Background(), &llm.PassRequest{
			Method: http.MethodPost, Path: "/x", Query: "api-version=100%",
			Header: http.Header{}, Body: strings.NewReader("{}"),
		})
		if err == nil {
			resp.Body.Close()
			t.Fatal("appending to a query nobody can read sends the same parameter twice")
		}
		if got := llm.AsError(err).Kind; got != llm.KindInvalidRequest {
			t.Errorf("kind = %q, want %q: the caller wrote it", got, llm.KindInvalidRequest)
		}
	})

	t.Run("caller keeps its own parameter", func(t *testing.T) {
		p, err := New(Options{ID: "az", BaseURL: srv.URL, APIKey: "k",
			AuthHeader: "api-key", Query: "api-version=2025-01-01-preview"})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.Pass(context.Background(), &llm.PassRequest{
			Method: http.MethodPost, Path: "/x", Query: "api-version=2024-10-21&n=1",
			Header: http.Header{}, Body: strings.NewReader("{}"),
		})
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if rawQuery != "api-version=2024-10-21&n=1" {
			t.Errorf("query = %q: the caller's api-version was not kept", rawQuery)
		}
	})

	t.Run("signature", func(t *testing.T) {
		p, err := New(Options{ID: "az", BaseURL: srv.URL, AuthHeader: "api-key",
			Sign: func(_ context.Context, r *http.Request, _ []byte) error {
				r.Header.Set("Authorization", "Bearer minted")
				return nil
			}})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.Pass(context.Background(), &llm.PassRequest{
			Method: http.MethodPost, Path: "/x", Header: http.Header{}, Body: strings.NewReader("{}"),
		})
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if auth != "Bearer minted" {
			t.Errorf("Authorization = %q, want the minted token", auth)
		}
	})
}

func TestMergeQuery(t *testing.T) {
	for _, tc := range []struct {
		name, configured, caller, want string
		wantErr                        bool
	}{
		{name: "nothing configured", caller: "a=1&b=2", want: "a=1&b=2"},
		{name: "nothing from the caller", configured: "api-version=v", want: "api-version=v"},
		{name: "both", configured: "api-version=v", caller: "n=1", want: "api-version=v&n=1"},
		{name: "the caller wins", configured: "api-version=v", caller: "api-version=w", want: "api-version=w"},
		{name: "several configured are ordered", configured: "b=2&a=1", caller: "n=1", want: "a=1&b=2&n=1"},
		{name: "a value is escaped once", configured: "api-version=a b&", caller: "n=1", want: "api-version=a+b&n=1"},
		{name: "an unreadable caller is refused", configured: "api-version=v", caller: "n=100%", wantErr: true},
		{name: "an unreadable configured query is refused", configured: "api-version=%zz", caller: "n=1", wantErr: true},
		{name: "nothing to merge is left alone", caller: "n=100%", want: "n=100%"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := mergeQuery(tc.configured, tc.caller)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("got %q, want a refusal: appending to a query nobody can read sends a parameter twice", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Fatalf("mergeQuery(%q, %q) = %q, want %q", tc.configured, tc.caller, got, tc.want)
			}
		})
	}
}

func pass(t *testing.T, o Options, r *llm.PassRequest) *http.Request {
	t.Helper()
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	o.BaseURL = srv.URL + o.BaseURL
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Method == "" {
		r.Method = http.MethodPost
	}
	if r.Header == nil {
		r.Header = http.Header{}
	}
	r.Body = strings.NewReader("{}")
	resp, err := p.Pass(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	return seen
}

// Every OpenAI-shaped base ends in its version, and a client puts the version
// in front of every path it asks for.
func TestAPassthroughPathSaysTheVersionOnce(t *testing.T) {
	for _, tc := range []struct{ base, asked, want string }{
		{"/v1", "/v1/embeddings", "/v1/embeddings"},
		{"/v1/", "/v1/files", "/v1/files"},
		{"/api/v1", "/v1/files", "/api/v1/files"},
		{"/openai/v1", "/v1/files", "/openai/v1/files"},
		{"/openai/v1", "/openai/v1/files", "/openai/v1/files"},
		{"/openai", "/openai/deployments/d/chat/completions", "/openai/deployments/d/chat/completions"},
		{"", "/v1/anything", "/v1/anything"},
		{"/v1", "/v1beta/models", "/v1/v1beta/models"},
	} {
		seen := pass(t, Options{APIKey: "k", BaseURL: tc.base}, &llm.PassRequest{Path: tc.asked})
		if seen.URL.Path != tc.want {
			t.Errorf("base %q, asked %q: upstream saw %q, want %q", tc.base, tc.asked, seen.URL.Path, tc.want)
		}
	}
}

// The gate takes a key in the query, and a backend handed one there keeps it in
// its logs with the rest of the address.
func TestAPassthroughSendsNoKeyInTheQuery(t *testing.T) {
	seen := pass(t, Options{APIKey: "k"}, &llm.PassRequest{Path: "/v1/files", Query: "key=GATE&limit=2"})
	if seen.URL.RawQuery != "limit=2" {
		t.Fatalf("query = %q", seen.URL.RawQuery)
	}
	seen = pass(t, Options{APIKey: "k", Query: "api-version=v"}, &llm.PassRequest{Path: "/x", Query: "key=GATE"})
	if seen.URL.RawQuery != "api-version=v" {
		t.Fatalf("query = %q", seen.URL.RawQuery)
	}
}

// A routed call carries the configured headers, organisation and project, and
// an unrouted one is billed and admitted by the same.
func TestAPassthroughCarriesWhatTheOperatorConfigured(t *testing.T) {
	o := Options{APIKey: "operator-key", Organisation: "org-operator", Project: "proj-operator",
		Header: http.Header{"X-Gateway-Auth": {"gw-secret"}}}
	seen := pass(t, o, &llm.PassRequest{Path: "/v1/embeddings", Key: "caller-key", Header: http.Header{
		"Openai-Organization": {"org-caller"}, "Openai-Project": {"proj-caller"}, "X-Gateway-Auth": {"forged"},
	}})
	for name, want := range map[string]string{
		"X-Gateway-Auth": "gw-secret", "OpenAI-Organization": "org-operator",
		"OpenAI-Project": "proj-operator", "Authorization": "Bearer operator-key",
	} {
		if got := seen.Header.Values(name); len(got) != 1 || got[0] != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}

	t.Run("the caller's own key", func(t *testing.T) {
		o := Options{AllowClientKey: true, Organisation: "org-operator"}
		seen := pass(t, o, &llm.PassRequest{Path: "/v1/embeddings", Key: "caller-key",
			Header: http.Header{"Openai-Organization": {"org-caller"}}})
		if seen.Header.Get("OpenAI-Organization") != "org-caller" || seen.Header.Get("Authorization") != "Bearer caller-key" {
			t.Fatalf("header = %+v: the caller's key bills the caller, who names its organisation", seen.Header)
		}
		seen = pass(t, o, &llm.PassRequest{Path: "/v1/embeddings", Key: "caller-key"})
		if seen.Header.Get("OpenAI-Organization") != "org-operator" {
			t.Fatalf("header = %+v", seen.Header)
		}
	})
}
