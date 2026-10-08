// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// Vertex mints a token rather than presenting a key, so a passthrough that
// only knows about x-goog-api-key forwards an unauthenticated request.
func TestPassthroughCarriesTheSignature(t *testing.T) {
	var auth, key string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, key = r.Header.Get("Authorization"), r.Header.Get("x-goog-api-key")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	p, err := New(Options{ID: "vx", BaseURL: srv.URL, APIKey: "static",
		Sign: func(_ context.Context, r *http.Request, _ []byte) error {
			r.Header.Set("Authorization", "Bearer minted")
			return nil
		}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Pass(context.Background(), &llm.PassRequest{
		Method: http.MethodPost, Path: "/v1/models/x:generateContent", Header: http.Header{},
		Body: strings.NewReader("{}"),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if auth != "Bearer minted" {
		t.Errorf("Authorization = %q, want the minted token", auth)
	}
	if key != "" {
		t.Errorf("x-goog-api-key = %q, want nothing: a signed backend mints its own credential", key)
	}
}

// A key in the query is the gate's, or the caller's own, which goes in the
// header when the operator allows it and nowhere when it does not.
func TestAPassthroughSendsNoKeyInTheQuery(t *testing.T) {
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	for _, allow := range []bool{false, true} {
		p, err := New(Options{BaseURL: srv.URL, AllowClientKey: allow,
			Header: http.Header{"X-Gateway-Auth": {"gw-secret"}}})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := p.Pass(context.Background(), &llm.PassRequest{
			Method: http.MethodGet, Path: "/v1beta/files", Query: "key=CALLERS-OWN&pageSize=2",
			Key: "CALLERS-OWN", Header: http.Header{},
		})
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if seen.URL.RawQuery != "pageSize=2" {
			t.Errorf("allow=%v: query = %q", allow, seen.URL.RawQuery)
		}
		want := ""
		if allow {
			want = "CALLERS-OWN"
		}
		if got := seen.Header.Get("x-goog-api-key"); got != want {
			t.Errorf("allow=%v: x-goog-api-key = %q, want %q", allow, got, want)
		}
		if got := seen.Header.Get("X-Gateway-Auth"); got != "gw-secret" {
			t.Errorf("X-Gateway-Auth = %q: the configured headers go with every call", got)
		}
	}
}
