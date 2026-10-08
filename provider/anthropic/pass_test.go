// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// Vertex mints a token rather than presenting a key, so a passthrough that
// only knows about x-api-key forwards an unauthenticated request.
func TestPassthroughCarriesTheSignature(t *testing.T) {
	var auth, key string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, key = r.Header.Get("Authorization"), r.Header.Get("x-api-key")
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
		Method: http.MethodPost, Path: "/v1/messages", Header: http.Header{},
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
		t.Errorf("x-api-key = %q, want nothing: a signed backend mints its own credential", key)
	}
}

// A gateway that admits a routed call on a configured header admits an
// unrouted one on the same, and the configured betas are asked for on both. A
// key in the query is the gate's or another vendor's, and goes no further.
func TestAPassthroughCarriesWhatTheOperatorConfigured(t *testing.T) {
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	p, err := New(Options{BaseURL: srv.URL, APIKey: "operator-key", Betas: []string{"b1", "b2"},
		Header: http.Header{"X-Gateway-Auth": {"gw-secret"}}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Pass(context.Background(), &llm.PassRequest{
		Method: http.MethodPost, Path: "/v1/messages/count_tokens", Query: "key=GATE&beta=true",
		Header: http.Header{"Anthropic-Beta": {"b2,own"}, "X-Gateway-Auth": {"forged"}},
		Body:   strings.NewReader("{}"),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if got := seen.Header.Values("X-Gateway-Auth"); len(got) != 1 || got[0] != "gw-secret" {
		t.Errorf("X-Gateway-Auth = %q", got)
	}
	if got := strings.Join(seen.Header.Values("anthropic-beta"), ","); got != "b2,own,b1" {
		t.Errorf("anthropic-beta = %q: each beta once, the caller's kept", got)
	}
	if seen.URL.RawQuery != "beta=true" {
		t.Errorf("query = %q", seen.URL.RawQuery)
	}
}
