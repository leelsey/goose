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
