// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

// Azure with Entra and Vertex both authenticate by signing rather than by a
// static key, and a health check that skips the signature reports a working
// backend as broken.
func TestTheHealthCheckCarriesTheSignature(t *testing.T) {
	for _, tc := range []struct {
		name string
		call func(*Provider) error
	}{
		{"models", func(p *Provider) error { _, err := p.Models(context.Background()); return err }},
		{"probe", func(p *Provider) error { return p.Probe(context.Background()) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var seen string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seen = r.Header.Get("Authorization")
				if seen == "" {
					w.WriteHeader(http.StatusUnauthorized)
					fmt.Fprint(w, `{"error":{"message":"no credential"}}`)
					return
				}
				fmt.Fprint(w, `{"data":[{"id":"m"}]}`)
			}))
			defer srv.Close()

			p, err := New(Options{ID: "up", BaseURL: srv.URL, MaxRetries: 0,
				Sign: func(_ context.Context, r *http.Request, _ []byte) error {
					r.Header.Set("Authorization", "Bearer minted")
					return nil
				}})
			if err != nil {
				t.Fatal(err)
			}
			if err := tc.call(p); err != nil {
				t.Fatalf("a signed health check was refused: %v", err)
			}
			if seen != "Bearer minted" {
				t.Fatalf("Authorization = %q, want the minted token", seen)
			}
		})
	}
}
