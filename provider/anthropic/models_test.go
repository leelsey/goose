// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelsFollowsTheCursor(t *testing.T) {
	var asked []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked = append(asked, r.URL.RawQuery)
		switch r.URL.Query().Get("after_id") {
		case "":
			fmt.Fprint(w, `{"data":[{"id":"a","display_name":"A"}],"has_more":true,"last_id":"a"}`)
		case "a":
			fmt.Fprint(w, `{"data":[{"id":"b"}],"has_more":false}`)
		default:
			t.Errorf("unexpected cursor %q", r.URL.Query().Get("after_id"))
		}
	}))
	defer srv.Close()

	p, err := New(Options{ID: "up", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "a" || models[1].ID != "b" {
		t.Fatalf("models = %+v", models)
	}
	if len(asked) != 2 || !strings.Contains(asked[1], "after_id=a") {
		t.Fatalf("queries = %v", asked)
	}
}

func TestModelsEscapesTheCursor(t *testing.T) {
	var second string
	page := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		page++
		if page == 1 {
			fmt.Fprint(w, `{"data":[{"id":"a"}],"has_more":true,"last_id":"x&limit=1"}`)
			return
		}
		second = r.URL.Query().Get("after_id")
		fmt.Fprint(w, `{"data":[],"has_more":false}`)
	}))
	defer srv.Close()

	p, err := New(Options{ID: "up", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Models(context.Background()); err != nil {
		t.Fatal(err)
	}
	if second != "x&limit=1" {
		t.Fatalf("after_id = %q: the cursor was read as more than one parameter", second)
	}
}

func TestProbeReportsAnUnreachableBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}`)
	}))
	defer srv.Close()

	p, err := New(Options{ID: "up", BaseURL: srv.URL, APIKey: "k", MaxRetries: 0})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.Probe(context.Background()); err == nil {
		t.Fatal("a backend that refuses the credential is not ready")
	}
}

// Vertex authenticates by minting a token rather than by a static key, and a
// health check that skips the signature reports a working backend as broken.
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
					fmt.Fprint(w, `{"type":"error","error":{"type":"authentication_error","message":"no credential"}}`)
					return
				}
				fmt.Fprint(w, `{"data":[{"id":"m"}],"has_more":false}`)
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
