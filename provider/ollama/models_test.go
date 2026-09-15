// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollama

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestModelsReadsTheTagList(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("asked for %s", r.URL.Path)
		}
		fmt.Fprint(w, `{"models":[{"name":"llama3.1:8b","modified_at":"2024-05-01T00:00:00Z"},{"model":"qwen:7b"}]}`)
	}))
	defer srv.Close()

	p, err := New(Options{ID: "up", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "llama3.1:8b" || models[1].ID != "qwen:7b" {
		t.Fatalf("models = %+v", models)
	}
	if models[0].Created.IsZero() {
		t.Fatal("the modification time was dropped")
	}
}

func TestModelsReportsAMalformedList(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `not json`)
	}))
	defer srv.Close()

	p, err := New(Options{ID: "up", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Models(context.Background()); err == nil {
		t.Fatal("a list that cannot be read is not an empty list")
	}
}

func TestProbeReportsAnUnreachableBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	p, err := New(Options{ID: "up", BaseURL: srv.URL, MaxRetries: 0})
	if err != nil {
		t.Fatal(err)
	}
	srv.Close()
	if err := p.Probe(context.Background()); err == nil {
		t.Fatal("a backend that is not listening is not ready")
	}
}
