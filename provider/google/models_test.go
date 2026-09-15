// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestModelsFollowsAndEscapesTheToken(t *testing.T) {
	var second string
	page := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		page++
		if page == 1 {
			fmt.Fprint(w, `{"models":[{"name":"models/gemini-a","displayName":"A"}],"nextPageToken":"t&pageSize=1"}`)
			return
		}
		second = r.URL.Query().Get("pageToken")
		fmt.Fprint(w, `{"models":[{"name":"models/gemini-b"}]}`)
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
	if len(models) != 2 || models[0].ID != "gemini-a" || models[1].ID != "gemini-b" {
		t.Fatalf("models = %+v", models)
	}
	if second != "t&pageSize=1" {
		t.Fatalf("pageToken = %q: the token was read as more than one parameter", second)
	}
}

func TestProbeReportsARefusedCredential(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		fmt.Fprint(w, `{"error":{"code":403,"message":"permission denied","status":"PERMISSION_DENIED"}}`)
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
