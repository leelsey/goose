// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

// A model without tools or thinking is refused with the 400 a malformed
// request gets, though another model may well have what this one lacks.
func TestAModelLackingACapabilityIsNotAMalformedRequest(t *testing.T) {
	for _, tc := range []struct {
		message string
		want    llm.ErrorKind
	}{
		{"registry.ollama.ai/library/gemma:2b does not support tools", llm.KindUnsupported},
		{`"llama3.2" does not support thinking`, llm.KindUnsupported},
		{"invalid message format", llm.KindInvalidRequest},
	} {
		body, _ := json.Marshal(map[string]string{"error": tc.message})
		p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write(body)
		})
		_, err := p.Complete(context.Background(), attempt(hello()))
		if e := llm.AsError(err); e.Kind != tc.want || e.Message != tc.message {
			t.Errorf("unary %q: err = %+v, want %s", tc.message, e, tc.want)
		}
		_, err = p.Stream(context.Background(), attempt(hello()))
		if e := llm.AsError(err); e.Kind != tc.want {
			t.Errorf("stream %q: err = %+v, want %s", tc.message, e, tc.want)
		}
	}
}

// A failure reported inside a 200 is the upstream's words as much as a
// refusal's body is, and the ledger is kept free of both alike.
func TestAFailureInsideA200IsRedactedLikeARefusal(t *testing.T) {
	const failure = `{"error":"SECRET upstream words quoting the prompt"}`
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(failure + "\n"))
	})
	_, uerr := p.Complete(context.Background(), attempt(hello()))
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	_, serr := wirecore.Fold(s, wirecore.FoldOptions{})
	for name, err := range map[string]error{"unary": uerr, "stream": serr} {
		e := llm.AsError(err)
		if e == nil || !strings.Contains(e.Message, "SECRET") {
			t.Fatalf("%s: err = %v", name, err)
		}
		if got := e.Reason(true); got != string(e.Kind) {
			t.Fatalf("%s: redacted reason = %q: the upstream's words reached the ledger", name, got)
		}
	}
}

// A gateway that admits a routed call on a configured header admits an
// unrouted one on the same. A key in the query is the gate's, and goes no
// further.
func TestAPassthroughCarriesWhatTheOperatorConfigured(t *testing.T) {
	var seen *http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	p, err := New(Options{BaseURL: srv.URL, Header: http.Header{"X-Gateway-Auth": {"gw-secret"}}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Pass(context.Background(), &llm.PassRequest{
		Method: http.MethodGet, Path: "/api/tags", Query: "key=GATE&verbose=1", Header: http.Header{},
	})
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if seen.URL.RawQuery != "verbose=1" {
		t.Errorf("query = %q", seen.URL.RawQuery)
	}
	if got := seen.Header.Get("X-Gateway-Auth"); got != "gw-secret" {
		t.Errorf("X-Gateway-Auth = %q", got)
	}
}
