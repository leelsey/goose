// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

// An account out of credit, capped or disabled is refused with the 400 a
// malformed request gets, and the request would fare no worse on any other
// backend.
func TestAnAccountOutOfCreditIsNotAMalformedRequest(t *testing.T) {
	for _, tc := range []struct {
		name, message string
		want          llm.ErrorKind
	}{
		{"credit", "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.", llm.KindPermission},
		{"spend cap", "You have reached your specified API usage limits. You will regain access on 2026-10-01 at 00:00 UTC.", llm.KindPermission},
		{"workspace cap", "You have reached your specified workspace API usage limits. You will regain access on 2026-10-01 at 00:00 UTC.", llm.KindPermission},
		{"disabled", "This organization has been disabled.", llm.KindPermission},
		{"malformed", "messages: roles must alternate", llm.KindInvalidRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"` + tc.message + `"}}`))
			})
			_, err := p.Complete(context.Background(), attempt(hello()))
			if e := llm.AsError(err); e.Kind != tc.want || e.Code != "invalid_request_error" {
				t.Fatalf("unary: err = %+v, want %s", e, tc.want)
			}
			_, err = p.Stream(context.Background(), attempt(hello()))
			if e := llm.AsError(err); e.Kind != tc.want {
				t.Fatalf("stream: err = %+v, want %s", e, tc.want)
			}
		})
	}
}

// A stream's error frame is the upstream's words as much as a refusal's body
// is, and the ledger is kept free of both alike.
func TestAStreamErrorIsRedactedLikeARefusal(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("event: message_start\ndata: {\"message\":{\"id\":\"m\"}}\n\n" +
			"event: error\ndata: {\"error\":{\"type\":\"api_error\",\"message\":\"SECRET upstream words quoting the prompt\"}}\n\n"))
	})
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	_, ferr := wirecore.Fold(s, wirecore.FoldOptions{})
	e := llm.AsError(ferr)
	if e == nil || !strings.Contains(e.Message, "SECRET") {
		t.Fatalf("err = %v", ferr)
	}
	if got := e.Reason(true); got != string(e.Kind) {
		t.Fatalf("redacted reason = %q: the upstream's words reached the ledger", got)
	}
	if !strings.Contains(e.Detail(false), "SECRET") {
		t.Fatalf("detail = %q", e.Detail(false))
	}
}
