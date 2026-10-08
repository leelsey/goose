// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"context"
	"encoding/base64"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

// Gemini answers a bad key, and a region or an account it will not serve, with
// the 400 it gives a malformed request. Only the last fails the same on every
// backend, and the name of the failure is kept rather than the status again.
func TestAKeyOrARegionRefusedIsNotAMalformedRequest(t *testing.T) {
	info := func(reason string) string {
		return `,"details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"` + reason + `","domain":"googleapis.com"}]`
	}
	for _, tc := range []struct {
		name, status, message, extra string
		want                         llm.ErrorKind
	}{
		{"invalid key", "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", info("API_KEY_INVALID"), llm.KindAuth},
		{"expired key", "INVALID_ARGUMENT", "API key expired. Please renew the API key.", info("API_KEY_EXPIRED"), llm.KindAuth},
		{"region", "FAILED_PRECONDITION", "User location is not supported for the API use.", "", llm.KindPermission},
		{"billing", "FAILED_PRECONDITION", "Gemini API free tier is not available in your country. Please enable billing on your project in Google AI Studio.", "", llm.KindPermission},
		// The same status names a request that is not ready yet, which is the
		// caller's to fix and no fault of the account.
		{"file not ready", "FAILED_PRECONDITION", "The File abc123 is not in an ACTIVE state and usage is not allowed.", "", llm.KindInvalidRequest},
		{"malformed", "INVALID_ARGUMENT", "Invalid JSON payload received.", "", llm.KindInvalidRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"code":400,"message":"` + tc.message + `","status":"` + tc.status + `"` + tc.extra + `}}`))
			})
			_, err := p.Complete(context.Background(), attempt(hello()))
			if e := llm.AsError(err); e.Kind != tc.want || e.Code != tc.status || e.Message != tc.message {
				t.Fatalf("unary: err = %+v, want %s", e, tc.want)
			}
			_, err = p.Stream(context.Background(), attempt(hello()))
			if e := llm.AsError(err); e.Kind != tc.want {
				t.Fatalf("stream: err = %+v, want %s", e, tc.want)
			}
		})
	}
}

// A failure reported inside a 200 is the upstream's words as much as a
// refusal's body is, and the ledger is kept free of both alike.
func TestAFailureInsideA200IsRedactedLikeARefusal(t *testing.T) {
	const failure = `{"error":{"code":500,"message":"SECRET upstream words quoting the prompt","status":"INTERNAL"}}`
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.RawQuery, "alt=sse") {
			_, _ = w.Write([]byte("data: " + failure + "\n\n"))
			return
		}
		_, _ = w.Write([]byte(failure))
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

// An image model sends each image as one line of base64, commonly past a
// megabyte; the same answer read in one piece is served.
func TestAStreamedImageFitsTheDefaultFrameCeiling(t *testing.T) {
	const size = 2 << 20
	image := base64.StdEncoding.EncodeToString(make([]byte, size))
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`data: {"candidates":[{"index":0,"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"` + image + `"}}]},"finishReason":"STOP"}]}` + "\n\n"))
	})
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	got := 0
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if img, ok := part.(*llm.Image); ok {
				got += len(img.Data)
			}
		}
	}
	if s.Err() != nil || got != size {
		t.Fatalf("image bytes = %d, err = %v", got, s.Err())
	}
}
