// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"net/http"
	"testing"
)

func TestUpstreamReadsEveryVendorShape(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
		code string
	}{
		{"openai", `{"error":{"message":"You exceeded your quota","type":"insufficient_quota","code":"quota"}}`, "You exceeded your quota", "quota"},
		{"anthropic", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`, "Overloaded", "overloaded_error"},
		{"google", `{"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}`, "Resource exhausted", "429"},
		{"ollama", `{"error":"model 'x' not found"}`, "model 'x' not found", ""},
		{"aws", `{"__type":"com.amazon#ThrottlingException","message":"Rate exceeded"}`, "Rate exceeded", "ThrottlingException"},
		{"cloudflare", `{"success":false,"errors":[{"code":7003,"message":"Could not route"}]}`, "Could not route", "7003"},
		{"fastapi", `{"detail":"field required"}`, "field required", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, code := Upstream([]byte(c.body))
			if got != c.want || code != c.code {
				t.Fatalf("got %q/%q, want %q/%q", got, code, c.want, c.code)
			}
		})
	}
}

func TestUpstreamYieldsNothingForAnUnreadableBody(t *testing.T) {
	for _, body := range []string{
		"<!DOCTYPE HTML><html><body>Error code: 501</body></html>",
		"upstream connect error",
		"",
		"{not json",
	} {
		if got, _ := Upstream([]byte(body)); got != "" {
			t.Fatalf("Upstream(%q) = %q, want empty", body, got)
		}
	}
}

func TestUpstreamBoundsWhatAVendorCanSay(t *testing.T) {
	long := make([]byte, 8<<10)
	for i := range long {
		long[i] = 'x'
	}
	got, _ := Upstream([]byte(`{"error":{"message":"` + string(long) + `"}}`))
	if len(got) > maxMessage+4 {
		t.Fatalf("message = %d bytes: a stack trace must not become the client's error", len(got))
	}
}

func TestForwardHeaderKeepsTheConnectionToItself(t *testing.T) {
	src := http.Header{}
	src.Set("Content-Type", "application/json")
	src.Set("Transfer-Encoding", "chunked")
	src.Set("Upgrade", "websocket")
	src.Set("Proxy-Authorization", "Basic c2VjcmV0")
	src.Set("Te", "trailers")
	src.Set("Keep-Alive", "timeout=5")
	src.Set("Authorization", "Bearer the-caller-key")
	src.Set("X-Trace", "keep-me")
	src.Set("Connection", "X-Hop-Scoped, Upgrade")
	src.Set("X-Hop-Scoped", "gone")

	dst := http.Header{}
	ForwardHeader(dst, src, "Authorization")

	for _, gone := range []string{
		"Transfer-Encoding", "Upgrade", "Proxy-Authorization", "Te",
		"Keep-Alive", "Connection", "X-Hop-Scoped", "Authorization",
	} {
		if dst.Get(gone) != "" {
			t.Errorf("%s travelled: %q", gone, dst.Get(gone))
		}
	}
	if dst.Get("Content-Type") != "application/json" || dst.Get("X-Trace") != "keep-me" {
		t.Fatalf("header = %+v: what belongs to the message still travels", dst)
	}
}
