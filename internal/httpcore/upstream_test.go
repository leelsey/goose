// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"unicode/utf8"
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
		{"google", `{"error":{"code":429,"message":"Resource exhausted","status":"RESOURCE_EXHAUSTED"}}`, "Resource exhausted", "RESOURCE_EXHAUSTED"},
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

// A vendor is free to add a field goose does not read, and to give a field
// goose does read a shape it did not expect. Neither is a reason to throw away
// the sentence the client needs.
func TestAnUnexpectedSiblingDoesNotSwallowTheMessage(t *testing.T) {
	for _, tc := range []struct{ body, want string }{
		{`{"error":{"message":"Resource exhausted"},"status":429}`, "Resource exhausted"},
		{`{"error":{"message":"Not found","status":404}}`, "Not found"},
		{`{"error":{"message":"Overloaded","type":503}}`, "Overloaded"},
		{`{"message":"Too Many Requests","errors":["rate limited"]}`, "Too Many Requests"},
		{`{"message":"Bad gateway","errors":{"reason":"x"}}`, "Bad gateway"},
		{`{"error":{"code":7003},"errors":[{"message":"Could not route"}]}`, "Could not route"},
		{`{"error":"plain words"}`, "plain words"},
	} {
		if got, _ := Upstream([]byte(tc.body)); got != tc.want {
			t.Errorf("Upstream(%s) = %q, want %q", tc.body, got, tc.want)
		}
	}
}

// The ceiling is in bytes and the message is in characters.
func TestAClippedMessageIsStillText(t *testing.T) {
	long := strings.Repeat("가", maxMessage)
	got := clip(long)
	if !utf8.ValidString(got) {
		t.Fatalf("clip cut a character in half: %q", got[len(got)-12:])
	}
}

// Gemini sends the status again as a number beside the name of the failure, and
// the name is what tells a bad key from a bad request.
func TestAVendorsNameForAFailureIsKeptOverItsNumber(t *testing.T) {
	for _, tc := range []struct{ body, want string }{
		{`{"error":{"code":400,"message":"API key not valid.","status":"INVALID_ARGUMENT"}}`, "INVALID_ARGUMENT"},
		{`{"error":{"message":"bad grammar","type":"BadRequestError","code":400}}`, "BadRequestError"},
		{`{"error":{"message":"Overloaded","type":503}}`, "503"},
		{`{"message":"no","code":7,"status":"DENIED"}`, "DENIED"},
		{`{"errors":[{"message":"Could not route","code":7003}]}`, "7003"},
	} {
		if _, got := Upstream([]byte(tc.body)); got != tc.want {
			t.Errorf("Upstream(%s) code = %q, want %q", tc.body, got, tc.want)
		}
	}
}

func TestACodeIsReadAsANameOrANumber(t *testing.T) {
	var got struct {
		A, B, C, D Code
	}
	if err := json.Unmarshal([]byte(`{"A":"quota","B":400,"C":null,"D":{"x":1}}`), &got); err != nil {
		t.Fatalf("err = %v: a code of an unexpected shape must not cost the message beside it", err)
	}
	if got.A != "quota" || got.B != "400" || got.C != "" || got.D != "" {
		t.Fatalf("codes = %+v", got)
	}
}
