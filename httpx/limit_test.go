// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/openaichat"
)

// With no key of the proxy's own, a credential is only something a caller
// wrote: the limiter runs before any backend judges it, and one that
// authenticates with the operator's key never does. Writing a new one each
// request must not buy a new allowance.
func TestAnUnprovenCredentialBuysNoFreshAllowance(t *testing.T) {
	srv := server(t, &stub{}, Options{MaxPerMinute: 1})
	headers := []string{"Authorization", "X-Api-Key", "X-Goog-Api-Key"}
	var last int
	for i := range 6 {
		name, value := headers[i%len(headers)], "made-up-"+strconv.Itoa(i)
		if name == "Authorization" {
			value = "Bearer " + value
		}
		resp := post(t, srv, "/v1/chat/completions?key=q"+strconv.Itoa(i),
			`{"model":"m","messages":[{"role":"user","content":"hi"}]}`, http.Header{name: {value}})
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: a credential nobody has proven is not a caller of its own", last)
	}
}

func TestMeteringWithNoKeyIsByAddressWhateverTheCredential(t *testing.T) {
	c, err := goose.New(goose.WithProvider(&stub{}))
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{
		Codecs:          []wire.Codec{openaichat.New(openaichat.Options{})},
		ForwardedHeader: "X-Forwarded-For",
	})
	if err != nil {
		t.Fatal(err)
	}
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.RemoteAddr = "192.0.2.1:1234"
	r.Header.Set("Authorization", "Bearer tenant-key")
	r.Header.Set("X-Forwarded-For", "203.0.113.7")
	if got := s.bucket(r); got != "addr:203.0.113.7" {
		t.Fatalf("bucket = %q, want the address the front proxy saw", got)
	}
}

// HAProxy's option forwardfor appends a line of its own rather than adding to
// the caller's, and the lines of a header are one list, so the entry the
// nearest hop added is the last of the last line.
func TestAForwardedLineOfTheCallersOwnBuysNoFreshAllowance(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.RemoteAddr = "192.0.2.1:1234"
	r.Header.Add("X-Forwarded-For", "10.9.9.1")
	r.Header.Add("X-Forwarded-For", "203.0.113.7")
	if got := addr(r, "X-Forwarded-For"); got != "addr:203.0.113.7" {
		t.Fatalf("addr = %q: the first line is the one the caller wrote", got)
	}

	srv := server(t, &stub{}, Options{MaxPerMinute: 1, ForwardedHeader: "X-Forwarded-For"})
	var last int
	for i := range 4 {
		header := http.Header{"X-Forwarded-For": {"10.9.9." + strconv.Itoa(i), "203.0.113.7"}}
		resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, header)
		resp.Body.Close()
		last = resp.StatusCode
	}
	if last != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429: the caller wrote the first line, not the last", last)
	}
}

// A handler whose answer goes on waiting after its work is done gives its
// slot back, once however often it says so, and the next request is served
// rather than refused.
func TestASlotGivenBackServesTheNextRequest(t *testing.T) {
	entered, hold := make(chan struct{}), make(chan struct{})
	defer close(hold)
	listening := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		httpcore.Release(r.Context())
		httpcore.Release(r.Context())
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		close(entered)
		<-hold
	})
	srv := server(t, &stub{}, Options{MaxConcurrent: 1, MCP: listening})

	go func() {
		resp := post(t, srv, "/mcp", `{}`, nil)
		resp.Body.Close()
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("giving the slot back twice took more than the one slot there was")
	}
	resp := post(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, nil)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: a request that only waits held the slot the work needed", resp.StatusCode)
	}
}
