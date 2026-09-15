// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func TestEncodeErrorSpeaksThisWire(t *testing.T) {
	cases := []struct {
		name   string
		err    *llm.Error
		status int
		state  string
	}{
		{"rate limit", &llm.Error{Kind: llm.KindRateLimit, Status: 429, Message: "quota exhausted"}, 429, "RESOURCE_EXHAUSTED"},
		{"auth", &llm.Error{Kind: llm.KindAuth, Status: 401, Message: "bad key"}, 401, "UNAUTHENTICATED"},
		{"permission", &llm.Error{Kind: llm.KindPermission, Status: 403, Message: "denied"}, 403, "PERMISSION_DENIED"},
		{"not found", &llm.Error{Kind: llm.KindNotFound, Status: 404, Message: "no such model"}, 404, "NOT_FOUND"},
		{"invalid", &llm.Error{Kind: llm.KindInvalidRequest, Message: "bad request"}, 400, "INVALID_ARGUMENT"},
		{"overloaded", &llm.Error{Kind: llm.KindOverloaded, Status: 503, Message: "busy"}, 503, "UNAVAILABLE"},
		{"timeout", &llm.Error{Kind: llm.KindTimeout, Message: "took too long"}, 504, "DEADLINE_EXCEEDED"},
		{"upstream status wins", &llm.Error{Kind: llm.KindUpstream, Status: 502, Message: "bad gateway"}, 502, ""},
	}
	codec := New(Options{})
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := codec.EncodeError(c.err, wire.Options{})
			if out == nil {
				t.Fatal("no error body at all")
			}
			if out.Status != c.status {
				t.Fatalf("status = %d, want %d", out.Status, c.status)
			}
			var body struct {
				Error struct {
					Code    int    `json:"code"`
					Message string `json:"message"`
					Status  string `json:"status"`
				} `json:"error"`
			}
			if err := json.Unmarshal(out.Body, &body); err != nil {
				t.Fatalf("body = %s: %v", out.Body, err)
			}
			if body.Error.Message != c.err.Message {
				t.Fatalf("message = %q, want %q", body.Error.Message, c.err.Message)
			}
			if body.Error.Code != c.status {
				t.Fatalf("code = %d, want %d", body.Error.Code, c.status)
			}
			if c.state != "" && body.Error.Status != c.state {
				t.Fatalf("status = %q, want %q", body.Error.Status, c.state)
			}
			if ct := out.Header.Get("Content-Type"); !strings.Contains(ct, "json") {
				t.Fatalf("content type = %q", ct)
			}
		})
	}
}

func TestEncodeErrorAlwaysSaysSomething(t *testing.T) {
	out := New(Options{}).EncodeError(&llm.Error{Kind: llm.KindInternal}, wire.Options{})
	if out == nil || len(out.Body) == 0 {
		t.Fatal("an error with no message is still an error")
	}
	if out.Status < http.StatusBadRequest {
		t.Fatalf("status = %d", out.Status)
	}
}

func TestAStreamedFailureSpeaksThisWireToo(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "part"}}},
		{Err: llm.Errorf(llm.KindAuth, "credential refused")},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err == nil {
		t.Fatal("a failed stream must be reported")
	}
	var found map[string]any
	for _, line := range strings.Split(w.Body.String(), "\n") {
		raw, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var frame map[string]any
		if err := json.Unmarshal([]byte(raw), &frame); err != nil {
			t.Fatalf("frame %q: %v", raw, err)
		}
		if inner, has := frame["error"].(map[string]any); has {
			found = inner
		}
	}
	if found == nil {
		t.Fatalf("no error frame in %q", w.Body.String())
	}
	if found["status"] != "UNAUTHENTICATED" {
		t.Fatalf("status = %v: this wire names its own", found["status"])
	}
	if found["code"].(float64) != 401 {
		t.Fatalf("code = %v: a failure with no upstream status still has one here", found["code"])
	}
}
