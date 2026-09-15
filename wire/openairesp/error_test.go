// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func TestEncodeErrorSpeaksThisWire(t *testing.T) {
	cases := []struct {
		name   string
		err    *llm.Error
		status int
	}{
		{"rate limit", &llm.Error{Kind: llm.KindRateLimit, Status: 429, Message: "quota exhausted", Code: "quota"}, 429},
		{"auth", &llm.Error{Kind: llm.KindAuth, Status: 401, Message: "bad key"}, 401},
		{"invalid", &llm.Error{Kind: llm.KindInvalidRequest, Message: "bad request"}, 400},
		{"overloaded", &llm.Error{Kind: llm.KindOverloaded, Status: 503, Message: "busy"}, 503},
		{"loss", &llm.Error{Kind: llm.KindLoss, Message: "translation would lose content"}, 400},
		{"unsupported", &llm.Error{Kind: llm.KindUnsupported, Message: "no backend can serve it"}, 422},
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
					Message string `json:"message"`
					Type    string `json:"type"`
					Code    string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal(out.Body, &body); err != nil {
				t.Fatalf("body = %s: %v", out.Body, err)
			}
			if body.Error.Message != c.err.Message {
				t.Fatalf("message = %q, want %q", body.Error.Message, c.err.Message)
			}
			if body.Error.Type == "" {
				t.Fatalf("body = %s: a client switches on the type", out.Body)
			}
			if c.err.Code != "" && body.Error.Code != c.err.Code {
				t.Fatalf("code = %q, want %q", body.Error.Code, c.err.Code)
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
