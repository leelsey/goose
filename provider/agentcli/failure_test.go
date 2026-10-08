// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

// A child stopped because the call ran out of time or its caller went did not
// fail, and how it exited says nothing about the binary. Read as the binary's
// own failure it would be a 502 rather than a 504, a cancelled call booked as
// an error, and a caller's deadline counted against the backend.
func TestAChildStoppedForTheCallSaysWhy(t *testing.T) {
	for _, tc := range []struct {
		name    string
		timeout time.Duration
		ctx     func() (context.Context, context.CancelFunc)
		kind    llm.ErrorKind
		cause   error
		says    string
	}{
		{"the caller's deadline", 0, func() (context.Context, context.CancelFunc) {
			return context.WithTimeout(context.Background(), 300*time.Millisecond)
		}, llm.KindTimeout, context.DeadlineExceeded, ""},
		{"the caller going", 0, func() (context.Context, context.CancelFunc) {
			ctx, cancel := context.WithCancel(context.Background())
			time.AfterFunc(300*time.Millisecond, cancel)
			return ctx, cancel
		}, llm.KindCancelled, context.Canceled, ""},
		{"this provider's timeout", 300 * time.Millisecond, func() (context.Context, context.CancelFunc) {
			return context.WithCancel(context.Background())
		}, llm.KindTimeout, context.DeadlineExceeded, "timeout of 300ms"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := provider(t, Claude, "fake-slow.sh", func(o *Options) { o.Timeout = tc.timeout })
			ctx, cancel := tc.ctx()
			defer cancel()
			s, err := p.Stream(ctx, attempt(hello()))
			if err != nil {
				t.Fatal(err)
			}
			_, err = wirecore.Fold(s, wirecore.FoldOptions{})
			e := llm.AsError(err)
			if e == nil || e.Kind != tc.kind || !errors.Is(err, tc.cause) {
				t.Fatalf("err = %+v: the child was stopped, it did not fail", e)
			}
			if !strings.Contains(e.Message, tc.says) {
				t.Fatalf("message = %q: a deadline this provider set is not the caller's", e.Message)
			}
		})
	}
}

// An agent binary's error is often the backend's refusal quoted whole, and the
// ledger is kept free of it as it is of a refusal's body. goose's own words for
// a failure that came with none are kept.
func TestAnAgentsErrorIsRedactedLikeARefusal(t *testing.T) {
	const secret = "SECRET upstream words quoting the prompt"
	p := provider(t, Claude, "fake-claude.sh", nil)
	for _, tc := range []struct {
		name  string
		parse func(*Provider, []byte, *state) ([]*llm.Frame, error)
		line  string
		own   bool
	}{
		{name: "claude", parse: claudeParse,
			line: `{"type":"result","subtype":"success","is_error":true,"api_error_status":400,"result":"` + secret + `","usage":{"input_tokens":3}}`},
		{name: "codex turn", parse: codexParse, line: `{"type":"turn.failed","error":{"message":"` + secret + `"}}`},
		{name: "codex item", parse: codexParse, line: `{"type":"item.completed","item":{"id":"i","type":"error","message":"` + secret + `"}}`},
		{name: "antigravity", parse: antigravityParse, line: `{"type":"error","error":"` + secret + `"}`},
		{name: "copilot", parse: copilotParse, line: `{"type":"error","error":{"message":"` + secret + `"}}`},
		{name: "no words", parse: codexParse, line: `{"type":"turn.failed"}`, own: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frames, err := tc.parse(p, []byte(tc.line), &state{})
			if err != nil {
				t.Fatal(err)
			}
			var e *llm.Error
			for _, f := range frames {
				if f.Err != nil {
					e = f.Err
				}
			}
			if e == nil {
				t.Fatalf("frames = %+v: the failure went missing", frames)
			}
			if tc.own {
				if got := e.Reason(true); got != e.Message || got == "" {
					t.Fatalf("redacted reason = %q: goose's own words hold nobody's request", got)
				}
				return
			}
			if !strings.Contains(e.Message, "SECRET") {
				t.Fatalf("err = %+v", e)
			}
			if got := e.Reason(true); got != string(e.Kind) {
				t.Fatalf("redacted reason = %q: the upstream's words reached the ledger", got)
			}
			if e.Detail(true) != "" || !strings.Contains(e.Detail(false), "SECRET") {
				t.Fatalf("detail = %q / %q", e.Detail(true), e.Detail(false))
			}
		})
	}
}
