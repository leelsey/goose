// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
)

func text(v any) string { return fmt.Sprint(v) }

type Progress struct {
	Done, Total float64
	Message     string
}

type progressKey struct{}

func WithProgress(ctx context.Context, report func(Progress)) context.Context {
	if report == nil {
		return ctx
	}
	return context.WithValue(ctx, progressKey{}, report)
}

func Report(ctx context.Context, p Progress) {
	if report, ok := ctx.Value(progressKey{}).(func(Progress)); ok {
		report(p)
	}
}

func Watched(ctx context.Context) bool {
	_, ok := ctx.Value(progressKey{}).(func(Progress))
	return ok
}

func progressFrame(token any, p Progress) map[string]any {
	params := map[string]any{"progressToken": token, "progress": p.Done}
	if p.Total > 0 {
		params["total"] = p.Total
	}
	if p.Message != "" {
		params["message"] = p.Message
	}
	return map[string]any{"jsonrpc": jsonrpcVersion, "method": "notifications/progress", "params": params}
}

func readProgress(params json.RawMessage, token string) (Progress, bool) {
	var body struct {
		Token    json.RawMessage `json:"progressToken"`
		Progress float64         `json:"progress"`
		Total    float64         `json:"total"`
		Message  string          `json:"message"`
	}
	if json.Unmarshal(params, &body) != nil {
		return Progress{}, false
	}
	var got any
	if json.Unmarshal(body.Token, &got) != nil {
		return Progress{}, false
	}
	if text(got) != token {
		return Progress{}, false
	}
	// A server chooses these words while a tool of its own runs, and they reach
	// whoever asked to watch. They are its text like any other.
	return Progress{Done: body.Progress, Total: body.Total, Message: tame(body.Message)}, true
}

func requestToken(params json.RawMessage) any {
	if len(params) == 0 {
		return nil
	}
	var envelope struct {
		Meta struct {
			Token json.RawMessage `json:"progressToken"`
		} `json:"_meta"`
	}
	if json.Unmarshal(params, &envelope) != nil || len(envelope.Meta.Token) == 0 {
		return nil
	}
	var token any
	if json.Unmarshal(envelope.Meta.Token, &token) != nil {
		return nil
	}
	switch token.(type) {
	case string, float64:
		return token
	default:
		return nil
	}
}
