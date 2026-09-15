// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func awkwardResponse() *llm.Response {
	return &llm.Response{
		ID: "chatcmpl-x", Model: "served", Provider: "stub",
		Choices: []llm.Choice{{
			Index: 0,
			Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
				&llm.Refusal{Text: "I can't do that"},
				&llm.ToolUse{ID: "call_1", Name: "search", Args: json.RawMessage(`{"q":"x"}`)},
			}},
			Stop: llm.StopToolUse,
		}},
		Usage: llm.Usage{Prompt: 40, Completion: 9, Reported: true, Turns: 1},
	}
}

func awkwardStream() *llm.Stream {
	return wirecore.Synthesise(awkwardResponse())
}

func TestFourPermutationsKeepEverything(t *testing.T) {
	c := New(Options{})
	ctx := context.Background()

	t.Run("unary client, unary upstream", func(t *testing.T) {
		out, _, err := c.EncodeResponse(awkwardResponse(), wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		assertUnary(t, out.Body)
	})

	t.Run("unary client, streaming upstream", func(t *testing.T) {
		resp, err := wirecore.Fold(awkwardStream(), wirecore.FoldOptions{})
		if err != nil {
			t.Fatal(err)
		}
		out, _, err := c.EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		assertUnary(t, out.Body)
	})

	t.Run("streaming client, unary upstream", func(t *testing.T) {
		w := &wire.BufferWriter{}
		if _, err := c.EncodeStream(ctx, wirecore.Synthesise(awkwardResponse()), wire.Options{Stream: true, StreamUsage: true}, w); err != nil {
			t.Fatal(err)
		}
		assertStream(t, w.Body.String())
	})

	t.Run("streaming client, streaming upstream", func(t *testing.T) {
		w := &wire.BufferWriter{}
		if _, err := c.EncodeStream(ctx, awkwardStream(), wire.Options{Stream: true, StreamUsage: true}, w); err != nil {
			t.Fatal(err)
		}
		assertStream(t, w.Body.String())
	})
}

func assertUnary(t *testing.T, body []byte) {
	t.Helper()
	s := string(body)
	if !strings.Contains(s, `"refusal":"I can't do that"`) {
		t.Fatalf("refusal lost: %s", s)
	}
	if !strings.Contains(s, `"call_1"`) || !strings.Contains(s, `"arguments":"{\"q\":\"x\"}"`) {
		t.Fatalf("tool call lost: %s", s)
	}
	if !strings.Contains(s, `"finish_reason":"tool_calls"`) {
		t.Fatalf("stop reason lost: %s", s)
	}
	if !strings.Contains(s, `"completion_tokens":9`) {
		t.Fatalf("usage lost: %s", s)
	}
}

func assertStream(t *testing.T, body string) {
	t.Helper()
	if !strings.Contains(body, `"refusal":"I can't do that"`) {
		t.Fatalf("refusal lost: %s", body)
	}
	if !strings.Contains(body, `"call_1"`) {
		t.Fatalf("tool call lost: %s", body)
	}
	if !strings.Contains(body, `"index":0`) {
		t.Fatalf("tool call index lost: %s", body)
	}
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Fatalf("stop reason lost: %s", body)
	}
	if !strings.Contains(body, `"completion_tokens":9`) {
		t.Fatalf("usage lost: %s", body)
	}
	if !strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Fatalf("sentinel missing: %s", body)
	}
}

func TestRequestRoundTripKeepsEverything(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"system","content":"be brief"},
		{"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}}]},
		{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"c1","content":"done"}
	],"max_completion_tokens":64,"temperature":0.5,"stop":["END"],"stream":true,
	"stream_options":{"include_usage":true},"reasoning_effort":"high","logit_bias":{"7":-1}}`

	req, opts, _, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 4 {
		t.Fatalf("messages = %d", len(req.Messages))
	}
	if *req.MaxOutputTokens != 64 || *req.Temperature != 0.5 || req.Stop[0] != "END" {
		t.Fatalf("request = %+v", req)
	}
	if req.Effort != llm.EffortHigh {
		t.Fatalf("effort = %q", req.Effort)
	}
	if string(req.Extra["logit_bias"]) != `{"7":-1}` {
		t.Fatalf("extra = %v", req.Extra)
	}
	if !opts.Stream || !opts.StreamUsage {
		t.Fatalf("options = %+v", opts)
	}
	clone := req.Clone()
	clone.Messages[1].Content[0].(*llm.Text).Text = "changed"
	if llm.TextOf(req.Messages[1].Content) != "hello" {
		t.Fatal("the per-attempt copy wrote through to the caller's request")
	}
}
