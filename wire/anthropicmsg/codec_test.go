// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func decode(t *testing.T, body string) (*llm.Request, wire.Options, llm.Diagnostics) {
	t.Helper()
	req, opts, diags, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Body: []byte(body)})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return req, opts, diags
}

func TestDecodeSystemToolsAndThinking(t *testing.T) {
	body := `{"model":"claude","max_tokens":1024,
		"system":[{"type":"text","text":"be brief","cache_control":{"type":"ephemeral","ttl":"1h"}}],
		"messages":[{"role":"user","content":"hi","cache_control":{"type":"ephemeral","ttl":"5m"}}],
		"tools":[{"name":"search","description":"d","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"tool","name":"search"},
		"thinking":{"type":"enabled","budget_tokens":4096},
		"stop_sequences":["END"],"top_k":40}`
	r, _, _ := decode(t, body)
	if llm.TextOf(r.System) != "be brief" {
		t.Fatalf("system = %q", llm.TextOf(r.System))
	}
	if r.SystemCache == nil || r.SystemCache.TTL != time.Hour {
		t.Fatalf("system cache = %+v", r.SystemCache)
	}
	if r.Messages[0].Cache == nil || r.Messages[0].Cache.TTL != 5*time.Minute {
		t.Fatalf("message cache = %+v", r.Messages[0].Cache)
	}
	if len(r.Tools) != 1 || r.Tools[0].Name != "search" {
		t.Fatalf("tools = %+v", r.Tools)
	}
	if r.ToolChoice.Mode != llm.ToolSpecific || r.ToolChoice.Name != "search" {
		t.Fatalf("tool choice = %+v", r.ToolChoice)
	}
	if r.Thinking == nil || !r.Thinking.Enabled || r.Thinking.Budget != 4096 {
		t.Fatalf("thinking = %+v", r.Thinking)
	}
	if *r.MaxOutputTokens != 1024 || r.Stop[0] != "END" || *r.TopK != 40 {
		t.Fatalf("request = %+v", r)
	}
}

func TestDecodeKeepsThinkingBlocksAndSignatures(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"assistant","content":[
		{"type":"thinking","thinking":"one","signature":"sig-1"},
		{"type":"tool_use","id":"t1","name":"f","input":{"a":1}},
		{"type":"thinking","thinking":"two","signature":"sig-2"},
		{"type":"redacted_thinking","data":"opaque"}
	]}]}`
	r, _, _ := decode(t, body)
	parts := r.Messages[0].Content
	if len(parts) != 4 {
		t.Fatalf("parts = %d", len(parts))
	}
	if parts[0].(*llm.Thinking).Signature != "sig-1" || parts[2].(*llm.Thinking).Signature != "sig-2" {
		t.Fatal("per-block signatures crossed")
	}
	if string(parts[3].(*llm.Thinking).Redacted) != "opaque" {
		t.Fatal("redacted thinking lost")
	}
	if tu := parts[1].(*llm.ToolUse); tu.ID != "t1" || string(tu.Args) != `{"a":1}` {
		t.Fatalf("tool use = %+v", tu)
	}
}

func TestDecodeToolResultAndImage(t *testing.T) {
	body := `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":[
		{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"done"}],"is_error":true},
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGk="}},
		{"type":"document","source":{"type":"url","url":"x"}}
	]}]}`
	r, _, diags := decode(t, body)
	parts := r.Messages[0].Content
	tr := parts[0].(*llm.ToolResult)
	if tr.ID != "t1" || !tr.IsError || llm.TextOf(tr.Content) != "done" {
		t.Fatalf("tool result = %+v", tr)
	}
	if img := parts[1].(*llm.Image); string(img.Data) != "hi" || img.MIME != "image/png" {
		t.Fatalf("image = %+v", img)
	}
	if raw := parts[2].(*llm.Raw); raw.Type != "document" {
		t.Fatalf("document = %+v", raw)
	}
	if len(diags) == 0 {
		t.Fatal("carrying an unknown block must be recorded")
	}
}

func TestEncodeUsageSplitsCacheBackOut(t *testing.T) {
	resp := &llm.Response{
		Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "hi"}}}, Stop: llm.StopEnd}},
		Usage:   llm.Usage{Prompt: 42208, Completion: 480, CacheRead: 9100, CacheWrite: 33096, CacheWrite5m: 33096, Reported: true},
	}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out.Body, &got); err != nil {
		t.Fatal(err)
	}
	u := got["usage"].(map[string]any)
	if u["input_tokens"].(float64) != 12 {
		t.Fatalf("input_tokens = %v, want the uncached remainder 12", u["input_tokens"])
	}
	if u["cache_read_input_tokens"].(float64) != 9100 || u["cache_creation_input_tokens"].(float64) != 33096 {
		t.Fatalf("cache figures = %v", u)
	}
	cc := u["cache_creation"].(map[string]any)
	if cc["ephemeral_5m_input_tokens"].(float64) != 33096 {
		t.Fatalf("ttl split = %v", cc)
	}
	if _, ok := cc["ephemeral_1h_input_tokens"]; ok {
		t.Fatalf("one-hour portion should be absent when the whole write was five-minute: %v", cc)
	}
}

func TestEncodeAlwaysCarriesUsage(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "x"}}}, Stop: llm.StopEnd}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"usage"`) {
		t.Fatalf("body = %s: a missing usage key reads as a backend that said nothing", out.Body)
	}
}

func TestEncodeRefusalBecomesVisibleText(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.Refusal{Text: "I won't"}}},
		Stop:    llm.StopRefusal,
	}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	s := string(out.Body)
	if !strings.Contains(s, `"I won't"`) || !strings.Contains(s, `"stop_reason":"refusal"`) {
		t.Fatalf("body = %s", s)
	}
}

func TestEncodeRefusesToDropCandidates(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{
		{Index: 0, Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "a"}}}, Stop: llm.StopEnd},
		{Index: 1, Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "b"}}}, Stop: llm.StopEnd},
	}}
	_, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if diags.Worst() != llm.SeverityFatal {
		t.Fatalf("diagnostics = %+v: losing a candidate is not a detail", diags)
	}
	if err := (llm.LossPolicy{}).Check(diags); err == nil {
		t.Fatal("the default policy must fail a call that cannot deliver a candidate")
	}
}

type event struct {
	name string
	body map[string]any
}

func stream(t *testing.T, frames ...*llm.Frame) ([]event, *wire.BufferWriter) {
	t.Helper()
	s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "claude"}, frames)
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatalf("encode: %v", err)
	}
	return parseEvents(t, w.Body.String()), w
}

func parseEvents(t *testing.T, body string) []event {
	t.Helper()
	var out []event
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var e event
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				e.name = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &e.body); err != nil {
					t.Fatalf("event body %q: %v", line, err)
				}
			}
		}
		out = append(out, e)
	}
	return out
}

func names(evs []event) []string {
	out := make([]string, len(evs))
	for i, e := range evs {
		out[i] = e.name
	}
	return out
}

func TestStreamSequenceIsBalanced(t *testing.T) {
	evs, _ := stream(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "think"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "sig"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "answer"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 10, Completion: 4, Reported: true}, TurnEnd: true},
	)
	got := names(evs)
	want := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_delta",
		"content_block_stop",
		"content_block_start", "content_block_delta", "content_block_stop",
		"message_delta", "message_stop",
	}
	if len(got) != len(want) {
		t.Fatalf("events = %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("events = %v, want %v", got, want)
		}
	}
	depth, maxIndex := 0, -1
	for _, e := range evs {
		switch e.name {
		case "content_block_start":
			depth++
			i := int(e.body["index"].(float64))
			if i != maxIndex+1 {
				t.Fatalf("block index %d after %d: indices increase monotonically", i, maxIndex)
			}
			maxIndex = i
		case "content_block_stop":
			depth--
		}
		if depth < 0 || depth > 1 {
			t.Fatalf("unbalanced blocks at %s", e.name)
		}
	}
	if depth != 0 {
		t.Fatal("a content block was left open")
	}
}

func TestStreamHoldsTerminalUsage(t *testing.T) {
	evs, _ := stream(t,
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 900, Completion: 12, CacheRead: 800, Reported: true}, TurnEnd: true},
	)
	last := evs[len(evs)-1]
	if last.name != "message_stop" {
		t.Fatalf("last event = %s", last.name)
	}
	md := evs[len(evs)-2]
	if md.name != "message_delta" {
		t.Fatalf("penultimate event = %s", md.name)
	}
	u := md.body["usage"].(map[string]any)
	if u["output_tokens"].(float64) != 12 || u["cache_read_input_tokens"].(float64) != 800 {
		t.Fatalf("terminal usage = %v: it must be the complete figure, not the one known at the finish", u)
	}
	if u["input_tokens"].(float64) != 100 {
		t.Fatalf("input_tokens = %v, want the uncached remainder", u["input_tokens"])
	}
}

func TestStreamEmitsToolCallsWithCompleteArguments(t *testing.T) {
	i0, i1 := 0, 1
	evs, _ := stream(t,
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i0, ID: "a", Name: "one", Args: json.RawMessage(`{"q":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i1, ID: "b", Name: "two", Args: json.RawMessage(`{"u":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Args: json.RawMessage(`1}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i1, Args: json.RawMessage(`2}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	var args []string
	var ids []string
	for _, e := range evs {
		if e.name == "content_block_start" {
			if cb, ok := e.body["content_block"].(map[string]any); ok && cb["type"] == "tool_use" {
				ids = append(ids, cb["id"].(string))
			}
		}
		if e.name == "content_block_delta" {
			if d, ok := e.body["delta"].(map[string]any); ok && d["type"] == "input_json_delta" {
				args = append(args, d["partial_json"].(string))
			}
		}
	}
	if len(ids) != 2 || ids[0] != "a" || ids[1] != "b" {
		t.Fatalf("tool ids = %v", ids)
	}
	if len(args) != 2 || args[0] != `{"q":1}` || args[1] != `{"u":2}` {
		t.Fatalf("arguments = %v: interleaved fragments must not cross calls", args)
	}
}

func TestStreamReportsFailureAsAnErrorEvent(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half"}}},
		{Err: &llm.Error{Kind: llm.KindOverloaded, Message: "overloaded"}},
	})
	w := &wire.BufferWriter{}
	diags, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if err == nil {
		t.Fatal("the failure must reach the caller")
	}
	evs := parseEvents(t, w.Body.String())
	var sawError, sawStop bool
	for _, e := range evs {
		if e.name == "error" {
			sawError = true
		}
		if e.name == "message_stop" {
			sawStop = true
		}
	}
	if !sawError || !sawStop {
		t.Fatalf("events = %v: an in-band failure still closes the message", names(evs))
	}
	if diags.Worst() < llm.SeverityLoss {
		t.Fatalf("diagnostics = %+v", diags)
	}
}

func TestTruncatedStreamSendsAnErrorEvent(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "msg_1", Model: "claude"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	})
	w := &wire.BufferWriter{}
	_, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if e := llm.AsError(err); e == nil || e.Code != llm.WireNoFinish {
		t.Fatalf("err = %v, want %s", err, llm.WireNoFinish)
	}
	body := w.Body.String()
	if !strings.Contains(body, "event: error") {
		t.Fatalf("body = %s: the truncation must reach the client", body)
	}
}

func TestAnImageURLKeepsItsOwnSpelling(t *testing.T) {
	blocks, diags := EncodeBlocks([]llm.Part{
		&llm.Image{URL: "https://leelsey.com/a.png"},
		&llm.Image{MIME: "image/png", Data: []byte{0, 1, 2}},
		&llm.Image{},
	})
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d: %s", len(blocks), blocks)
	}
	if !strings.Contains(string(blocks[0]), `"type":"url"`) ||
		!strings.Contains(string(blocks[0]), "https://leelsey.com/a.png") {
		t.Errorf("url image = %s", blocks[0])
	}
	if !strings.Contains(string(blocks[1]), `"type":"base64"`) {
		t.Errorf("byte image = %s", blocks[1])
	}
	if !diags.Has(llm.DropImage) {
		t.Errorf("diagnostics = %v", diags.Codes())
	}
}

func TestACacheBreakpointIsNotExtendedPastWhereItWasAsked(t *testing.T) {
	body := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[
		{"type":"text","text":"stable","cache_control":{"type":"ephemeral"}},
		{"type":"text","text":"volatile"}]}]}`
	req, _, diags, err := New(Options{}).Decode(&wire.Inbound{Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if req.Messages[0].Cache != nil {
		t.Error("a breakpoint on an earlier block was extended to the whole turn")
	}
	if !diags.Has(llm.DropCacheHint) {
		t.Errorf("diagnostics = %v", diags.Codes())
	}

	last := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[
		{"type":"text","text":"stable"},
		{"type":"text","text":"volatile","cache_control":{"type":"ephemeral"}}]}]}`
	req, _, diags, err = New(Options{}).Decode(&wire.Inbound{Body: []byte(last)})
	if err != nil {
		t.Fatal(err)
	}
	if req.Messages[0].Cache == nil {
		t.Error("a breakpoint on the last block was declined")
	}
	if diags.Has(llm.DropCacheHint) {
		t.Errorf("diagnostics = %v", diags.Codes())
	}
}

// A redacted thinking block is opaque bytes the client hands back next turn, so
// what comes out must be what went in whichever backend produced it.
func TestRedactedThinkingSurvivesEveryBackend(t *testing.T) {
	// The token is opaque: unpadded, URL-safe or not base64 at all, it must come
	// back exactly as it went in.
	for _, token := range []string{
		"EroBCoYBGAIiQKx3ab9xQw==", "EroBCoYBGAIiQKx3ab9xQw", "a-b_cd==", "YWJ=", "opaque-token!",
	} {
		in := json.RawMessage(`[{"type":"redacted_thinking","data":"` + token + `"}]`)
		parts, _, _, err := DecodeBlocks(in)
		if err != nil {
			t.Fatal(err)
		}
		back, _ := EncodeBlocks(parts)
		if len(back) != 1 {
			t.Fatalf("%q: blocks = %d", token, len(back))
		}
		var out struct {
			Data string `json:"data"`
		}
		if err := json.Unmarshal(back[0], &out); err != nil {
			t.Fatal(err)
		}
		if out.Data != token {
			t.Errorf("token %q came back as %q", token, out.Data)
		}
	}

	raw := []byte("EroBCoYBGAIiQKx3ab9xQw==")
	blocks, _ := EncodeBlocks([]llm.Part{&llm.Thinking{Redacted: raw}})
	if len(blocks) != 1 {
		t.Fatalf("blocks = %d", len(blocks))
	}
	var out struct {
		Type string `json:"type"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal(blocks[0], &out); err != nil {
		t.Fatal(err)
	}
	if out.Type != "redacted_thinking" {
		t.Fatalf("type = %q", out.Type)
	}
	if out.Data != string(raw) {
		t.Fatalf("data = %q, want %q", out.Data, raw)
	}
}
