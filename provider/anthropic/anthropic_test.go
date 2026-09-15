// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

func serve(t *testing.T, handler http.HandlerFunc) (*Provider, *[]string) {
	t.Helper()
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		handler(w, r)
	}))
	t.Cleanup(srv.Close)
	p, err := New(Options{BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	return p, &bodies
}

func attempt(req *llm.Request) *llm.Attempt {
	return &llm.Attempt{Request: req, Model: "claude-test", Recorder: llm.NopRecorder{}}
}

func TestCompleteNormalisesUsage(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":40,
			"cache_read_input_tokens":9100,"cache_creation_input_tokens":33096,
			"cache_creation":{"ephemeral_5m_input_tokens":33096}}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model: "claude-test", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Prompt != 12+9100+33096 {
		t.Fatalf("prompt = %d: the cache figures sit beside input_tokens on this wire", resp.Usage.Prompt)
	}
	if resp.Usage.CacheWrite5m != 33096 || !resp.Usage.Reported {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if llm.TextOf(resp.Choices[0].Message.Content) != "hi" {
		t.Fatalf("content = %+v", resp.Choices[0].Message.Content)
	}
}

func TestRequestCoalescesConsecutiveUserTurns(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	_, err := p.Complete(context.Background(), attempt(&llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "one"}}},
			{Role: llm.RoleTool, Content: []llm.Part{&llm.ToolResult{ID: "t1", Content: []llm.Part{&llm.Text{Text: "result"}}}}},
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}},
		},
	}))
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Messages []struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Messages) != 2 {
		t.Fatalf("turns = %d: consecutive user items must coalesce", len(sent.Messages))
	}
	if len(sent.Messages[0].Content) != 2 {
		t.Fatalf("first turn = %v: the tool result must join the user turn", sent.Messages[0].Content)
	}
}

func TestRequestDropsUnsignedThinkingAndTheTurnItEmpties(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	_, err := p.Complete(context.Background(), attempt(&llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "q"}}},
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "unsigned"}}},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "again"}}},
		},
	}))
	if err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[0]
	if strings.Contains(body, "unsigned") {
		t.Fatalf("body = %s: thinking without a signature cannot be replayed", body)
	}
	var sent struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	_ = json.Unmarshal([]byte(body), &sent)
	if len(sent.Messages) != 1 {
		t.Fatalf("turns = %+v: the emptied assistant turn must be omitted, and the two user turns merge", sent.Messages)
	}
}

func TestCacheBreakpointsAreBoundedAndNeverOnThinking(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	msgs := []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "t", Signature: "s"}}, Cache: &llm.CacheHint{}},
	}
	for i := 0; i < 6; i++ {
		msgs = append(msgs,
			llm.Message{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "u"}}, Cache: &llm.CacheHint{}},
			llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "a"}}},
		)
	}
	_, err := p.Complete(context.Background(), attempt(&llm.Request{
		System: []llm.Part{&llm.Text{Text: "sys"}}, SystemCache: &llm.CacheHint{}, Messages: msgs,
	}))
	if err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[0]
	if n := strings.Count(body, "cache_control"); n > maxBreakpoints {
		t.Fatalf("breakpoints = %d, want at most %d", n, maxBreakpoints)
	}
	var sent struct {
		Messages []struct {
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	_ = json.Unmarshal([]byte(body), &sent)
	for _, m := range sent.Messages {
		for _, block := range m.Content {
			if block["type"] == "thinking" {
				if _, ok := block["cache_control"]; ok {
					t.Fatal("a breakpoint landed on a thinking block, which the API rejects")
				}
			}
		}
	}
}

func TestStreamParsesEventsAndUsage(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range []string{
			`event: message_start` + "\n" + `data: {"message":{"id":"msg_9","model":"claude-test","usage":{"input_tokens":12,"cache_read_input_tokens":900}}}`,
			`event: content_block_start` + "\n" + `data: {"index":0,"content_block":{"type":"thinking"}}`,
			`event: content_block_delta` + "\n" + `data: {"index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}`,
			`event: content_block_delta` + "\n" + `data: {"index":0,"delta":{"type":"signature_delta","signature":"sig"}}`,
			`event: content_block_stop` + "\n" + `data: {"index":0}`,
			`event: content_block_start` + "\n" + `data: {"index":1,"content_block":{"type":"tool_use","id":"tu_1","name":"search"}}`,
			`event: content_block_delta` + "\n" + `data: {"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}`,
			`event: message_delta` + "\n" + `data: {"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":42}}`,
			`event: message_stop` + "\n" + `data: {}`,
		} {
			_, _ = w.Write([]byte(line + "\n\n"))
		}
	})
	s, err := p.Stream(context.Background(), attempt(&llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := wirecore.Fold(s, wirecore.FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	parts := resp.Choices[0].Message.Content
	th, ok := parts[0].(*llm.Thinking)
	if !ok || th.Text != "hmm" || th.Signature != "sig" {
		t.Fatalf("thinking = %+v", parts[0])
	}
	tu := llm.ToolUses(parts)
	if len(tu) != 1 || tu[0].ID != "tu_1" || string(tu[0].Args) != `{"q":1}` {
		t.Fatalf("tool call = %+v", tu)
	}
	if resp.Choices[0].Stop != llm.StopToolUse {
		t.Fatalf("stop = %q", resp.Choices[0].Stop)
	}
	if resp.Usage.Prompt != 912 || resp.Usage.Completion != 42 {
		t.Fatalf("usage = %+v: message_start carries the prompt breakdown and message_delta the output", resp.Usage)
	}
}

func TestStreamSurfacesAnInBandError(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("event: message_start\ndata: {\"message\":{\"id\":\"m\"}}\n\n" +
			"event: error\ndata: {\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n\n"))
	})
	s, err := p.Stream(context.Background(), attempt(&llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}))
	if err != nil {
		t.Fatal(err)
	}
	_, ferr := wirecore.Fold(s, wirecore.FoldOptions{})
	if ferr == nil {
		t.Fatal("an error event inside a 200 body must not read as a short success")
	}
	if llm.AsError(ferr).Kind != llm.KindOverloaded {
		t.Fatalf("err = %v", ferr)
	}
}

func TestUpstreamStatusIsTyped(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "3")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
	})
	_, err := p.Complete(context.Background(), attempt(&llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}))
	e := llm.AsError(err)
	if e.Kind != llm.KindRateLimit || e.Status != 429 {
		t.Fatalf("error = %+v", e)
	}
	if e.RetryAfter.Seconds() != 3 || !e.Retryable || !e.Sent {
		t.Fatalf("error = %+v: the back-off hint and the billing flag both matter", e)
	}
	if !strings.Contains(e.Message, "slow down") {
		t.Fatalf("message = %q: the upstream explanation must survive", e.Message)
	}
}

func TestAnErrorBodyInA200IsAnError(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	})
	resp, err := p.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err == nil {
		t.Fatalf("a 200 carrying an error body was served as an answer: %+v", resp)
	}
	e := llm.AsError(err)
	if e.Kind != llm.KindRateLimit || e.Message != "slow down" {
		t.Fatalf("error = %+v", e)
	}
}

func TestAnUnknownBlockIsCarriedOnBothPaths(t *testing.T) {
	const block = `{"type":"server_tool_use","id":"srv_1","name":"web_search"}`

	unary, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",
			"content":[` + block + `],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	resp, err := unary.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if !carriesRaw(resp.Choices[0].Message.Content, "server_tool_use") {
		t.Fatalf("unary dropped it: %+v", resp.Choices[0].Message.Content)
	}

	streamed, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range []string{
			`{"type":"message_start","message":{"id":"msg_1","model":"m","usage":{"input_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":` + block + `}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`,
			`{"type":"message_stop"}`,
		} {
			_, _ = w.Write([]byte("data: " + e + "\n\n"))
		}
	})
	s, err := streamed.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var got []llm.Part
	for f := range s.Frames() {
		got = append(got, f.Delta...)
	}
	if !carriesRaw(got, "server_tool_use") {
		t.Fatalf("the stream dropped it: %+v", got)
	}
}

func hello() *llm.Request {
	return &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}
}

func carriesRaw(parts []llm.Part, kind string) bool {
	for _, p := range parts {
		if r, ok := p.(*llm.Raw); ok && r.Type == kind {
			return true
		}
	}
	return false
}

func TestUsageIsReportedOnlyWhereTheBackendSaidSomething(t *testing.T) {
	silent, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",
			"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"}`))
	})
	resp, err := silent.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Reported {
		t.Fatalf("usage = %+v: the backend said nothing and it books as a measurement", resp.Usage)
	}
	if !resp.Usage.Unmeasured().Reported && !hasAnomalyCode(resp.Usage.Unmeasured(), llm.AnomalyNotMeasurable) {
		t.Error("a blind call must be marked as one")
	}

	spoke, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",
			"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":0,"output_tokens":0}}`))
	})
	resp2, err := spoke.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if !resp2.Usage.Reported {
		t.Fatalf("usage = %+v: the backend said zero and it reads as silence", resp2.Usage)
	}
}

func hasAnomalyCode(u llm.Usage, code string) bool {
	for _, a := range u.Anomalies {
		if a.Code == code {
			return true
		}
	}
	return false
}

func TestASuppliedCeilingIsReported(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	ask := func(ceiling *int) llm.Diagnostics {
		resp, err := p.Complete(context.Background(), attempt(&llm.Request{
			Model:           "claude-test",
			MaxOutputTokens: ceiling,
			Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		}))
		if err != nil {
			t.Fatal(err)
		}
		return resp.Diagnostics
	}

	if d := ask(nil); !d.Has(llm.CapCeilingSupplied) {
		t.Fatalf("diagnostics = %+v: a ceiling the caller did not ask for was invented in silence", d)
	}
	if !strings.Contains((*bodies)[0], `"max_tokens"`) {
		t.Fatalf("body = %s", (*bodies)[0])
	}

	asked := 64
	if d := ask(&asked); d.Has(llm.CapCeilingSupplied) {
		t.Fatalf("diagnostics = %+v: the caller's own ceiling was sent", d)
	}

	none := 0
	if d := ask(&none); !d.Has(llm.CapCeilingSupplied) {
		t.Fatalf("diagnostics = %+v: a request for no ceiling cannot be served as asked", d)
	}
}
