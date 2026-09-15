// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/ollama"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openairesp"
)

func vendorProxy(t testing.TB, build func(baseURL string) (llm.Provider, error), codec wire.Codec) (*httptest.Server, func(string)) {
	t.Helper()
	var mu sync.Mutex
	reply := `{}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		mu.Lock()
		body := reply
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(upstream.Close)

	p, err := build(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{codec}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, func(body string) {
		mu.Lock()
		reply = body
		mu.Unlock()
	}
}

func ask(t *testing.T, front *httptest.Server, path, body string) (int, map[string]any, string) {
	t.Helper()
	resp, err := front.Client().Post(front.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &out)
	}
	return resp.StatusCode, out, string(raw)
}

func TestAnthropicAnswerSurvivesItsOwnWire(t *testing.T) {
	front, answer := vendorProxy(t,
		func(url string) (llm.Provider, error) {
			return anthropic.New(anthropic.Options{ID: "up", BaseURL: url, APIKey: "k"})
		},
		anthropicmsg.New(anthropicmsg.Options{}))

	answer(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[
		{"type":"thinking","thinking":"because","signature":"sig"},
		{"type":"text","text":"hello"},
		{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}],
		"stop_reason":"tool_use","usage":{"input_tokens":9,"output_tokens":4,"cache_read_input_tokens":7}}`)

	status, out, raw := ask(t, front, "/v1/messages",
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	if status != http.StatusOK {
		t.Fatalf("status = %d: %s", status, raw)
	}
	for _, want := range []string{
		`"type":"thinking"`, `"signature":"sig"`, `"text":"hello"`,
		`"type":"tool_use"`, `"name":"f"`, `"stop_reason":"tool_use"`,
	} {
		if !strings.Contains(raw, want) {
			t.Errorf("the answer is missing %s:\n%s", want, raw)
		}
	}
	usage, _ := out["usage"].(map[string]any)
	if in, cached := usage["input_tokens"], usage["cache_read_input_tokens"]; in != float64(9) || cached != float64(7) {
		t.Errorf("usage = %v, want the 9 and 7 the backend reported", usage)
	}
}

func TestOllamaAnswerSurvivesItsOwnWire(t *testing.T) {
	front, answer := vendorProxy(t,
		func(url string) (llm.Provider, error) {
			return ollama.New(ollama.Options{ID: "up", BaseURL: url})
		},
		ollamaapi.NewChat(ollamaapi.Options{}))

	answer(`{"model":"m","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant",
		"content":"hello","thinking":"because","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},
		"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":2}`)

	status, out, raw := ask(t, front, "/api/chat",
		`{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`)
	if status != http.StatusOK {
		t.Fatalf("status = %d: %s", status, raw)
	}
	for _, want := range []string{`"content":"hello"`, `"thinking":"because"`, `"name":"f"`, `"done":true`} {
		if !strings.Contains(raw, want) {
			t.Errorf("the answer is missing %s:\n%s", want, raw)
		}
	}
	if out["prompt_eval_count"] != float64(3) || out["eval_count"] != float64(2) {
		t.Errorf("counters = %v / %v", out["prompt_eval_count"], out["eval_count"])
	}
}

func TestResponsesAnswerSurvivesItsOwnWire(t *testing.T) {
	front, answer := vendorProxy(t,
		func(url string) (llm.Provider, error) {
			return openai.New(openai.Options{ID: "up", BaseURL: url, APIKey: "k", Dialect: openai.OpenAIDialect()})
		},
		openairesp.New(openairesp.Options{}))

	answer(`{"id":"resp_1","object":"response","model":"m","status":"completed","output":[
		{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"because"}]},
		{"type":"message","id":"msg_1","role":"assistant","status":"completed",
		 "content":[{"type":"output_text","text":"hello"}]},
		{"type":"function_call","id":"fc_1","call_id":"c1","name":"f","arguments":"{\"a\":1}"}],
		"usage":{"input_tokens":3,"output_tokens":2}}`)

	status, out, raw := ask(t, front, "/v1/responses", `{"model":"m","input":"hi"}`)
	if status != http.StatusOK {
		t.Fatalf("status = %d: %s", status, raw)
	}
	for _, want := range []string{
		`"type":"reasoning"`, `"summary_text"`, `"because"`,
		`"output_text"`, `"hello"`, `"type":"function_call"`, `"name":"f"`,
	} {
		if !strings.Contains(raw, want) {
			t.Errorf("the answer is missing %s:\n%s", want, raw)
		}
	}
	if out["id"] != "resp_1" {
		t.Errorf("id = %v, want the backend's own", out["id"])
	}
}
