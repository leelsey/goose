// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openaichat"
)

func crossProxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",
			"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := anthropic.New(anthropic.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func crossPost(t *testing.T, front *httptest.Server, body string) (*http.Response, string) {
	t.Helper()
	resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp, string(out)
}

func TestACeilingSubstitutedAcrossFormatsIsReported(t *testing.T) {
	front, seen := crossProxy(t)
	resp, _ := crossPost(t, front, `{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`)
	if resp.Header.Get(httpx.DiagnosticsHeader) == "" {
		t.Fatalf("the substitute went unreported; upstream got %s", *seen)
	}
	if !strings.Contains(string(*seen), `"max_tokens":4096`) {
		t.Fatalf("upstream = %s", *seen)
	}
}

func TestACrossFormatRequestKeepsWhatItCarried(t *testing.T) {
	front, seen := crossProxy(t)
	resp, body := crossPost(t, front, `{"model":"m","max_tokens":64,"temperature":0.25,"top_p":0.8,
		"stop":["END"],"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"lookup","description":"d","parameters":{"type":"object"}}}]}`)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	var got map[string]any
	if err := json.Unmarshal(*seen, &got); err != nil {
		t.Fatal(err)
	}
	if got["max_tokens"] != float64(64) || got["temperature"] != 0.25 || got["top_p"] != 0.8 {
		t.Fatalf("sampling lost: %s", *seen)
	}
	if !strings.Contains(string(*seen), `"stop_sequences":["END"]`) {
		t.Fatalf("stop sequences lost: %s", *seen)
	}
	if !strings.Contains(string(*seen), `"system"`) || !strings.Contains(string(*seen), "be brief") {
		t.Fatalf("the system block lost: %s", *seen)
	}
	if !strings.Contains(string(*seen), `"name":"lookup"`) || !strings.Contains(string(*seen), `"input_schema"`) {
		t.Fatalf("the tool lost: %s", *seen)
	}
	if !strings.Contains(body, `"object":"chat.completion"`) || !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Fatalf("the answer was not rendered for the client: %s", body)
	}
}

func crossProxyStrict(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := openai.New(openai.Options{
		ID: "up", BaseURL: upstream.URL + "/v1", APIKey: "k", Dialect: openai.OpenAIDialect(),
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p), goose.WithLossPolicy(llm.LossPolicy{Strict: true}))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{anthropicmsg.New(anthropicmsg.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func TestTheEndUserReachesTheBackendUnderItsOwnName(t *testing.T) {
	front, seen := crossProxy(t)
	crossPost(t, front, `{"model":"m","max_tokens":16,"user":"tenant-7","messages":[{"role":"user","content":"hi"}]}`)
	if !strings.Contains(string(*seen), `"user_id":"tenant-7"`) {
		t.Fatalf("the end user did not reach the Anthropic backend: %s", *seen)
	}
}

func TestMetadataBesideTheEndUserIsReported(t *testing.T) {
	front, seen := crossProxy(t)
	resp, _ := crossPost(t, front, `{"model":"m","max_tokens":9,"user":"u-1","metadata":{"tenant":"acme"},"messages":[{"role":"user","content":"hi"}]}`)
	if !strings.Contains(string(*seen), `"user_id":"u-1"`) {
		t.Fatalf("the end user did not travel: %s", *seen)
	}
	if strings.Contains(string(*seen), "acme") {
		t.Fatalf("a field this wire has no place for was sent: %s", *seen)
	}
	if resp.Header.Get(httpx.DiagnosticsHeader) == "" {
		t.Fatal("a dropped metadata key went unreported")
	}
}

func TestAnUnmodelledFieldDoesNotCrossVendors(t *testing.T) {
	front, seen := crossProxy(t)
	resp, _ := crossPost(t, front, `{"model":"m","max_tokens":9,"anthropic_only_thing":{"a":1},"messages":[{"role":"user","content":"hi"}]}`)
	if strings.Contains(string(*seen), "anthropic_only_thing") {
		t.Fatalf("an Anthropic field reached an OpenAI backend: %s", *seen)
	}
	if resp.Header.Get(httpx.DiagnosticsHeader) == "" {
		t.Fatal("the field was dropped and nothing said so")
	}
}

func TestAnthropicStreamCarriesTheWholeBillAtTheEnd(t *testing.T) {
	front, _ := crossStreamProxy(t)
	resp, err := front.Client().Post(front.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"m","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	start, delta := usageOf(t, string(body), "message_start"), usageOf(t, string(body), "message_delta")
	if start.In != 0 {
		t.Logf("message_start carried input_tokens=%d", start.In)
	}
	if delta.In != 11 || delta.Out != 5 {
		t.Fatalf("message_delta usage = %d/%d, want the whole bill 11/5", delta.In, delta.Out)
	}
	var header struct {
		Prompt     int `json:"prompt"`
		Completion int `json:"completion"`
	}
	if h := resp.Header.Get(httpx.UsageHeader); h != "" {
		if err := json.Unmarshal([]byte(h), &header); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("usage header: %d/%d", header.Prompt, header.Completion)
}

type anthropicUsage struct{ In, Out int }

func usageOf(t *testing.T, body, event string) anthropicUsage {
	t.Helper()
	for _, block := range strings.Split(body, "\n\n") {
		if !strings.Contains(block, "event: "+event) {
			continue
		}
		for _, line := range strings.Split(block, "\n") {
			data, ok := strings.CutPrefix(line, "data: ")
			if !ok {
				continue
			}
			var payload struct {
				Usage   *struct{ In, Out int } `json:"-"`
				Raw     map[string]any         `json:"-"`
				Message *json.RawMessage       `json:"message"`
				U       map[string]any         `json:"usage"`
			}
			if json.Unmarshal([]byte(data), &payload) != nil {
				continue
			}
			u := payload.U
			if u == nil && payload.Message != nil {
				var m struct {
					U map[string]any `json:"usage"`
				}
				if json.Unmarshal(*payload.Message, &m) == nil {
					u = m.U
				}
			}
			if u == nil {
				continue
			}
			in, _ := u["input_tokens"].(float64)
			out, _ := u["output_tokens"].(float64)
			return anthropicUsage{In: int(in), Out: int(out)}
		}
	}
	t.Fatalf("no %s event carried usage:\n%s", event, body)
	return anthropicUsage{}
}

func crossStreamProxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		for _, chunk := range []string{
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16}}`,
		} {
			io.WriteString(w, "data: "+chunk+"\n\n")
			if f != nil {
				f.Flush()
			}
		}
		io.WriteString(w, "data: [DONE]\n\n")
		if f != nil {
			f.Flush()
		}
	}))
	t.Cleanup(upstream.Close)

	p, err := openai.New(openai.Options{
		ID: "up", BaseURL: upstream.URL + "/v1", APIKey: "k", Dialect: openai.OpenAIDialect(),
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{anthropicmsg.New(anthropicmsg.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func TestBothPenaltiesCrossEveryWire(t *testing.T) {
	cases := []struct {
		name, path, body string
	}{
		{"gemini", "/v1beta/models/m:generateContent",
			`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
			  "generationConfig":{"frequencyPenalty":0.5,"presencePenalty":0.25}}`},
		{"ollama", "/api/chat",
			`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false,
			  "options":{"frequency_penalty":0.5,"presence_penalty":0.25}}`},
		{"openai", "/v1/chat/completions",
			`{"model":"m","messages":[{"role":"user","content":"hi"}],
			  "frequency_penalty":0.5,"presence_penalty":0.25}`},
	}
	for _, c := range cases {
		req := decodeOn(t, c.path, c.body)
		if req.FrequencyPenalty == nil || *req.FrequencyPenalty != 0.5 {
			t.Errorf("%s: frequency penalty = %v", c.name, req.FrequencyPenalty)
		}
		if req.PresencePenalty == nil || *req.PresencePenalty != 0.25 {
			t.Errorf("%s: presence penalty = %v", c.name, req.PresencePenalty)
		}
	}
}

func decodeOn(t *testing.T, path, body string) *llm.Request {
	t.Helper()
	for _, c := range []wire.Codec{
		openaichat.New(openaichat.Options{}),
		geminiapi.New(geminiapi.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
	} {
		for _, r := range c.Routes() {
			if strings.HasPrefix(path, strings.TrimSuffix(r.Pattern, "/")) {
				req, _, _, err := c.Decode(&wire.Inbound{Method: "POST", Path: path, Body: []byte(body)})
				if err != nil {
					t.Fatalf("%s: %v", path, err)
				}
				return req
			}
		}
	}
	t.Fatalf("no codec claims %s", path)
	return nil
}

func TestOneToolCallAtATimeCrossesTheWires(t *testing.T) {
	openaiBody := `{"model":"m","messages":[{"role":"user","content":"hi"}],"parallel_tool_calls":false,
		"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]}`
	anthropicBody := `{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"f","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"auto","disable_parallel_tool_use":true}}`

	for _, c := range []struct{ name, path, body string }{
		{"openai", "/v1/chat/completions", openaiBody},
		{"anthropic", "/v1/messages", anthropicBody},
	} {
		req := decodeAny(t, c.path, c.body)
		if req.ParallelTools == nil || *req.ParallelTools {
			t.Fatalf("%s: parallel = %v: the client asked for one at a time", c.name, req.ParallelTools)
		}
	}

	sent := sendTo(t, "anthropic", &llm.Request{
		Model: "m", ParallelTools: new(bool),
		Tools:    []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`)}},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if !strings.Contains(sent, `"disable_parallel_tool_use":true`) {
		t.Fatalf("anthropic body = %s", sent)
	}
	sent = sendTo(t, "openai", &llm.Request{
		Model: "m", ParallelTools: new(bool),
		Tools:    []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`)}},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if !strings.Contains(sent, `"parallel_tool_calls":false`) {
		t.Fatalf("openai body = %s", sent)
	}
}

func decodeAny(t *testing.T, path, body string) *llm.Request {
	t.Helper()
	for _, c := range []wire.Codec{
		openaichat.New(openaichat.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
	} {
		for _, r := range c.Routes() {
			if path == r.Pattern {
				req, _, _, err := c.Decode(&wire.Inbound{Method: "POST", Path: path, Body: []byte(body)})
				if err != nil {
					t.Fatalf("%s: %v", path, err)
				}
				return req
			}
		}
	}
	t.Fatalf("no codec claims %s", path)
	return nil
}

func sendTo(t *testing.T, kind string, req *llm.Request) string {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		if kind == "anthropic" {
			_, _ = w.Write([]byte(`{"id":"m1","model":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn"}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer upstream.Close()

	var p llm.Provider
	var err error
	if kind == "anthropic" {
		p, err = anthropic.New(anthropic.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	} else {
		p, err = openai.New(openai.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	}
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	return string(seen)
}
