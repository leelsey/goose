// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/openaichat"
)

func answering(t testing.TB) (*httptest.Server, func(string)) {
	t.Helper()
	var mu sync.Mutex
	reply := `{}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		body := reply
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
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
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
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

func FuzzOpenAIAnswerSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","refusal":"no"},"finish_reason":"content_filter"}]}`,
		`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi","reasoning_content":"because"},"finish_reason":"length"}]}`,
		`{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"a"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":7},"completion_tokens_details":{"reasoning_tokens":2}}}`,
	} {
		f.Add(seed)
	}
	front, answer := answering(f)
	f.Fuzz(func(t *testing.T, reply string) {
		if !json.Valid([]byte(reply)) {
			return
		}
		answer(reply)
		resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		out, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compareAnswer(t, reply, out, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compareAnswer(t testing.TB, sent string, got []byte, reported string) {
	t.Helper()
	var in, out map[string]any
	if json.Unmarshal([]byte(sent), &in) != nil || json.Unmarshal(got, &out) != nil {
		return
	}
	lower(in)
	lower(out)
	fail := func(what string, want, have any) {
		t.Helper()
		if reported == "" {
			t.Fatalf("%s changed and nothing said so\n backend: %v\n  client: %v\n    from: %s", what, want, have, sent)
		}
	}

	wantChoice := firstChoice(in)
	haveChoice := firstChoice(out)
	if wantChoice == nil {
		return
	}
	if haveChoice == nil {
		fail("the answer", wantChoice, nil)
		return
	}
	wantMsg, _ := wantChoice["message"].(map[string]any)
	haveMsg, _ := haveChoice["message"].(map[string]any)
	lower(wantMsg)
	lower(haveMsg)

	if text, ok := wantMsg["content"].(string); ok && text != "" {
		if have, _ := haveMsg["content"].(string); have != text {
			fail("content", text, have)
		}
	}
	if refusal, ok := wantMsg["refusal"].(string); ok && refusal != "" {
		have, _ := haveMsg["refusal"].(string)
		if have != refusal && !strings.Contains(text(haveMsg["content"]), refusal) {
			fail("refusal", refusal, haveMsg["refusal"])
		}
	}
	if want, have := calls(wantMsg["tool_calls"]), calls(haveMsg["tool_calls"]); !reflect.DeepEqual(want, have) {
		fail("tool_calls", want, have)
	}
	if reason, ok := wantChoice["finish_reason"].(string); ok && reason != "" {
		if have, _ := haveChoice["finish_reason"].(string); have == "" {
			fail("finish_reason", reason, have)
		}
	}
	wantUsage, _ := in["usage"].(map[string]any)
	haveUsage, _ := out["usage"].(map[string]any)
	lower(wantUsage)
	lower(haveUsage)
	for _, field := range []string{"prompt_tokens", "completion_tokens"} {
		want, had := wantUsage[field]
		if !had || want == float64(0) {
			continue
		}
		if have := haveUsage[field]; have != want {
			fail("usage."+field, want, have)
		}
	}
}

func text(v any) string {
	s, _ := v.(string)
	return s
}

func firstChoice(body map[string]any) map[string]any {
	list, ok := body["choices"].([]any)
	if !ok || len(list) == 0 {
		return nil
	}
	c, _ := list[0].(map[string]any)
	if c != nil {
		lower(c)
	}
	return c
}

func calls(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, entry := range list {
		call, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		lower(call)
		fn, ok := call["function"].(map[string]any)
		if !ok {
			continue
		}
		lower(fn)
		if name, ok := fn["name"].(string); ok && name != "" {
			out = append(out, name)
		}
	}
	return out
}

func TestUsageColumnsFoldWhatTheVendorSplitOut(t *testing.T) {
	front, answer := answering(t)
	answer(`{"id":"c","object":"chat.completion","created":1,"model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,
		"prompt_tokens_details":{"cached_tokens":80},
		"completion_tokens_details":{"reasoning_tokens":15}}}`)

	resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)

	var u struct {
		Prompt     int `json:"prompt"`
		Completion int `json:"completion"`
		Total      int `json:"total"`
		CacheRead  int `json:"cache_read"`
		Reasoning  int `json:"reasoning"`
	}
	header := resp.Header.Get(httpx.UsageHeader)
	if header == "" {
		t.Fatal("a client that did not ask for usage in band still gets the header")
	}
	if err := json.Unmarshal([]byte(header), &u); err != nil {
		t.Fatalf("usage header = %q: %v", header, err)
	}
	if u.Prompt != 100 || u.CacheRead != 80 {
		t.Fatalf("usage = %+v: the cache figures are part of the prompt, not beside it", u)
	}
	if u.Completion != 20 || u.Reasoning != 15 {
		t.Fatalf("usage = %+v: reasoning is part of the completion, not beside it", u)
	}
	if u.Total != u.Prompt+u.Completion {
		t.Fatalf("total %d is not %d + %d", u.Total, u.Prompt, u.Completion)
	}
}
