// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"bufio"
	"encoding/json"
	"fmt"
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

func streaming(t testing.TB) (*httptest.Server, func([]string)) {
	t.Helper()
	var mu sync.Mutex
	var chunks []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		out := append([]string(nil), chunks...)
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, c := range out {
			fmt.Fprintf(w, "data: %s\n\n", c)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
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
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, func(out []string) {
		mu.Lock()
		chunks = out
		mu.Unlock()
	}
}

func chunkScript(script []byte) []string {
	var out []string
	for i, b := range script {
		switch b % 7 {
		case 0:
			out = append(out, `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"}}]}`)
		case 1:
			out = append(out, fmt.Sprintf(`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"t%d "}}]}`, i))
		case 2:
			out = append(out, `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]}}]}`)
		case 3:
			out = append(out, `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]}}]}`)
		case 4:
			out = append(out, `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"reasoning_content":"because "}}]}`)
		case 5:
			out = append(out, `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		case 6:
			out = append(out, `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}`)
		}
	}
	return out
}

func FuzzOpenAIStreamSurvivesTheProxy(f *testing.F) {
	for _, seed := range [][]byte{
		{0, 1, 5, 6}, {1, 1, 1, 5}, {2, 3, 5}, {0, 4, 1, 5, 6}, {5}, {1}, {},
	} {
		f.Add(seed)
	}
	front, emit := streaming(f)
	f.Fuzz(func(t *testing.T, script []byte) {
		if len(script) > 24 {
			script = script[:24]
		}
		sent := chunkScript(script)
		emit(sent)

		resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		got := readChunks(t, resp.Body)
		compareStream(t, sent, got)
	})
}

func readChunks(t *testing.T, r io.Reader) []map[string]any {
	t.Helper()
	var out []map[string]any
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatalf("the proxy wrote a chunk that will not parse: %q", data)
		}
		out = append(out, m)
	}
	return out
}

func compareStream(t *testing.T, sent []string, got []map[string]any) {
	t.Helper()
	var wantText, haveText strings.Builder
	var wantCalls, haveCalls []string
	wantFinish, haveFinish := "", ""

	for _, raw := range sent {
		var m map[string]any
		if json.Unmarshal([]byte(raw), &m) != nil {
			continue
		}
		text, calls, finish := chunkParts(m)
		wantText.WriteString(text)
		wantCalls = append(wantCalls, calls...)
		if finish != "" {
			wantFinish = finish
		}
	}
	for _, m := range got {
		text, calls, finish := chunkParts(m)
		haveText.WriteString(text)
		haveCalls = append(haveCalls, calls...)
		if finish != "" {
			haveFinish = finish
		}
	}

	if wantText.String() != haveText.String() {
		t.Fatalf("the text streamed differs\n backend: %q\n  client: %q\n  script: %v", wantText.String(), haveText.String(), sent)
	}
	if !reflect.DeepEqual(wantCalls, haveCalls) {
		t.Fatalf("the tool calls differ\n backend: %v\n  client: %v\n  script: %v", wantCalls, haveCalls, sent)
	}
	if wantFinish != "" && haveFinish == "" {
		t.Fatalf("the finish reason was lost\n script: %v", sent)
	}
}

func chunkParts(m map[string]any) (string, []string, string) {
	choices, _ := m["choices"].([]any)
	var text, finish string
	var calls []string
	for _, entry := range choices {
		c, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		if r, ok := c["finish_reason"].(string); ok && r != "" {
			finish = r
		}
		delta, _ := c["delta"].(map[string]any)
		if delta == nil {
			continue
		}
		if s, ok := delta["content"].(string); ok {
			text += s
		}
		calls = append(calls, calls2(delta["tool_calls"])...)
	}
	return text, calls, finish
}

func calls2(v any) []string {
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
		fn, ok := call["function"].(map[string]any)
		if !ok {
			continue
		}
		if name, ok := fn["name"].(string); ok && name != "" {
			out = append(out, name)
		}
	}
	return out
}
