// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/ollama"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/openaichat"
)

func TestAnAnthropicClientStreamsFromAnOllamaBackend(t *testing.T) {
	backend, emit := sseBackend(t, "application/x-ndjson")
	emit([]string{
		`{"model":"m","message":{"role":"assistant","content":"hel"},"done":false}` + "\n",
		`{"model":"m","message":{"role":"assistant","content":"lo"},"done":false}` + "\n",
		`{"model":"m","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},"done":false}` + "\n",
		`{"model":"m","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":11,"eval_count":4}` + "\n",
	})
	p, err := ollama.New(ollama.Options{ID: "up", BaseURL: backend.URL})
	if err != nil {
		t.Fatal(err)
	}
	srv := front(t, p, anthropicmsg.New(anthropicmsg.Options{}))

	resp, err := srv.Client().Post(srv.URL+"/v1/messages", "application/json",
		strings.NewReader(`{"model":"m","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	var names []string
	var text strings.Builder
	var opens, stops int
	var stop string
	var usage map[string]any
	for _, line := range strings.Split(string(body), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var ev struct {
			Type  string `json:"type"`
			Delta struct {
				Type       string `json:"type"`
				Text       string `json:"text"`
				StopReason string `json:"stop_reason"`
			} `json:"delta"`
			Usage map[string]any `json:"usage"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		names = append(names, ev.Type)
		switch ev.Type {
		case "content_block_start":
			opens++
		case "content_block_stop":
			stops++
		case "content_block_delta":
			text.WriteString(ev.Delta.Text)
		case "message_delta":
			stop = ev.Delta.StopReason
			usage = ev.Usage
		}
	}

	if len(names) == 0 || names[0] != "message_start" {
		t.Fatalf("the stream did not open with message_start: %v\n%s", names, body)
	}
	if names[len(names)-1] != "message_stop" {
		t.Fatalf("the stream did not close with message_stop: %v", names)
	}
	if opens != stops || opens == 0 {
		t.Errorf("content blocks do not balance: %d opened, %d closed", opens, stops)
	}
	if text.String() != "hello" {
		t.Errorf("text = %q: every token must arrive, in order", text.String())
	}
	if stop != "tool_use" {
		t.Errorf("stop_reason = %q", stop)
	}
	if !strings.Contains(string(body), `"name":"f"`) {
		t.Errorf("the tool call did not reach the client:\n%s", body)
	}
	if usage["input_tokens"] != float64(11) || usage["output_tokens"] != float64(4) {
		t.Errorf("usage = %v", usage)
	}
	if code := resp.Header.Get("X-Goose-Provider"); code != "up" {
		t.Errorf("provider header = %q", code)
	}
}

func TestGeminiParallelToolCallsReachAnOpenAIClientApart(t *testing.T) {
	backend, emit := sseBackend(t, "text/event-stream")
	emit([]string{
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"index":0}]}` + "\n\n",
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"g","args":{"b":2}}}]},"index":0}]}` + "\n\n",
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2}}` + "\n\n",
	})
	p, err := google.New(google.Options{ID: "up", BaseURL: backend.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	srv := front(t, p, openaichat.New(openaichat.Options{}))

	resp, err := srv.Client().Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	args := map[int]string{}
	names := map[int]string{}
	for _, line := range strings.Split(string(body), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    *int `json:"index"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		for _, ch := range chunk.Choices {
			for _, tc := range ch.Delta.ToolCalls {
				if tc.Index == nil {
					t.Fatalf("a tool call reached the client with no index:\n%s", body)
				}
				args[*tc.Index] += tc.Function.Arguments
				if tc.Function.Name != "" {
					names[*tc.Index] = tc.Function.Name
				}
			}
		}
	}
	if names[0] != "f" || names[1] != "g" {
		t.Fatalf("calls = %v", names)
	}
	if args[0] != `{"a":1}` || args[1] != `{"b":2}` {
		t.Fatalf("arguments = %v", args)
	}
}
