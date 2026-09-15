// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestUnmodelledAnswerFieldsReachTheSameWire(t *testing.T) {
	front, reply := answering(t)
	reply(`{"id":"c","object":"chat.completion","created":1,"model":"m",
		"system_fingerprint":"fp_42","service_tier":"scale","obscure":{"a":1},
		"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)

	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var got map[string]json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"system_fingerprint": `"fp_42"`,
		"service_tier":       `"scale"`,
		"obscure":            `{"a":1}`,
	} {
		if string(got[key]) != want {
			t.Errorf("%s = %s, want %s", key, got[key], want)
		}
	}
	if string(got["model"]) != `"m"` {
		t.Errorf("model = %s", got["model"])
	}
}

func TestEveryStreamedCandidateReachesTheClient(t *testing.T) {
	front, emit := streaming(t)
	emit([]string{
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"one"}},{"index":1,"delta":{"role":"assistant","content":"two"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"},{"index":1,"delta":{},"finish_reason":"stop"}]}`,
	})

	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"n":2}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	text := map[int]string{}
	finished := map[int]bool{}
	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimPrefix(line, "data: ")
		if line == "" || line == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Index        int                      `json:"index"`
				Delta        struct{ Content string } `json:"delta"`
				FinishReason *string                  `json:"finish_reason"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(line), &chunk) != nil {
			continue
		}
		for _, ch := range chunk.Choices {
			text[ch.Index] += ch.Delta.Content
			if ch.FinishReason != nil {
				finished[ch.Index] = true
			}
		}
	}
	if text[0] != "one" || text[1] != "two" {
		t.Errorf("candidates = %q and %q", text[0], text[1])
	}
	if !finished[0] || !finished[1] {
		t.Errorf("finished = %v", finished)
	}
}

func TestUnmodelledStreamFieldsReachTheSameWire(t *testing.T) {
	front, emit := streaming(t)
	emit([]string{
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"fp_9","choices":[{"index":0,"delta":{"content":"hi"}}]}`,
		`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
	})
	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), `"system_fingerprint":"fp_9"`) {
		t.Errorf("the fingerprint did not reach the client:\n%s", body)
	}
}
