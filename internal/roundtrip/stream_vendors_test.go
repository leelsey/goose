// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"bufio"
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
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
)

func sseBackend(t testing.TB, contentType string) (*httptest.Server, func([]string)) {
	t.Helper()
	var mu sync.Mutex
	var events []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		out := append([]string(nil), events...)
		mu.Unlock()
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		for _, e := range out {
			io.WriteString(w, e)
			if f != nil {
				f.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv, func(e []string) {
		mu.Lock()
		events = e
		mu.Unlock()
	}
}

func front(t testing.TB, p llm.Provider, codec wire.Codec) *httptest.Server {
	t.Helper()
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{codec}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv
}

func TestAnthropicStreamSurvivesItsOwnWire(t *testing.T) {
	backend, emit := sseBackend(t, "text/event-stream")
	emit([]string{
		"event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":11,"output_tokens":0}}}` + "\n\n",
		"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}` + "\n\n",
		"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"because"}}` + "\n\n",
		"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}` + "\n\n",
		"event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":0}` + "\n\n",
		"event: content_block_start\ndata: " + `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}` + "\n\n",
		"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"the answer"}}` + "\n\n",
		"event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":1}` + "\n\n",
		"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}` + "\n\n",
		"event: message_stop\ndata: " + `{"type":"message_stop"}` + "\n\n",
	})
	p, err := anthropic.New(anthropic.Options{ID: "up", BaseURL: backend.URL, APIKey: "k"})
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
	got := string(body)

	for _, want := range []string{
		`"thinking_delta"`, "because", `"signature_delta"`, "sig-1",
		"the answer", `"stop_reason":"end_turn"`, "message_stop",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("%q did not survive:\n%s", want, got)
		}
	}
	if strings.Index(got, "because") > strings.Index(got, "sig-1") {
		t.Fatalf("the signature arrived before the thinking it signs:\n%s", got)
	}
	if !strings.Contains(got, `"input_tokens":11`) || !strings.Contains(got, `"output_tokens":7`) {
		t.Fatalf("the bill did not survive:\n%s", got)
	}
}

func TestGeminiStreamSurvivesItsOwnWire(t *testing.T) {
	backend, emit := sseBackend(t, "text/event-stream")
	emit([]string{
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"one "}]},"index":0}]}` + "\n\n",
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"q":1}}}]},"index":0}]}` + "\n\n",
		"data: " + `{"candidates":[{"content":{"role":"model","parts":[{"text":"two"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8}}` + "\n\n",
	})
	p, err := google.New(google.Options{ID: "up", BaseURL: backend.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	srv := front(t, p, geminiapi.New(geminiapi.Options{}))

	resp, err := srv.Client().Post(srv.URL+"/v1beta/models/m:streamGenerateContent?alt=sse", "application/json",
		strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	var text strings.Builder
	var calls []string
	finish := ""
	usage := map[string]any{}
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok {
			continue
		}
		var chunk map[string]any
		if json.Unmarshal([]byte(data), &chunk) != nil {
			t.Fatalf("the proxy wrote a chunk that will not parse: %q", data)
		}
		part, call := geminiParts(chunk)
		text.WriteString(part)
		calls = append(calls, call...)
		if cands, _ := chunk["candidates"].([]any); len(cands) > 0 {
			if cand, _ := cands[0].(map[string]any); cand != nil {
				if r, _ := cand["finishreason"].(string); r != "" {
					finish = r
				}
			}
		}
		if u, _ := chunk["usagemetadata"].(map[string]any); u != nil {
			lower(u)
			usage = u
		}
	}
	if text.String() != "one two" {
		t.Fatalf("text = %q", text.String())
	}
	if len(calls) != 1 || calls[0] != "lookup" {
		t.Fatalf("calls = %v", calls)
	}
	if finish != "STOP" {
		t.Fatalf("finishReason = %q: a stream that does not say why it ended is not a whole answer", finish)
	}
	if usage["prompttokencount"] != float64(5) {
		t.Fatalf("usage = %v", usage)
	}
	if got := resp.Trailer.Get(httpx.UsageHeader); got == "" {
		t.Fatal("the stream ended without saying what it cost")
	} else {
		t.Logf("trailer: %s", got)
	}
}
