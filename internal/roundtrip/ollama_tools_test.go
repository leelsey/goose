// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/ollamaapi"
)

func TestAnOllamaToolConversationReplaysToAnotherWire(t *testing.T) {
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",
			"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	p, err := anthropic.New(anthropic.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{ollamaapi.NewChat(ollamaapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	defer front.Close()

	resp, err := front.Client().Post(front.URL+"/api/chat", "application/json", strings.NewReader(
		`{"model":"m","stream":false,"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},
			{"role":"tool","tool_name":"f","content":"sunny"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}

	var sent struct {
		Messages []struct {
			Content []struct {
				Type      string `json:"type"`
				ID        string `json:"id"`
				ToolUseID string `json:"tool_use_id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(seen, &sent); err != nil {
		t.Fatalf("%v: %s", err, seen)
	}
	var call, result string
	for _, m := range sent.Messages {
		for _, b := range m.Content {
			switch b.Type {
			case "tool_use":
				call = b.ID
			case "tool_result":
				result = b.ToolUseID
			}
		}
	}
	if call == "" || result == "" {
		t.Fatalf("the backend was sent an unpaired call %q and result %q:\n%s", call, result, seen)
	}
	if call != result {
		t.Fatalf("call %q does not answer to result %q", call, result)
	}
}
