// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/mcp"
)

func clientFor(t *testing.T, handler http.HandlerFunc) *goose.Client {
	t.Helper()
	upstream := httptest.NewServer(handler)
	t.Cleanup(upstream.Close)
	cfg := &Config{Providers: []Provider{{
		ID: "local", Kind: "compatible", Model: "m", BaseURL: upstream.URL,
	}}}
	client, err := build(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Close(context.Background()) })
	return client
}

func TestAskCarriesTheAnswerAndWhatServedIt(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`)
	})
	tools := gooseTools(client)
	if len(tools) != 2 || tools[0].Definition.Name != "ask" || tools[1].Definition.Name != "models" {
		t.Fatalf("tools = %+v", tools)
	}

	result, err := tools[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError || llm.TextOf(result.Content) != "hello" {
		t.Fatalf("result = %+v", result)
	}
	var structured struct {
		Text  string    `json:"text"`
		Model string    `json:"model"`
		Stop  string    `json:"stop"`
		Usage llm.Usage `json:"usage"`
	}
	if err := json.Unmarshal(result.Structured, &structured); err != nil {
		t.Fatal(err)
	}
	if structured.Text != "hello" || structured.Model != "m" || structured.Usage.Total != 5 {
		t.Fatalf("structured = %+v: a client with an output schema has nothing to validate", structured)
	}
}

func TestAskReportsAnUpstreamRefusalAsAToolError(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"no such model","type":"invalid_request_error"}}`)
	})
	result, err := gooseTools(client)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatalf("an upstream refusal became a protocol error: %v", err)
	}
	if !result.IsError || !strings.Contains(llm.TextOf(result.Content), "no such model") {
		t.Fatalf("result = %+v", result)
	}
}

func TestAskRefusesAnEmptyPrompt(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {})
	if _, err := gooseTools(client)[0].Call(context.Background(), json.RawMessage(`{"prompt":"  "}`)); err == nil {
		t.Fatal("a prompt with nothing in it is a call that cannot be served")
	}
}

func TestModelsListsWhatTheProxyRoutesTo(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","data":[{"id":"m","object":"model","owned_by":"local"}]}`)
	})
	result, err := gooseTools(client)[1].Call(context.Background(), json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(llm.TextOf(result.Content), "m") {
		t.Fatalf("result = %+v", result)
	}
	var structured struct {
		Models []map[string]string `json:"models"`
	}
	if err := json.Unmarshal(result.Structured, &structured); err != nil {
		t.Fatal(err)
	}
	if len(structured.Models) != 1 || structured.Models[0]["id"] != "m" {
		t.Fatalf("structured = %+v", structured)
	}
}

func TestAskNamesWhyTheTurnEnded(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	})
	result, err := gooseTools(client)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	var structured struct {
		Stop string `json:"stop"`
	}
	if err := json.Unmarshal(result.Structured, &structured); err != nil {
		t.Fatal(err)
	}
	if structured.Stop != string(llm.StopLength) {
		t.Fatalf("stop = %q: the answer was cut off and said so nowhere", structured.Stop)
	}
}

func TestAskReportsAnAnswerWithNoText(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":"content_filter"}],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}`)
	})
	result, err := gooseTools(client)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError || !strings.Contains(llm.TextOf(result.Content), string(llm.StopContentFilter)) {
		t.Fatalf("result = %+v", result)
	}
}

func TestAskReportsTheAnswerFormingWhereAnyoneIsWatching(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		for _, chunk := range []string{"hel", "lo ", "there"} {
			io.WriteString(w, `data: {"id":"1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"`+chunk+`"}}]}`+"\n\n")
			flusher.Flush()
		}
		io.WriteString(w, `data: {"id":"1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n")
		io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	})

	var reports []mcp.Progress
	ctx := mcp.WithProgress(context.Background(), func(p mcp.Progress) { reports = append(reports, p) })
	result, err := gooseTools(client)[0].Call(ctx, json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError || llm.TextOf(result.Content) != "hello there" {
		t.Fatalf("result = %+v", result)
	}
	if len(reports) < 2 {
		t.Fatalf("reports = %+v: the caller watched a minute of nothing", reports)
	}
	for i := 1; i < len(reports); i++ {
		if reports[i].Done <= reports[i-1].Done {
			t.Fatalf("reports = %+v: progress must rise", reports)
		}
	}
	var structured struct {
		Text  string    `json:"text"`
		Stop  string    `json:"stop"`
		Usage llm.Usage `json:"usage"`
	}
	if err := json.Unmarshal(result.Structured, &structured); err != nil {
		t.Fatal(err)
	}
	if structured.Text != "hello there" || structured.Stop != string(llm.StopEnd) {
		t.Fatalf("structured = %+v", structured)
	}
}
