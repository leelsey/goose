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
	"sync/atomic"
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
	tools := gooseTools(client, &Config{}, nil)
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
	result, err := gooseTools(client, &Config{}, nil)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatalf("an upstream refusal became a protocol error: %v", err)
	}
	if !result.IsError || !strings.Contains(llm.TextOf(result.Content), "no such model") {
		t.Fatalf("result = %+v", result)
	}
}

// A route that pins no model sends a call naming none on with an empty one,
// which a backend that addresses models by name refuses in words that do not
// say ask could have named one.
func TestAskWithNoModelWhereNoneIsConfiguredSaysToPassOne(t *testing.T) {
	var sent atomic.Int32
	var asked atomic.Value
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sent.Add(1)
		var body struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		asked.Store(body.Model)
		if body.Model == "" {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":"model is required"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"model":"`+body.Model+`","created_at":"2026-01-01T00:00:00Z","message":{"role":"assistant","content":"hello"},"done":true,"done_reason":"stop"}`)
	}))
	t.Cleanup(upstream.Close)
	askWith := func(cfg *Config, args string) *mcp.Result {
		t.Helper()
		client, err := build(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { client.Close(context.Background()) })
		result, err := gooseTools(client, cfg, nil)[0].Call(context.Background(), json.RawMessage(args))
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	bare := &Config{
		Providers: []Provider{{ID: "local", Kind: "ollama", BaseURL: upstream.URL}},
		Routes:    []Route{{Name: "claude", Models: []string{"claude-*"}, Chain: []string{"local"}}, {Name: "rest", Chain: []string{"local"}}},
	}
	result := askWith(bare, `{"prompt":"hi"}`)
	if !result.IsError || !strings.Contains(llm.TextOf(result.Content), "pass model") || !strings.Contains(llm.TextOf(result.Content), `"local"`) {
		t.Fatalf("result = %+v: the call was refused in words that do not say what to do", result)
	}
	if n := sent.Load(); n != 0 {
		t.Fatalf("sent = %d: a call with no model to ask for went upstream", n)
	}
	if result := askWith(bare, `{"prompt":"hi","model":"llama3"}`); result.IsError {
		t.Fatalf("result = %+v", result)
	}
	pinned := &Config{Providers: []Provider{{ID: "local", Kind: "ollama", BaseURL: upstream.URL, Model: "llama3"}}}
	if result := askWith(pinned, `{"prompt":"hi"}`); result.IsError || asked.Load() != "llama3" {
		t.Fatalf("result = %+v, asked = %v: the provider's own model is the one to ask for", result, asked.Load())
	}

	for name, c := range map[string]struct {
		cfg  *Config
		want string
	}{
		"a route's model":      {&Config{Providers: []Provider{{ID: "local", Kind: "ollama"}}, Routes: []Route{{Chain: []string{"local"}, Model: "m"}}}, ""},
		"a chain entry's":      {&Config{Providers: []Provider{{ID: "local", Kind: "ollama"}}, Routes: []Route{{Chain: []string{"local:m"}}}}, ""},
		"an agent CLI":         {&Config{Providers: []Provider{{Kind: "claude"}}}, ""},
		"an Azure deployment":  {&Config{Providers: []Provider{{ID: "az", Kind: "azure", Options: map[string]string{"deployment": "d"}}}}, ""},
		"a server by shape":    {&Config{Providers: []Provider{{ID: "gw", Kind: "compatible"}}}, ""},
		"Azure without one":    {&Config{Providers: []Provider{{ID: "az", Kind: "azure"}}}, "az"},
		"a kind with no model": {&Config{Providers: []Provider{{Kind: "anthropic"}}}, "anthropic"},
	} {
		if got := unnamed(c.cfg); got != c.want {
			t.Errorf("%s: unnamed = %q, want %q", name, got, c.want)
		}
	}
}

// A backend's account of a failure routinely quotes the request back, and the
// listener's other endpoints hold it back under redact_upstream_errors.
func TestAskHoldsBackTheUpstreamsWordsWhenAskedTo(t *testing.T) {
	const secret = "Your credit balance is too low ZZSECRET"
	for name, handler := range map[string]http.HandlerFunc{
		"refused": func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadRequest)
			io.WriteString(w, `{"error":{"message":"`+secret+`","type":"invalid_request_error"}}`)
		},
		"failed part way": func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, `data: {"id":"1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"hel"}}]}`+"\n\n")
			w.(http.Flusher).Flush()
			io.WriteString(w, `data: {"error":{"message":"`+secret+`","type":"server_error"}}`+"\n\n")
		},
	} {
		upstream := httptest.NewServer(handler)
		t.Cleanup(upstream.Close)
		cfg := &Config{
			Server:    Server{RedactUpstream: true},
			Providers: []Provider{{ID: "local", Kind: "compatible", Model: "m", BaseURL: upstream.URL}},
		}
		client, err := build(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { client.Close(context.Background()) })
		for _, ctx := range []context.Context{
			context.Background(),
			mcp.WithProgress(context.Background(), func(mcp.Progress) {}),
		} {
			if name == "failed part way" && !mcp.Watched(ctx) {
				continue
			}
			result, err := gooseTools(client, cfg, nil)[0].Call(ctx, json.RawMessage(`{"prompt":"hi"}`))
			if err != nil {
				t.Fatal(err)
			}
			text := llm.TextOf(result.Content)
			if !result.IsError || strings.Contains(text, "ZZSECRET") || !strings.Contains(text, "local") {
				t.Fatalf("%s, streamed = %v: result = %q: the backend's own words reached a caller the operator asked to keep them from",
					name, mcp.Watched(ctx), text)
			}
		}
	}
}

// A refusal goose raises itself before a call is sent is not an upstream's
// words, so redact_upstream_errors must not hold it back the way it holds
// back a backend's own sentence.
func TestAskKeepsGoosesOwnRefusalUnderRedact(t *testing.T) {
	var hits atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(upstream.Close)
	cfg := &Config{
		Server:    Server{RedactUpstream: true},
		Providers: []Provider{{ID: "claude", Kind: "anthropic", Model: "claude-x", BaseURL: upstream.URL}},
		Routes:    []Route{{Chain: []string{"claude"}, Deny: []string{"cap.ceiling_supplied"}}},
	}
	client, err := build(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Close(context.Background()) })

	result, err := gooseTools(client, cfg, nil)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	text := llm.TextOf(result.Content)
	if !result.IsError || !strings.Contains(text, "cap.ceiling_supplied") || strings.Contains(text, "details withheld") {
		t.Fatalf("result = %+v: goose's own refusal was redacted like an upstream's words", result)
	}
	if n := hits.Load(); n != 0 {
		t.Fatalf("hits = %d: a call goose refused before sending reached the upstream", n)
	}
}

// A call the tool turns down is the tool's failure, which the model that made
// it reads and can correct, as it does every other refusal of ask's; a
// protocol error says goose itself broke.
func TestAskRefusesAnEmptyPrompt(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {})
	for args, want := range map[string]string{
		`{"prompt":"  "}`: "prompt is empty",
		`{"prompt":5}`:    "arguments",
		`not json`:        "arguments",
	} {
		result, err := gooseTools(client, &Config{}, nil)[0].Call(context.Background(), json.RawMessage(args))
		if err != nil {
			t.Fatalf("%s: err = %v: a call the tool turns down became a protocol error", args, err)
		}
		if !result.IsError || !strings.Contains(llm.TextOf(result.Content), want) {
			t.Fatalf("%s: result = %+v", args, result)
		}
	}
}

// Every backend is asked for its models, so a note that one could not be read
// must say which.
func TestModelsNamesTheBackendItCouldNotRead(t *testing.T) {
	failing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(failing.Close)
	for _, redact := range []bool{false, true} {
		cfg := &Config{
			Server: Server{RedactUpstream: redact},
			Providers: []Provider{
				{ID: "first", Kind: "compatible", Model: "m", BaseURL: failing.URL},
				{ID: "second", Kind: "compatible", Model: "m", BaseURL: failing.URL},
			},
			Routes: []Route{{Chain: []string{"first", "second"}}},
		}
		client, err := build(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { client.Close(context.Background()) })
		result, err := gooseTools(client, cfg, nil)[1].Call(context.Background(), json.RawMessage(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		var structured struct {
			Notes []string `json:"notes"`
		}
		if err := json.Unmarshal(result.Structured, &structured); err != nil {
			t.Fatal(err)
		}
		if len(structured.Notes) != 2 || !strings.HasPrefix(structured.Notes[0], "first: ") || !strings.HasPrefix(structured.Notes[1], "second: ") {
			t.Fatalf("redact = %v: notes = %q: which backend failed went unsaid", redact, structured.Notes)
		}
		if text := llm.TextOf(result.Content); !strings.Contains(text, "first: ") || !strings.Contains(text, "second: ") {
			t.Fatalf("redact = %v: text = %q", redact, text)
		}
	}
}

// What redact_upstream_errors keeps from the caller is still the operator's
// to read, in the debug log, as the listener keeps it.
func TestAskLogsWhatItHoldsBack(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"quota gone ZZSECRET","type":"invalid_request_error"}}`)
	}))
	t.Cleanup(upstream.Close)
	cfg := &Config{
		Server:    Server{RedactUpstream: true},
		Providers: []Provider{{ID: "local", Kind: "compatible", Model: "m", BaseURL: upstream.URL}},
	}
	client, err := build(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Close(context.Background()) })
	var log strings.Builder
	logger := slog.New(slog.NewTextHandler(&log, &slog.HandlerOptions{Level: slog.LevelDebug}))

	tools := logged(gooseTools(client, cfg, logger), logger)
	result, err := tools[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError || strings.Contains(llm.TextOf(result.Content), "ZZSECRET") {
		t.Fatalf("result = %+v", result)
	}
	said := log.String()
	if !strings.Contains(said, "ZZSECRET") || !strings.Contains(said, "provider=local") {
		t.Fatalf("log = %q: the reason held back from the caller was recorded nowhere", said)
	}
	if !strings.Contains(said, "tool call") || !strings.Contains(said, "tool=ask") || !strings.Contains(said, "failed=true") {
		t.Fatalf("log = %q: the call itself went unlogged", said)
	}
}

func TestModelsListsWhatTheProxyRoutesTo(t *testing.T) {
	client := clientFor(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"object":"list","data":[{"id":"m","object":"model","owned_by":"local"}]}`)
	})
	result, err := gooseTools(client, &Config{}, nil)[1].Call(context.Background(), json.RawMessage(`{}`))
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
	result, err := gooseTools(client, &Config{}, nil)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
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
	result, err := gooseTools(client, &Config{}, nil)[0].Call(context.Background(), json.RawMessage(`{"prompt":"hi"}`))
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
	result, err := gooseTools(client, &Config{}, nil)[0].Call(ctx, json.RawMessage(`{"prompt":"hi"}`))
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
