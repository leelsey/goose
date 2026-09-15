// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

//go:build conformance

// Package conformance runs goose against the real backends, behind a tag.
package conformance

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/ollama"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/provider/openrouter"
)

func env(t *testing.T, name string) string {
	t.Helper()
	v := os.Getenv(name)
	if v == "" {
		t.Skipf("%s is not set", name)
	}
	return v
}

func modelOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func backends() map[string]func(*testing.T) (*goose.Client, string) {
	return map[string]func(*testing.T) (*goose.Client, string){
		"anthropic": func(t *testing.T) (*goose.Client, string) {
			p, err := anthropic.New(anthropic.Options{ID: "anthropic", APIKey: env(t, "ANTHROPIC_API_KEY")})
			if err != nil {
				t.Fatal(err)
			}
			return client(t, p), modelOr("GOOSE_ANTHROPIC_MODEL", "claude-sonnet-5")
		},
		"openai": func(t *testing.T) (*goose.Client, string) {
			p, err := openai.New(openai.Options{
				ID: "openai", APIKey: env(t, "OPENAI_API_KEY"), Dialect: openai.OpenAIDialect(),
			})
			if err != nil {
				t.Fatal(err)
			}
			return client(t, p), modelOr("GOOSE_OPENAI_MODEL", "gpt-5")
		},
		"google": func(t *testing.T) (*goose.Client, string) {
			p, err := google.New(google.Options{ID: "google", APIKey: env(t, "GEMINI_API_KEY")})
			if err != nil {
				t.Fatal(err)
			}
			return client(t, p), modelOr("GOOSE_GOOGLE_MODEL", "gemini-2.5-flash")
		},
		"ollama": func(t *testing.T) (*goose.Client, string) {
			p, err := ollama.New(ollama.Options{ID: "ollama", BaseURL: env(t, "OLLAMA_HOST")})
			if err != nil {
				t.Fatal(err)
			}
			return client(t, p), modelOr("GOOSE_OLLAMA_MODEL", "llama3.1:8b")
		},
		"openrouter": func(t *testing.T) (*goose.Client, string) {
			p, err := openrouter.New(openrouter.Options{
				ID: "openrouter", APIKey: env(t, "OPENROUTER_API_KEY"),
			})
			if err != nil {
				t.Fatal(err)
			}
			return client(t, p), modelOr("GOOSE_OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
		},
	}
}

func client(t *testing.T, p llm.Provider) *goose.Client {
	t.Helper()
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close(context.Background()) })
	return c
}

var seams = []string{"answer", "stream", "tools", "refusal", "ready", "models"}

var reached = struct {
	sync.Mutex
	configured map[string]bool
	seams      map[string]map[string]bool
}{configured: map[string]bool{}, seams: map[string]map[string]bool{}}

func configured(backend string) {
	reached.Lock()
	defer reached.Unlock()
	reached.configured[backend] = true
}

func exercised(backend, seam string) {
	reached.Lock()
	defer reached.Unlock()
	if reached.seams[backend] == nil {
		reached.seams[backend] = map[string]bool{}
	}
	reached.seams[backend][seam] = true
}

func TestMain(m *testing.M) {
	code := m.Run()
	if summary := report(os.Stderr); code == 0 && summary != nil {
		fmt.Fprintln(os.Stderr, summary)
		code = 1
	}
	os.Exit(code)
}

func report(w io.Writer) error {
	reached.Lock()
	defer reached.Unlock()

	names := make([]string, 0, len(reached.configured))
	for name := range reached.configured {
		names = append(names, name)
	}
	sort.Strings(names)

	if len(names) == 0 {
		return fmt.Errorf("conformance: no backend was configured, so nothing was checked against a vendor; " +
			"set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY or OLLAMA_HOST")
	}

	fmt.Fprintln(w, "\nconformance: what a vendor actually agreed to")
	silent := make([]string, 0, len(names))
	for _, name := range names {
		var met, missed []string
		for _, seam := range seams {
			if reached.seams[name][seam] {
				met = append(met, seam)
				continue
			}
			missed = append(missed, seam)
		}
		fmt.Fprintf(w, "  %-12s met %v", name, met)
		if len(missed) > 0 {
			fmt.Fprintf(w, "  NOT EXERCISED %v", missed)
		}
		fmt.Fprintln(w)
		if len(met) == 0 {
			silent = append(silent, name)
		}
	}
	if len(silent) > 0 {
		return fmt.Errorf("conformance: %s was configured and verified nothing", strings.Join(silent, ", "))
	}
	return nil
}

func ask(model, text string) *llm.Request {
	return &llm.Request{
		Model:           model,
		MaxOutputTokens: llm.Ptr(16),
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: text}}}},
	}
}

func TestAnswerAndBill(t *testing.T) {
	for name, open := range backends() {
		t.Run(name, func(t *testing.T) {
			c, model := open(t)
			configured(name)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			resp, err := c.Chat(ctx, ask(model, "Reply with the single word: ready"))
			if err != nil {
				t.Fatalf("%v", err)
			}
			if len(resp.Choices) == 0 || llm.TextOf(resp.Choices[0].Message.Content) == "" {
				t.Fatalf("no text came back: %+v", resp.Choices)
			}
			if resp.Choices[0].Stop == llm.StopNone {
				t.Fatal("the answer does not say why it ended")
			}
			if !resp.Usage.Reported {
				t.Fatal("the call was not measured")
			}
			if resp.Usage.Prompt == 0 || resp.Usage.Completion == 0 {
				t.Fatalf("usage = %+v", resp.Usage)
			}
			if resp.Usage.Total != resp.Usage.Prompt+resp.Usage.Completion {
				t.Fatalf("total %d is not the columns it derives from: %+v", resp.Usage.Total, resp.Usage)
			}
			if resp.Provider != name {
				t.Fatalf("provider = %q", resp.Provider)
			}
			exercised(name, "answer")
			t.Logf("%s: %q stop=%s usage=%d/%d", name,
				llm.TextOf(resp.Choices[0].Message.Content), resp.Choices[0].Stop,
				resp.Usage.Prompt, resp.Usage.Completion)
		})
	}
}

func TestStreamSaysWhyItEnded(t *testing.T) {
	for name, open := range backends() {
		t.Run(name, func(t *testing.T) {
			c, model := open(t)
			configured(name)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			s, err := c.Stream(ctx, ask(model, "Count: one two three"))
			if err != nil {
				t.Fatalf("%v", err)
			}
			defer s.Close()

			var text strings.Builder
			var sawStop bool
			for f := range s.Frames() {
				text.WriteString(llm.TextOf(f.Delta))
				if f.Stop != llm.StopNone {
					sawStop = true
				}
			}
			if err := s.Err(); err != nil {
				t.Fatalf("stream failed: %v", err)
			}
			if !sawStop {
				t.Fatal("no frame said why the stream ended; the truncation check would report every stream from this backend")
			}
			if text.Len() == 0 {
				t.Fatal("no text was streamed")
			}
			if !s.Usage().Reported {
				t.Fatal("the streamed call was not measured")
			}
			exercised(name, "stream")
			t.Logf("%s: %d chars, usage=%d/%d", name, text.Len(), s.Usage().Prompt, s.Usage().Completion)
		})
	}
}

func TestToolCallRoundTrip(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)
	for name, open := range backends() {
		t.Run(name, func(t *testing.T) {
			c, model := open(t)
			configured(name)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			req := ask(model, "What is the weather in Seoul? Use the tool.")
			req.MaxOutputTokens = llm.Ptr(256)
			req.Tools = []llm.Tool{{Name: "get_weather", Description: "Current weather for a city", Schema: schema}}

			resp, err := c.Chat(ctx, req)
			if err != nil {
				t.Fatalf("%v", err)
			}
			var call *llm.ToolUse
			for _, p := range resp.Choices[0].Message.Content {
				if t, ok := p.(*llm.ToolUse); ok {
					call = t
				}
			}
			if call == nil {
				t.Skipf("the model answered without calling the tool: %q",
					llm.TextOf(resp.Choices[0].Message.Content))
			}
			if call.Name != "get_weather" {
				t.Fatalf("called %q", call.Name)
			}
			if !json.Valid(call.Args) {
				t.Fatalf("arguments did not survive as JSON: %q", call.Args)
			}
			exercised(name, "tools")
			t.Logf("%s: %s(%s) stop=%s", name, call.Name, call.Args, resp.Choices[0].Stop)
		})
	}
}

func TestARefusedCallIsTyped(t *testing.T) {
	for name, open := range backends() {
		t.Run(name, func(t *testing.T) {
			c, _ := open(t)
			configured(name)
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()

			_, err := c.Chat(ctx, ask("a-model-that-does-not-exist-anywhere", "hi"))
			if err == nil {
				t.Skip("the backend served a model that should not exist")
			}
			e := llm.AsError(err)
			if e.Status == 0 {
				t.Fatalf("no status on %v", e)
			}
			if strings.HasPrefix(strings.TrimSpace(e.Message), "{") {
				t.Fatalf("the message is a body, not a sentence: %q", e.Message)
			}
			exercised(name, "refusal")
			t.Logf("%s: %d %s: %s", name, e.Status, e.Kind, e.Message)
		})
	}
}

func TestReadinessAnswers(t *testing.T) {
	for name, open := range backends() {
		t.Run(name, func(t *testing.T) {
			c, _ := open(t)
			configured(name)
			ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			health, ok := c.Ready(ctx)
			if !ok {
				t.Fatalf("not ready: %+v", health)
			}
			exercised(name, "ready")
			models, diags := c.Models(ctx)
			if len(models) == 0 {
				t.Logf("%s lists no models: %v", name, diags.Codes())
				return
			}
			t.Logf("%s lists %d models, first %q", name, len(models), models[0].ID)
			exercised(name, "models")
		})
	}
}
