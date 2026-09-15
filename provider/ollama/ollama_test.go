// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollama

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

func serve(t *testing.T, h http.HandlerFunc) (*Provider, *[]string) {
	t.Helper()
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	p, err := New(Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	return p, &bodies
}

func attempt(req *llm.Request) *llm.Attempt {
	return &llm.Attempt{Request: req, Model: "llama3.1:8b", Recorder: llm.NopRecorder{}}
}

func hello() *llm.Request {
	return &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}
}

func TestStreamCountersBecomeUsage(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		for _, line := range []string{
			`{"model":"llama3.1:8b","message":{"role":"assistant","content":"he"},"done":false}`,
			`{"model":"llama3.1:8b","message":{"role":"assistant","content":"llo"},"done":false}`,
			`{"model":"llama3.1:8b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":18,"eval_count":6,"total_duration":2000000}`,
		} {
			_, _ = w.Write([]byte(line + "\n"))
		}
	})
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := wirecore.Fold(s, wirecore.FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(resp.Choices[0].Message.Content) != "hello" {
		t.Fatalf("content = %q", llm.TextOf(resp.Choices[0].Message.Content))
	}
	if resp.Usage.Prompt != 18 || resp.Usage.Completion != 6 || resp.Usage.DurationMS != 2 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.Choices[0].Stop != llm.StopEnd {
		t.Fatalf("stop = %q", resp.Choices[0].Stop)
	}
}

func TestToolCallsAndThinkingSurvive(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","message":{"role":"assistant","content":"","thinking":"hmm",
			"tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},"done":true,"done_reason":"stop",
			"prompt_eval_count":5,"eval_count":2}`))
	})
	req := hello()
	req.Thinking = &llm.ThinkingConfig{Enabled: true}
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains((*bodies)[0], `"think":true`) {
		t.Fatalf("body = %s", (*bodies)[0])
	}
	parts := resp.Choices[0].Message.Content
	if th, ok := parts[0].(*llm.Thinking); !ok || th.Text != "hmm" {
		t.Fatalf("thinking = %+v", parts[0])
	}
	tu := llm.ToolUses(parts)
	if len(tu) != 1 || tu[0].Name != "f" {
		t.Fatalf("tool call = %+v", tu)
	}
	if resp.Choices[0].Stop != llm.StopToolUse {
		t.Fatalf("stop = %q: a turn that called a tool did not merely stop", resp.Choices[0].Stop)
	}
}

func TestMalformedFrameIsAnError(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("{\"model\":\"m\",\"message\":{\"content\":\"ok\"}}\nnot json\n"))
	})
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	_, ferr := wirecore.Fold(s, wirecore.FoldOptions{})
	if ferr == nil {
		t.Fatal("a frame that cannot be parsed must not be skipped in silence")
	}
}

func TestTheBackendsOwnTimestampSurvives(t *testing.T) {
	p, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","created_at":"2024-03-04T05:06:07Z","done":true,
			"done_reason":"stop","message":{"role":"assistant","content":"hi"},
			"prompt_eval_count":1,"eval_count":1}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Created.UTC().Format(time.RFC3339); got != "2024-03-04T05:06:07Z" {
		t.Fatalf("created = %s: the backend said when it made this", got)
	}

	quiet, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"model":"m","done":true,"done_reason":"stop",
			"message":{"role":"assistant","content":"hi"}}`))
	})
	resp, err = quiet.Complete(context.Background(), attempt(&llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Created.IsZero() {
		t.Fatal("a backend that said nothing still needs a stamp")
	}
}
