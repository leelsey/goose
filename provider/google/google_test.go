// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

func serve(t *testing.T, h http.HandlerFunc) (*Provider, *[]string, *[]string) {
	t.Helper()
	var bodies, urls []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		urls = append(urls, r.URL.String())
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	p, err := New(Options{BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	return p, &bodies, &urls
}

func attempt(req *llm.Request) *llm.Attempt {
	return &llm.Attempt{Request: req, Model: "gemini-2.5-pro", Recorder: llm.NopRecorder{}}
}

func hello() *llm.Request {
	return &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}
}

func TestUsageConventionsAreNormalised(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":1000,"candidatesTokenCount":30,"thoughtsTokenCount":70,
			"cachedContentTokenCount":400,"toolUsePromptTokenCount":50}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Prompt != 1050 {
		t.Fatalf("prompt = %d: the tool prompt adds, the cache is a breakdown", resp.Usage.Prompt)
	}
	if resp.Usage.Completion != 100 || resp.Usage.Reasoning != 70 {
		t.Fatalf("completion = %d reasoning = %d: thinking sits beside the candidate count here", resp.Usage.Completion, resp.Usage.Reasoning)
	}
}

func TestPromptBlockIsNotAnEmptySuccess(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":9}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Block == nil || resp.Block.Reason != "SAFETY" || resp.Stop != llm.StopContentFilter {
		t.Fatalf("response = %+v: a blocked prompt has no candidate and must say why", resp)
	}
	if resp.Usage.Prompt != 9 {
		t.Fatalf("usage = %+v: a blocked prompt was still read and billed", resp.Usage)
	}
}

func TestStreamingUsesSSEAndKeepsUsageFromCandidatelessFrames(t *testing.T) {
	p, _, urls := serve(t, func(w http.ResponseWriter, r *http.Request) {
		for _, c := range []string{
			`{"candidates":[{"content":{"role":"model","parts":[{"text":"one"}]},"index":0}]}`,
			`{"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":3}}`,
			`{"candidates":[{"content":{"role":"model","parts":[]},"finishReason":"STOP","index":0}]}`,
		} {
			_, _ = w.Write([]byte("data: " + c + "\n\n"))
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
	if !strings.Contains((*urls)[0], "alt=sse") {
		t.Fatalf("url = %s: the transport is chosen in the query", (*urls)[0])
	}
	if resp.Usage.Prompt != 12 || resp.Usage.Completion != 3 {
		t.Fatalf("usage = %+v: a frame with no candidate still carries the figures", resp.Usage)
	}
	if llm.TextOf(resp.Choices[0].Message.Content) != "one" {
		t.Fatalf("content = %+v", resp.Choices[0].Message.Content)
	}
}

const answered = `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`

func TestConsecutiveUserTurnsCoalesce(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(answered))
	})
	_, err := p.Complete(context.Background(), attempt(&llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "one"}}},
		{Role: llm.RoleTool, Content: []llm.Part{&llm.ToolResult{Name: "f", Content: []llm.Part{&llm.Text{Text: "{}"}}}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Contents []struct {
			Role  string            `json:"role"`
			Parts []json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Contents) != 2 || len(sent.Contents[0].Parts) != 2 {
		t.Fatalf("contents = %+v: roles must alternate strictly", sent.Contents)
	}
}

func TestCredentialGoesInTheHeaderNotTheQuery(t *testing.T) {
	var seen *http.Request
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		seen = r
		_, _ = w.Write([]byte(answered))
	})
	if _, err := p.Complete(context.Background(), attempt(hello())); err != nil {
		t.Fatal(err)
	}
	if seen.Header.Get("x-goog-api-key") != "k" {
		t.Fatal("the key must travel in a header")
	}
	if strings.Contains(seen.URL.RawQuery, "key=") {
		t.Fatal("a credential in a URL is logged by every hop between here and the backend")
	}
}

func TestPenaltiesReachGemini(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	})
	_, err := p.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{
			Model:            "m",
			Messages:         []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
			FrequencyPenalty: llm.Ptr(0.25),
			PresencePenalty:  llm.Ptr(-0.5),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[0]
	for _, want := range []string{`"frequencyPenalty":0.25`, `"presencePenalty":-0.5`} {
		if !strings.Contains(body, want) {
			t.Errorf("body is missing %s:\n%s", want, body)
		}
	}
}

func TestADynamicThinkingBudgetTravels(t *testing.T) {
	for _, tc := range []struct {
		thinking *llm.ThinkingConfig
		want     string
	}{
		{&llm.ThinkingConfig{Enabled: true, Budget: -1}, `"thinkingBudget":-1`},
		{&llm.ThinkingConfig{Enabled: true, Budget: 2048}, `"thinkingBudget":2048`},
		{&llm.ThinkingConfig{Enabled: false}, `"thinkingBudget":0`},
	} {
		p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
		})
		_, err := p.Complete(context.Background(), &llm.Attempt{
			Model: "m", Recorder: llm.NopRecorder{},
			Request: &llm.Request{
				Model:    "m",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
				Thinking: tc.thinking,
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		if got := (*bodies)[0]; !strings.Contains(got, tc.want) {
			t.Errorf("body is missing %s:\n%s", tc.want, got)
		}
	}
}

func TestCandidateMetadataIsNamedAsALoss(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},
			"finishReason":"STOP","groundingMetadata":{"webSearchQueries":["x"]},
			"safetyRatings":[{"category":"HARM_CATEGORY_HATE_SPEECH","probability":"NEGLIGIBLE"}]}]}`))
	})
	resp, err := p.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	found := resp.Diagnostics.Find(llm.DropExtraField)
	if len(found) == 0 {
		t.Fatalf("diagnostics = %v", resp.Diagnostics.Codes())
	}
	if !strings.Contains(found[0].Field, "groundingMetadata") || !strings.Contains(found[0].Field, "safetyRatings") {
		t.Errorf("field = %q", found[0].Field)
	}
	p2, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	})
	resp2, err := p2.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp2.Diagnostics.Has(llm.DropExtraField) {
		t.Errorf("a plain candidate was reported as a loss: %v", resp2.Diagnostics)
	}
}

func TestNoCandidateAndNoReasonIsAFailure(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"usageMetadata":{"promptTokenCount":3}}`))
	})
	_, err := p.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err == nil {
		t.Fatal("an empty answer was served as a success")
	}
	if e := llm.AsError(err); !e.Sent || !e.Retryable {
		t.Fatalf("error = %+v: the backend was paid and the next link may still serve it", e)
	}

	blocked, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"promptFeedback":{"blockReason":"SAFETY"}}`))
	})
	resp, err := blocked.Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatalf("a blocked prompt is an outcome, not a failure: %v", err)
	}
	if resp.Block == nil || resp.Block.Reason != "SAFETY" {
		t.Fatalf("block = %+v", resp.Block)
	}
}

func TestTheBackendsResponseIdSurvives(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"responseId":"abc-123","modelVersion":"gemini-test",
			"candidates":[{"content":{"role":"model","parts":[{"text":"hi"}]},"finishReason":"STOP"}]}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model: "gemini-test", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.ID != "abc-123" {
		t.Fatalf("id = %q: the backend named its own answer", resp.ID)
	}
}

// The model names the collection it is served from, and a name that already
// carries a publisher must not have the slash escaped into the model id.
func TestTheModelPathNamesItsCollection(t *testing.T) {
	for _, tc := range []struct{ model, want string }{
		{"gemini-2.5-pro", "models/gemini-2.5-pro"},
		{"models/gemini-2.5-pro", "models/gemini-2.5-pro"},
		{"google/gemini-2.5-pro", "models/gemini-2.5-pro"},
		{"publishers/google/models/gemini-2.5-pro", "models/gemini-2.5-pro"},
		{"tunedModels/mine", "tunedModels/mine"},
		{"tunedModels/", "models/"},
		{"a b", "models/a%20b"},
	} {
		if got := modelPath(tc.model); got != tc.want {
			t.Errorf("modelPath(%q) = %q, want %q", tc.model, got, tc.want)
		}
	}
}

// The path is what the backend is actually asked for, so the rule about which
// collection a model sits in has to reach it.
func TestTheRequestPathNamesTheCollection(t *testing.T) {
	for _, tc := range []struct{ model, want string }{
		{"gemini-2.5-pro", "/v1beta/models/gemini-2.5-pro:generateContent"},
		{"google/gemini-2.5-pro", "/v1beta/models/gemini-2.5-pro:generateContent"},
		{"tunedModels/mine", "/v1beta/tunedModels/mine:generateContent"},
	} {
		var path string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			path = r.URL.Path
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},` +
				`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3}}`))
		}))
		p, err := New(Options{ID: "g", BaseURL: srv.URL, APIKey: "k"})
		if err != nil {
			t.Fatal(err)
		}
		_, err = p.Complete(context.Background(), &llm.Attempt{
			Model: tc.model, Recorder: llm.NopRecorder{},
			Request: &llm.Request{Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			}},
		})
		srv.Close()
		if err != nil {
			t.Fatalf("%s: %v", tc.model, err)
		}
		if path != tc.want {
			t.Errorf("model %q asked for %q, want %q", tc.model, path, tc.want)
		}
	}
}
