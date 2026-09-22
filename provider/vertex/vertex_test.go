// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package vertex

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/cloudauth"
	"github.com/leelsey/goose/llm"
)

func serve(t *testing.T, o Options, h http.HandlerFunc) (llm.Provider, *[]*http.Request, *[]string) {
	t.Helper()
	var reqs []*http.Request
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		reqs = append(reqs, r)
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	o.BaseURL = srv.URL
	o.Project = "proj"
	o.Region = "us-east5"
	o.Token = cloudauth.StaticToken("tok")
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	return p, &reqs, &bodies
}

func attempt(model string) *llm.Attempt {
	return &llm.Attempt{
		Request:  &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
		Model:    model,
		Recorder: llm.NopRecorder{},
	}
}

func TestPublisherComesFromTheModelName(t *testing.T) {
	cases := map[string]string{
		"claude-sonnet-4@20250514": "anthropic",
		"gemini-2.5-pro":           "google",
		"llama-3.1-70b":            "meta",
		"mistral-large":            "mistral",
		"something-else":           "google",
	}
	for model, want := range cases {
		if got := Publisher(model); got != want {
			t.Fatalf("Publisher(%q) = %q, want %q", model, got, want)
		}
	}
}

func TestGeminiGoesToTheProjectScopedURL(t *testing.T) {
	p, reqs, _ := serve(t, Options{Model: "gemini-2.5-pro"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3}}`))
	})
	if _, err := p.Complete(context.Background(), attempt("gemini-2.5-pro")); err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	want := "/v1/projects/proj/locations/us-east5/publishers/google/models/gemini-2.5-pro:generateContent"
	if r.URL.Path != want {
		t.Fatalf("path = %q, want %q", r.URL.Path, want)
	}
	if r.Header.Get("Authorization") != "Bearer tok" {
		t.Fatalf("authorization = %q", r.Header.Get("Authorization"))
	}
}

func TestClaudeUsesRawPredictAndDropsTheModelFromTheBody(t *testing.T) {
	p, reqs, bodies := serve(t, Options{Model: "claude-sonnet-4@20250514"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`))
	})
	resp, err := p.Complete(context.Background(), attempt("claude-sonnet-4@20250514"))
	if err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	if !strings.HasSuffix(r.URL.Path, "/publishers/anthropic/models/claude-sonnet-4@20250514:rawPredict") {
		t.Fatalf("path = %q", r.URL.Path)
	}
	body := (*bodies)[0]
	if strings.Contains(body, `"model"`) {
		t.Fatalf("body = %s: this host carries the model in the URL and rejects it in the body", body)
	}
	if !strings.Contains(body, `"anthropic_version":"vertex-2023-10-16"`) {
		t.Fatalf("body = %s", body)
	}
	if resp.Usage.Prompt != 5 || llm.TextOf(resp.Choices[0].Message.Content) != "ok" {
		t.Fatalf("response = %+v", resp)
	}
}

func TestClaudeStreamUsesStreamRawPredict(t *testing.T) {
	p, reqs, _ := serve(t, Options{Model: "claude-sonnet-4@20250514"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("event: message_stop\ndata: {}\n\n"))
	})
	s, err := p.Stream(context.Background(), attempt("claude-sonnet-4@20250514"))
	if err != nil {
		t.Fatal(err)
	}
	_ = s.Close()
	if !strings.HasSuffix((*reqs)[0].URL.Path, ":streamRawPredict") {
		t.Fatalf("path = %q", (*reqs)[0].URL.Path)
	}
}

func TestMissingProjectOrTokenIsRefused(t *testing.T) {
	if _, err := New(Options{Token: cloudauth.StaticToken("t")}); err == nil {
		t.Fatal("a project-scoped URL cannot be built without a project")
	}
	if _, err := New(Options{Project: "p"}); err == nil {
		t.Fatal("an unauthenticated request must not be attempted")
	}
}

// Model Garden sits beside the project rather than inside it, so the surface
// that answers a call is not the one that lists what it serves: asking the
// project-scoped path leaves every Vertex backend reading as unready.
func TestTheCatalogueComesFromModelGarden(t *testing.T) {
	for _, tc := range []struct{ name, model, publisher string }{
		{"gemini", "gemini-2.5-pro", "google"},
		{"claude", "claude-sonnet-4@20250514", "anthropic"},
		{"llama", "llama-3.1-70b", "meta"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var asked []string
			p, reqs, _ := serve(t, Options{Model: tc.model}, func(w http.ResponseWriter, r *http.Request) {
				asked = append(asked, r.URL.Path)
				want := "/v1beta1/publishers/" + tc.publisher + "/models"
				if r.URL.Path != want {
					w.WriteHeader(http.StatusNotFound)
					_, _ = w.Write([]byte(`{"error":{"code":404,"message":"not found"}}`))
					return
				}
				_, _ = w.Write([]byte(`{"publisherModels":[
					{"name":"publishers/` + tc.publisher + `/models/one"},
					{"name":"publishers/` + tc.publisher + `/models/two"}]}`))
			})
			_ = reqs

			lister, ok := p.(llm.Lister)
			if !ok {
				t.Fatal("a Vertex backend has a catalogue")
			}
			models, err := lister.Models(context.Background())
			if err != nil {
				t.Fatal(err)
			}
			if len(models) != 2 || models[0].ID != "one" || models[1].ID != "two" {
				t.Fatalf("models = %+v", models)
			}
			prober, ok := p.(llm.Prober)
			if !ok {
				t.Fatal("a Vertex backend can be probed")
			}
			if err := prober.Probe(context.Background()); err != nil {
				t.Fatalf("a reachable backend reads as unready: %v", err)
			}
			if _, ok := p.(llm.Passer); !ok {
				t.Fatal("a Vertex backend still forwards opaque requests")
			}
			for _, path := range asked {
				if !strings.HasPrefix(path, "/v1beta1/publishers/") {
					t.Fatalf("asked %q, which this host does not serve", path)
				}
			}
		})
	}
}
