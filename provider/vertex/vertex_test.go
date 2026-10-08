// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package vertex

import (
	"context"
	"encoding/json"
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

// Vertex refuses a model it will not serve this project or region with the 400
// a malformed request gets. Only a malformed request fails the same on every
// backend, so the chain walks on past the rest, whichever publisher it was.
func TestAModelVertexWillNotServeIsNotAMalformedRequest(t *testing.T) {
	for _, model := range []string{"claude-x@1", "gemini-x", "llama-x"} {
		for message, want := range map[string]llm.ErrorKind{
			"Project `proj` is not allowed to use Publisher Model `projects/proj/locations/us-east5/publishers/p/models/" + model + "`": llm.KindNotFound,
			"Publisher Model `projects/proj/locations/us-east5/publishers/p/models/" + model + "` not found.":                           llm.KindNotFound,
			"Invalid JSON payload received.": llm.KindInvalidRequest,
		} {
			p, _, _ := serve(t, Options{Model: model}, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"code":400,"message":"` + message + `","status":"FAILED_PRECONDITION"}}`))
			})
			_, err := p.Complete(context.Background(), attempt(model))
			if e := llm.AsError(err); e.Kind != want {
				t.Errorf("%s unary %q: err = %+v, want %s", model, message, e, want)
			}
			_, err = p.Stream(context.Background(), attempt(model))
			if e := llm.AsError(err); e.Kind != want {
				t.Errorf("%s stream %q: err = %+v, want %s", model, message, e, want)
			}
		}
	}
}

// Vertex takes only some of Anthropic's betas and refuses a request naming
// another, so a caller's first-party flags, and the fields they open, stay
// here and the loss is reported.
func TestTheCallersBetasDoNotReachVertex(t *testing.T) {
	p, reqs, bodies := serve(t, Options{Model: "claude-sonnet-4@20250514"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"claude","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`))
	})
	a := attempt("claude-sonnet-4@20250514")
	a.Request.Betas = []string{"context-management-2025-06-27"}
	a.Request.Extra = map[string]json.RawMessage{"context_management": json.RawMessage(`{}`)}
	a.Request.ExtraFrom = "anthropic"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if got := (*reqs)[0].Header.Values("anthropic-beta"); len(got) != 0 {
		t.Fatalf("anthropic-beta = %v", got)
	}
	if strings.Contains((*bodies)[0], "context_management") || !resp.Diagnostics.Has(llm.DropExtraField) {
		t.Fatalf("body = %s, diagnostics = %v", (*bodies)[0], resp.Diagnostics.Codes())
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

// The region is spliced into the host and the path a call and its token go
// to, so one that could end the host early or name another is refused at once.
func TestARegionThatCouldMoveTheHostIsRefused(t *testing.T) {
	for _, region := range []string{"127.0.0.1:18409/#", "x@127.0.0.1/#", "europe-west4.evil.example", "a/b"} {
		if _, err := New(Options{Project: "p", Model: "gemini-2.5-pro", Region: region, Token: cloudauth.StaticToken("t")}); err == nil {
			t.Fatalf("region %q was spliced into the host", region)
		}
	}
	for _, region := range []string{"global", "europe-west4", "us-central1"} {
		if _, err := New(Options{Project: "p", Model: "gemini-2.5-pro", Region: region, Token: cloudauth.StaticToken("t")}); err != nil {
			t.Fatalf("region %q was refused: %v", region, err)
		}
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
