// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package vertex

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/cloudauth"
	"github.com/leelsey/goose/llm"
)

type recorder struct {
	seen  []*http.Request
	reply string
}

func (r *recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	r.seen = append(r.seen, req)
	reply := r.reply
	if reply == "" {
		reply = `{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},` +
			`"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3}}`
	}
	return &http.Response{
		StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(reply)), Request: req,
	}, nil
}

func built(t *testing.T, o Options, reply string) (llm.Provider, *recorder) {
	t.Helper()
	o.Project, o.Token = "proj", cloudauth.StaticToken("tok")
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{reply: reply}
	if b, ok := p.(llm.Binder); ok {
		if err := b.Bind(llm.Env{HTTP: &http.Client{Transport: rec}}); err != nil {
			t.Fatal(err)
		}
	}
	return p, rec
}

func ask(t *testing.T, p llm.Provider, model string) {
	t.Helper()
	_, err := p.Complete(context.Background(), &llm.Attempt{
		Model: model, Recorder: llm.NopRecorder{},
		Request: &llm.Request{Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// An operator names a project and a region, not a URL. What those turn into is
// what Vertex actually receives, and a test that hands the provider a base URL
// of its own cannot say whether it is right.
func TestWhereAProjectAndRegionActuallySend(t *testing.T) {
	for _, tc := range []struct {
		name  string
		opts  Options
		model string
		want  string
		reply string
	}{
		{
			name:  "a regional endpoint takes the region as a prefix",
			opts:  Options{Region: "us-east5", Model: "gemini-2.5-pro"},
			model: "gemini-2.5-pro",
			want: "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5/" +
				"publishers/google/models/gemini-2.5-pro:generateContent",
		},
		{
			name:  "the global one is the bare host",
			opts:  Options{Region: "global", Model: "gemini-2.5-pro"},
			model: "gemini-2.5-pro",
			want: "https://aiplatform.googleapis.com/v1/projects/proj/locations/global/" +
				"publishers/google/models/gemini-2.5-pro:generateContent",
		},
		{
			name:  "the region left out takes the default",
			opts:  Options{Model: "gemini-2.5-pro"},
			model: "gemini-2.5-pro",
			want: "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/" +
				"publishers/google/models/gemini-2.5-pro:generateContent",
		},
		{
			name:  "a model written the way Vertex writes it names its publisher once",
			opts:  Options{Region: "us-east5", Model: "gemini-2.5-pro"},
			model: "google/gemini-2.5-pro",
			want: "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5/" +
				"publishers/google/models/gemini-2.5-pro:generateContent",
		},
		{
			name:  "anthropic is asked to predict rather than to generate",
			opts:  Options{Region: "us-east5", Model: "claude-sonnet-4@20250514"},
			model: "claude-sonnet-4@20250514",
			want: "https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5/" +
				"publishers/anthropic/models/claude-sonnet-4@20250514:rawPredict",
			reply: `{"id":"m","model":"c","content":[{"type":"text","text":"ok"}],` +
				`"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, rec := built(t, tc.opts, tc.reply)
			ask(t, p, tc.model)
			if len(rec.seen) != 1 {
				t.Fatalf("requests = %d", len(rec.seen))
			}
			if got := rec.seen[0].URL.String(); got != tc.want {
				t.Fatalf("asked\n  %q\nwant\n  %q", got, tc.want)
			}
			if rec.seen[0].Header.Get("Authorization") != "Bearer tok" {
				t.Errorf("authorization = %q", rec.seen[0].Header.Get("Authorization"))
			}
		})
	}
}

// Model Garden sits beside the project, not inside it, and the listing lives
// on a revision of its own.
func TestWhereTheCatalogueActuallyComesFrom(t *testing.T) {
	for _, tc := range []struct{ model, want string }{
		{"gemini-2.5-pro", "https://us-east5-aiplatform.googleapis.com/v1beta1/publishers/google/models?pageSize=200"},
		{"claude-sonnet-4@20250514", "https://us-east5-aiplatform.googleapis.com/v1beta1/publishers/anthropic/models?pageSize=200"},
		{"llama-3.1-70b", "https://us-east5-aiplatform.googleapis.com/v1beta1/publishers/meta/models?pageSize=200"},
	} {
		p, rec := built(t, Options{Region: "us-east5", Model: tc.model}, `{"publisherModels":[]}`)
		lister, ok := p.(llm.Lister)
		if !ok {
			t.Fatal("a Vertex backend has a catalogue")
		}
		if _, err := lister.Models(context.Background()); err != nil {
			t.Fatal(err)
		}
		if len(rec.seen) != 1 {
			t.Fatalf("%s: requests = %d", tc.model, len(rec.seen))
		}
		if got := rec.seen[0].URL.String(); got != tc.want {
			t.Errorf("%s asked\n  %q\nwant\n  %q", tc.model, got, tc.want)
		}
	}
}
