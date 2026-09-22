// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package azure

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// recorder answers every request with a minimal valid body and keeps the one
// it was asked, so a provider built the way an operator builds it can be asked
// where it would actually have gone.
type recorder struct {
	seen  *http.Request
	body  []byte
	reply string
}

func (r *recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	r.seen = req
	if req.Body != nil {
		r.body, _ = io.ReadAll(req.Body)
	}
	reply := r.reply
	if reply == "" {
		reply = `{"id":"c","model":"m","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	}
	return &http.Response{
		StatusCode: 200, Header: http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(reply)), Request: req,
	}, nil
}

func asked(t *testing.T, o Options, a *llm.Attempt, reply string) (*http.Request, []byte) {
	t.Helper()
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	rec := &recorder{reply: reply}
	if err := p.Bind(llm.Env{HTTP: &http.Client{Transport: rec}}); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	return rec.seen, rec.body
}

func turn(model, format string) *llm.Attempt {
	return &llm.Attempt{
		Model: model, Format: format, Recorder: llm.NopRecorder{},
		Request: &llm.Request{Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	}
}

// An operator names a resource, not a URL. What that resource turns into is
// what the backend actually receives, and no test that hands the provider a
// base URL of its own can say whether it is right.
func TestWhereAResourceActuallySends(t *testing.T) {
	const responses = `{"id":"r","model":"m","status":"completed","output":[]}`
	for _, tc := range []struct {
		name  string
		opts  Options
		a     *llm.Attempt
		want  string
		reply string
	}{
		{
			name: "the rolling surface",
			opts: Options{Resource: "my-resource", APIKey: "k"},
			a:    turn("gpt-4o", ""),
			want: "https://my-resource.openai.azure.com/openai/v1/chat/completions",
		},
		{
			name: "a dated surface names its deployment in the path",
			opts: Options{Resource: "my-resource", APIKey: "k", Deployment: "prod", APIVersion: "2024-10-21"},
			a:    turn("gpt-4o", ""),
			want: "https://my-resource.openai.azure.com/openai/deployments/prod/chat/completions?api-version=2024-10-21",
		},
		{
			name:  "and names it in the body for Responses",
			opts:  Options{Resource: "my-resource", APIKey: "k", Deployment: "prod", APIVersion: "2025-03-01-preview"},
			a:     turn("gpt-4o", "openai-responses"),
			want:  "https://my-resource.openai.azure.com/openai/responses?api-version=2025-03-01-preview",
			reply: responses,
		},
		{
			name:  "the rolling surface serves Responses under v1",
			opts:  Options{Resource: "my-resource", APIKey: "k"},
			a:     turn("gpt-4o", "openai-responses"),
			want:  "https://my-resource.openai.azure.com/openai/v1/responses",
			reply: responses,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req, body := asked(t, tc.opts, tc.a, tc.reply)
			if got := req.URL.String(); got != tc.want {
				t.Fatalf("asked %q, want %q", got, tc.want)
			}
			if req.Header.Get("api-key") != "k" {
				t.Errorf("this host takes its own header: %v", req.Header)
			}
			if tc.a.Format != "openai-responses" {
				return
			}
			var sent struct {
				Model string `json:"model"`
			}
			if err := json.Unmarshal(body, &sent); err != nil {
				t.Fatal(err)
			}
			want := "gpt-4o"
			if tc.opts.APIVersion != "" && tc.opts.APIVersion != "v1" {
				want = "prod"
			}
			if sent.Model != want {
				t.Fatalf("body model = %q, want %q", sent.Model, want)
			}
		})
	}
}
