// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"encoding/base64"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

// vLLM and llama.cpp give the code of a failure as a number. The sentence
// beside it is the backend's all the same, and is what the caller needs.
func TestAnErrorWithANumericCodeKeepsItsMessage(t *testing.T) {
	const failure = `{"error":{"message":"guided decoding failed: bad grammar","type":"BadRequestError","code":400}}`

	t.Run("stream", func(t *testing.T) {
		p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte(`data: {"id":"c","model":"m","choices":[{"index":0,"delta":{"content":"par"}}]}` + "\n\n"))
			_, _ = w.Write([]byte("data: " + failure + "\n\n"))
		})
		s, err := p.Stream(context.Background(), attempt("m", hello()))
		if err != nil {
			t.Fatal(err)
		}
		_, ferr := wirecore.Fold(s, wirecore.FoldOptions{})
		e := llm.AsError(ferr)
		if e == nil || e.Code == llm.WireFrameMalformed || !strings.Contains(e.Message, "bad grammar") {
			t.Fatalf("err = %+v: the backend said why and it was lost", e)
		}
		if e.Code != "400" || e.Kind != llm.KindUpstream {
			t.Fatalf("err = %+v", e)
		}
	})

	t.Run("unary", func(t *testing.T) {
		p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(failure))
		})
		_, err := p.Complete(context.Background(), attempt("m", hello()))
		if e := llm.AsError(err); e == nil || !strings.Contains(e.Message, "bad grammar") {
			t.Fatalf("err = %v", err)
		}
	})
}

// OpenAI refuses to stream some models to an organisation it has not verified,
// with the 400 a malformed request gets; another backend may serve the request
// as it stands.
func TestAnUnverifiedOrganisationIsNotAMalformedRequest(t *testing.T) {
	for _, tc := range []struct {
		name, message string
		want          llm.ErrorKind
	}{
		{"unverified", "Your organization must be verified to stream this model. Please go to: https://platform.openai.com/settings/organization/general and click on Verify Organization.", llm.KindUnsupported},
		{"malformed", "Unsupported parameter: 'max_tokens' is not supported with this model.", llm.KindInvalidRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"message":"` + tc.message + `","type":"invalid_request_error","param":"stream","code":"unsupported_value"}}`))
			})
			_, err := p.Stream(context.Background(), attempt("o3", hello()))
			if e := llm.AsError(err); e.Kind != tc.want || e.Code != "unsupported_value" {
				t.Fatalf("err = %+v, want %s", e, tc.want)
			}
		})
	}
}

// A failure reported inside a 200 is the upstream's words as much as a
// refusal's body is, and the ledger is kept free of both alike.
func TestAFailureInsideA200IsRedactedLikeARefusal(t *testing.T) {
	const secret = "SECRET: your message 'the quoted prompt' was flagged"
	for _, tc := range []struct {
		name, format, body string
		stream             bool
	}{
		{name: "chat", body: `{"error":{"message":"` + secret + `","type":"server_error","code":"flagged"}}`},
		{name: "chat stream", stream: true,
			body: `data: {"error":{"message":"` + secret + `","type":"server_error","code":"flagged"}}` + "\n\n"},
		{name: "responses", format: "openai-responses",
			body: `{"id":"r","status":"failed","error":{"code":"flagged","message":"` + secret + `"}}`},
		{name: "responses error event", format: "openai-responses", stream: true,
			body: `data: {"type":"error","code":"flagged","message":"` + secret + `"}` + "\n\n"},
		{name: "responses failed", format: "openai-responses", stream: true,
			body: `data: {"type":"response.failed","response":{"id":"r","status":"failed","error":{"code":"flagged","message":"` + secret + `"}}}` + "\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(tc.body))
			})
			a := attempt("m", hello())
			a.Format = tc.format
			var err error
			if tc.stream {
				s, serr := p.Stream(context.Background(), a)
				if serr != nil {
					t.Fatal(serr)
				}
				_, err = wirecore.Fold(s, wirecore.FoldOptions{})
			} else {
				_, err = p.Complete(context.Background(), a)
			}
			e := llm.AsError(err)
			if e == nil || !strings.Contains(e.Message, "quoted prompt") {
				t.Fatalf("err = %v", err)
			}
			if got := e.Reason(true); got != string(e.Kind) {
				t.Fatalf("redacted reason = %q: the upstream's words reached the ledger", got)
			}
			if e.Detail(true) != "" || !strings.Contains(e.Detail(false), "quoted prompt") {
				t.Fatalf("detail = %q / %q", e.Detail(true), e.Detail(false))
			}
		})
	}
}

// A failed response carries the caller's instructions and what was generated
// before it failed. The ledger is told of the failure, not handed the rest.
func TestAFailedResponseKeepsOnlyItsError(t *testing.T) {
	const response = `{"id":"r","status":"failed","instructions":"SYSTEM-PROMPT-VERBATIM",
		"output":[{"type":"message","id":"m","role":"assistant","content":[{"type":"output_text","text":"PARTIAL-OUTPUT"}]}],
		"error":{"code":"server_error","message":"the model failed"}}`
	for _, tc := range []struct {
		name, body string
		stream     bool
	}{
		{name: "unary", body: response},
		{name: "stream", stream: true, body: `data: {"type":"response.failed","response":` + strings.ReplaceAll(response, "\n", "") + "}\n\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
				_, _ = w.Write([]byte(tc.body))
			})
			a := attempt("m", hello())
			a.Format = "openai-responses"
			var err error
			if tc.stream {
				s, serr := p.Stream(context.Background(), a)
				if serr != nil {
					t.Fatal(serr)
				}
				_, err = wirecore.Fold(s, wirecore.FoldOptions{})
			} else {
				_, err = p.Complete(context.Background(), a)
			}
			e := llm.AsError(err)
			if e == nil || e.Code != "server_error" || !strings.Contains(e.Detail(false), "the model failed") {
				t.Fatalf("err = %+v", e)
			}
			if d := e.Detail(false); strings.Contains(d, "SYSTEM-PROMPT") || strings.Contains(d, "PARTIAL-OUTPUT") {
				t.Fatalf("detail = %q: more than the failure reached the ledger", d)
			}
		})
	}
}

// A gateway that failed partway answers with both, and the note that says so
// is written to the ledger beside the answer.
func TestAnErrorBesideAnAnswerIsRedactedFromItsNote(t *testing.T) {
	p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"half"},"finish_reason":"stop"}],
			"error":{"message":"SECRET upstream words quoting the prompt"}}`))
	})
	for _, redact := range []bool{false, true} {
		a := attempt("m", hello())
		a.Redact = redact
		resp, err := p.Complete(context.Background(), a)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, d := range resp.Diagnostics {
			if d.Code != llm.WireErrorInBody {
				continue
			}
			found = true
			if strings.Contains(d.Detail, "SECRET") == redact {
				t.Errorf("redact=%v: detail = %q", redact, d.Detail)
			}
		}
		if !found {
			t.Fatalf("diagnostics = %+v: the answer is not the whole one and must say so", resp.Diagnostics)
		}
	}
}

// An image streamed on the Responses wire arrives as one event, commonly past
// a megabyte, and the frame ceiling is the embedder's to raise past that.
func TestAStreamedImageFitsTheFrameCeiling(t *testing.T) {
	for _, tc := range []struct {
		name  string
		size  int
		limit int
	}{
		{name: "default", size: 2 << 20},
		{name: "raised past the event default", size: 10 << 20, limit: 16 << 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			image := base64.StdEncoding.EncodeToString(make([]byte, tc.size))
			p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte(`data: {"type":"response.image_generation_call.partial_image","item_id":"ig","partial_image_index":0,"partial_image_b64":"` + image + `"}` + "\n\n"))
				_, _ = w.Write([]byte(`data: {"type":"response.completed","response":{"id":"r","status":"completed"}}` + "\n\n"))
			})
			a := attempt("gpt-image-1", hello())
			a.Format = "openai-responses"
			a.Limits.MaxFrameBytes = tc.limit
			s, err := p.Stream(context.Background(), a)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			got := 0
			for f := range s.Frames() {
				for _, part := range f.Delta {
					if img, ok := part.(*llm.Image); ok {
						got += len(img.Data)
					}
				}
			}
			if s.Err() != nil || got != tc.size {
				t.Fatalf("image bytes = %d, err = %v", got, s.Err())
			}
		})
	}
}
