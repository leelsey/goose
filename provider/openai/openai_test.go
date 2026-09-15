// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

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

func serve(t *testing.T, o Options, handler http.HandlerFunc) (*Provider, *[]string, *[]string) {
	t.Helper()
	var bodies, paths []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		paths = append(paths, r.URL.Path)
		handler(w, r)
	}))
	t.Cleanup(srv.Close)
	o.BaseURL = srv.URL
	if o.APIKey == "" {
		o.APIKey = "k"
	}
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	return p, &bodies, &paths
}

func attempt(model string, req *llm.Request) *llm.Attempt {
	return &llm.Attempt{Request: req, Model: model, Recorder: llm.NopRecorder{}}
}

func hello() *llm.Request {
	return &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}
}

func TestCeilingSpellingFollowsTheModelFamily(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	})
	req := hello()
	req.MaxOutputTokens = llm.Ptr(64)
	for model, want := range map[string]string{
		"gpt-4o":                      "max_tokens",
		"gpt-5":                       "max_completion_tokens",
		"o3-mini":                     "max_completion_tokens",
		"publishers/openai/models/o1": "max_completion_tokens",
	} {
		if _, err := p.Complete(context.Background(), attempt(model, req)); err != nil {
			t.Fatal(err)
		}
		body := (*bodies)[len(*bodies)-1]
		if !strings.Contains(body, `"`+want+`":64`) {
			t.Fatalf("model %q sent %s, want %s", model, body, want)
		}
	}
}

func TestStreamAsksForUsageOnlyWhereItExists(t *testing.T) {
	for _, tc := range []struct {
		dialect Dialect
		want    bool
	}{{OpenAIDialect(), true}, {CompatibleDialect(), false}} {
		p, bodies, _ := serve(t, Options{Dialect: tc.dialect}, func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("data: [DONE]\n\n"))
		})
		s, err := p.Stream(context.Background(), attempt("m", hello()))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = wirecore.Fold(s, wirecore.FoldOptions{})
		got := strings.Contains((*bodies)[0], "include_usage")
		if got != tc.want {
			t.Fatalf("include_usage = %v, want %v (a backend that does not know the field rejects the request)", got, tc.want)
		}
	}
}

func TestUnmeasurableStreamIsReported(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: CompatibleDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	})
	var diags llm.Diagnostics
	a := attempt("m", hello())
	a.Recorder = &diags
	if _, err := p.Stream(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, d := range diags {
		if d.Code == llm.CapNoStreamUsage {
			found = true
		}
	}
	if !found {
		t.Fatalf("diagnostics = %+v: an unmeasurable backend must say so rather than book zero", diags)
	}
}

func TestErrorBodyInA200IsNotASuccess(t *testing.T) {
	p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"error":{"message":"upstream exploded","type":"server_error","code":"boom"}}`))
	})
	_, err := p.Complete(context.Background(), attempt("m", hello()))
	if err == nil {
		t.Fatal("a 200 carrying an error object must not decode as an empty success")
	}
	if !strings.Contains(llm.AsError(err).Message, "exploded") {
		t.Fatalf("err = %v", err)
	}
}

func TestStreamMergesUsageAndToolCalls(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		for _, chunk := range []string{
			`{"id":"c1","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			`{"id":"c1","choices":[],"usage":{"prompt_tokens":30,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":20}}}`,
		} {
			_, _ = w.Write([]byte("data: " + chunk + "\n\n"))
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	})
	s, err := p.Stream(context.Background(), attempt("m", hello()))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := wirecore.Fold(s, wirecore.FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	tu := llm.ToolUses(resp.Choices[0].Message.Content)
	if len(tu) != 1 || string(tu[0].Args) != `{"a":1}` {
		t.Fatalf("tool call = %+v", tu)
	}
	if resp.Usage.Prompt != 30 || resp.Usage.CacheRead != 20 || resp.Usage.Completion != 8 {
		t.Fatalf("usage = %+v: the trailing frame carries it after the finish", resp.Usage)
	}
	if resp.Choices[0].Stop != llm.StopToolUse {
		t.Fatalf("stop = %q", resp.Choices[0].Stop)
	}
}

func TestResponsesEndpointChosenByFormatAndModel(t *testing.T) {
	p, _, paths := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/responses") {
			_, _ = w.Write([]byte(`{"id":"resp_1","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":3,"output_tokens":1}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	})
	a := attempt("gpt-5", hello())
	a.Format = "openai-responses"
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix((*paths)[0], "/responses") {
		t.Fatalf("path = %s: the client called the Responses shape", (*paths)[0])
	}
	if _, err := p.Complete(context.Background(), attempt("gpt-image-1", hello())); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix((*paths)[1], "/responses") {
		t.Fatalf("path = %s: an image-only model exists only there", (*paths)[1])
	}
	if _, err := p.Complete(context.Background(), attempt("gpt-4o", hello())); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix((*paths)[2], "/chat/completions") {
		t.Fatalf("path = %s", (*paths)[2])
	}
}

func TestResponsesCarriesSessionAndReasoning(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"resp_9","model":"m","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
			"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thought"}],"encrypted_content":"enc"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"half"}]}],
			"usage":{"input_tokens":10,"output_tokens":5,"output_tokens_details":{"reasoning_tokens":3}}}`))
	})
	a := attempt("gpt-5", hello())
	a.Format = "openai-responses"
	a.Session = "resp_prev"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains((*bodies)[0], `"previous_response_id":"resp_prev"`) {
		t.Fatalf("body = %s", (*bodies)[0])
	}
	if resp.Session != "resp_9" {
		t.Fatalf("session = %q: the response id is the next turn's handle", resp.Session)
	}
	th, ok := resp.Choices[0].Message.Content[0].(*llm.Thinking)
	if !ok || th.Text != "thought" || th.Signature != "enc" {
		t.Fatalf("reasoning = %+v", resp.Choices[0].Message.Content[0])
	}
	if resp.Choices[0].Stop != llm.StopLength {
		t.Fatalf("stop = %q: an incomplete response is not a finished one", resp.Choices[0].Stop)
	}
	if resp.Usage.Reasoning != 3 || resp.Usage.Completion != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestRequestKeepsRefusalAndImages(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[]}`))
	})
	req := &llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "look"}, &llm.Image{MIME: "image/png", Data: []byte("hi")}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Refusal{Text: "no"}}},
	}}
	if _, err := p.Complete(context.Background(), attempt("m", req)); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	parts := sent.Messages[0]["content"].([]any)
	if len(parts) != 2 {
		t.Fatalf("content = %v", parts)
	}
	if sent.Messages[1]["refusal"] != "no" {
		t.Fatalf("refusal lost: %v", sent.Messages[1])
	}
}

func TestModelsAndProbe(t *testing.T) {
	p, _, paths := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-5","created":1700000000,"owned_by":"openai"},{"id":"gpt-4o"}]}`))
	})
	models, err := p.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 2 || models[0].ID != "gpt-5" || models[0].Owner != "openai" {
		t.Fatalf("models = %+v", models)
	}
	if models[1].Owner != "openai" {
		t.Fatalf("a model with no owner is still served by this provider: %+v", models[1])
	}
	if !strings.HasSuffix((*paths)[0], "/models") {
		t.Fatalf("path = %q", (*paths)[0])
	}
	if err := p.Probe(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestProbeReportsARefusedCredential(t *testing.T) {
	p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key"}}`))
	})
	err := p.Probe(context.Background())
	if err == nil {
		t.Fatal("a credential the backend refuses is not readiness")
	}
	if llm.AsError(err).Kind != llm.KindAuth {
		t.Fatalf("err = %v", err)
	}
}

func TestResponsesSendsTheToolChoice(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[]}`))
	})
	for _, tc := range []struct {
		choice *llm.ToolChoice
		want   string
	}{
		{&llm.ToolChoice{Mode: llm.ToolRequired}, `"tool_choice":"required"`},
		{&llm.ToolChoice{Mode: llm.ToolNone}, `"tool_choice":"none"`},
		{&llm.ToolChoice{Mode: llm.ToolSpecific, Name: "f"}, `"tool_choice":{"name":"f","type":"function"}`},
	} {
		a := attempt("m", &llm.Request{
			Model:      "m",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
			Tools:      []llm.Tool{{Name: "f"}},
			ToolChoice: tc.choice,
		})
		a.Format = "openai-responses"
		if _, err := p.Complete(context.Background(), a); err != nil {
			t.Fatal(err)
		}
		if got := (*bodies)[len(*bodies)-1]; !strings.Contains(got, tc.want) {
			t.Errorf("body is missing %s:\n%s", tc.want, got)
		}
	}
}

func TestResponsesForwardsAnUnmodelledField(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[]}`))
	})
	a := attempt("m", &llm.Request{
		Model:     "m",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		Extra:     map[string]json.RawMessage{"store": json.RawMessage(`false`)},
		ExtraFrom: "openai-responses",
	})
	a.Format = "openai-responses"
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	if got := (*bodies)[len(*bodies)-1]; !strings.Contains(got, `"store":false`) {
		t.Fatalf("a caller asking not to have its prompt retained was not asking anything:\n%s", got)
	}
}

func TestStreamedToolFragmentsWithoutAnIndexStayApart(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"1","function":{"name":"f","arguments":""}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"{\"a\":1}"}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"2","function":{"name":"g","arguments":""}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"function":{"arguments":"{\"b\":2}"}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			`[DONE]`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	s, err := p.Stream(context.Background(), attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	args := map[int]string{}
	names := map[int]string{}
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if tu, ok := part.(*llm.ToolUse); ok {
				i := 0
				if tu.Index != nil {
					i = *tu.Index
				}
				args[i] += string(tu.Args)
				if tu.Name != "" {
					names[i] = tu.Name
				}
			}
		}
	}
	if names[0] != "f" || args[0] != `{"a":1}` {
		t.Errorf("call 0 = %q %q", names[0], args[0])
	}
	if names[1] != "g" || args[1] != `{"b":2}` {
		t.Errorf("call 1 = %q %q", names[1], args[1])
	}
}

func TestAStreamFrameCarryingAWholeMessageIsRead(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"whole"},"finish_reason":"stop"}]}`,
			`[DONE]`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	s, err := p.Stream(context.Background(), attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	stopped := false
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if tx, ok := part.(*llm.Text); ok {
				text.WriteString(tx.Text)
			}
		}
		if f.Stop != llm.StopNone {
			stopped = true
		}
	}
	if text.String() != "whole" {
		t.Errorf("text = %q: the frame carried the answer and it was dropped", text.String())
	}
	if !stopped {
		t.Error("the finish reason was lost with it")
	}
}

func TestARepeatedToolIDContinuesTheSameCall(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"name":"f","arguments":"{\"a\":"}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","function":{"arguments":"1}"}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_2","function":{"name":"g","arguments":"{}"}}]}}]}`,
			`{"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			`[DONE]`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	s, err := p.Stream(context.Background(), attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	args := map[int]string{}
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if tu, ok := part.(*llm.ToolUse); ok && tu.Index != nil {
				args[*tu.Index] += string(tu.Args)
			}
		}
	}
	if len(args) != 2 {
		t.Fatalf("calls = %d: %v", len(args), args)
	}
	if args[0] != `{"a":1}` || args[1] != `{}` {
		t.Fatalf("arguments = %v", args)
	}
}

func TestTheBackendsOwnTimestampSurvives(t *testing.T) {
	p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","created":1709528767,
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	})
	resp, err := p.Complete(context.Background(), attempt("m", hello()))
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Created.UTC().Unix(); got != 1709528767 {
		t.Fatalf("created = %d: the backend said when it made this", got)
	}

	quiet, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c1","model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	})
	resp, err = quiet.Complete(context.Background(), attempt("m", hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Created.IsZero() {
		t.Fatal("a backend that said nothing still needs a stamp")
	}
}

func TestTheServedTierReachesTheLedger(t *testing.T) {
	p, _, _ := serve(t, Options{}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","service_tier":"flex",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":3,"completion_tokens":1}}`))
	})
	resp, err := p.Complete(context.Background(), attempt("m", hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Tier != "flex" {
		t.Fatalf("tier = %q: the backend said which one it served under", resp.Usage.Tier)
	}
}

func TestAFlatteningNamesTheBackendThatDidIt(t *testing.T) {
	p, _, _ := serve(t, Options{ID: "up"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"c1","model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	})
	req := &llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.ToolResult{
			ID: "t1", Content: []llm.Part{&llm.Text{Text: "ok"}, &llm.Image{MIME: "image/png", Data: []byte("x")}},
		}}},
	}}
	resp, err := p.Complete(context.Background(), attempt("m", req))
	if err != nil {
		t.Fatal(err)
	}
	named := false
	for _, d := range resp.Diagnostics {
		if d.Code == llm.DropImage {
			named = d.Provider == "up"
		}
	}
	if !named {
		t.Fatalf("diagnostics = %+v: a loss with no backend on it names nothing", resp.Diagnostics)
	}
}
