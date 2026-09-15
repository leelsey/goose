// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/openairesp"
)

func responsesProxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed",
			"output":[{"type":"message","role":"assistant","status":"completed",
			"content":[{"type":"output_text","text":"ok"}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := openai.New(openai.Options{
		ID: "up", BaseURL: upstream.URL + "/v1", APIKey: "k", Dialect: openai.OpenAIDialect(),
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openairesp.New(openairesp.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func FuzzResponsesRequestSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"model":"m","input":"hi"}`,
		`{"model":"m","input":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}],"temperature":0.5,"top_p":0.9,"max_output_tokens":64}`,
		`{"model":"m","input":"hi","instructions":"be brief"}`,
		`{"model":"m","input":"hi","tools":[{"type":"function","name":"f","description":"d","parameters":{"type":"object"}}]}`,
		`{"model":"m","input":"hi","reasoning":{"effort":"high"}}`,
		`{"model":"m","input":[{"type":"function_call","call_id":"1","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"1","output":"42"}]}`,
		`{"model":"m","input":"hi","text":{"format":{"type":"json_schema","name":"s","schema":{"type":"object"}}}}`,
	} {
		f.Add(seed)
	}
	front, seen := responsesProxy(f)
	f.Fuzz(func(t *testing.T, body string) {
		if !json.Valid([]byte(body)) {
			return
		}
		resp, err := front.Client().Post(front.URL+"/v1/responses", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compareResponses(t, body, *seen, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compareResponses(t testing.TB, sent string, got []byte, reported string) {
	t.Helper()
	var in, out map[string]any
	if json.Unmarshal([]byte(sent), &in) != nil || json.Unmarshal(got, &out) != nil {
		return
	}
	fail := func(what string, want, have any) {
		t.Helper()
		if reported == "" {
			t.Fatalf("%s changed and nothing said so\n sent: %v\n  got: %v\n from: %s", what, want, have, sent)
		}
	}
	for _, field := range []string{"temperature", "top_p", "max_output_tokens"} {
		want, had := in[field]
		if !had || empty(want) {
			continue
		}
		if have, kept := out[field]; !kept || !reflect.DeepEqual(want, have) {
			fail(field, want, have)
		}
	}
	if names := responsesToolNames(in["tools"]); len(names) > 0 {
		if have := responsesToolNames(out["tools"]); !reflect.DeepEqual(names, have) {
			fail("tools", names, have)
		}
	}
	if want, ok := in["instructions"].(string); ok && want != "" {
		if have, _ := out["instructions"].(string); have != want {
			fail("instructions", want, have)
		}
	}
	if want, have := itemKinds(in["input"]), itemKinds(out["input"]); !reflect.DeepEqual(want, have) {
		fail("input", want, have)
	}
}

func responsesToolNames(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, entry := range list {
		tool, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		if name, ok := tool["name"].(string); ok && name != "" {
			out = append(out, name)
		}
	}
	return out
}

func itemKinds(v any) []string {
	switch input := v.(type) {
	case string:
		return []string{"message"}
	case []any:
		var out []string
		for _, entry := range input {
			m, ok := entry.(map[string]any)
			if !ok {
				continue
			}
			kind, _ := m["type"].(string)
			if kind == "" {
				kind = "message"
			}
			out = append(out, kind)
		}
		return out
	}
	return nil
}

func TestResponsesNativeObjectsReachTheBackend(t *testing.T) {
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"m","status":"completed",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
	}))
	defer upstream.Close()

	p, err := openai.New(openai.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k",
		Dialect: openai.Dialect{Responses: true, Extensions: true}})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openairesp.New(openairesp.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	defer front.Close()

	resp, err := front.Client().Post(front.URL+"/v1/responses", "application/json", strings.NewReader(
		`{"model":"m","input":"hi","reasoning":{"effort":"high","summary":"detailed"},"text":{"verbosity":"low"}}`))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	var out struct {
		Reasoning map[string]any `json:"reasoning"`
		Text      map[string]any `json:"text"`
	}
	if err := json.Unmarshal(seen, &out); err != nil {
		t.Fatalf("%v: %s", err, seen)
	}
	if out.Reasoning["summary"] != "detailed" {
		t.Fatalf("reasoning = %v: the client asked for a detailed summary", out.Reasoning)
	}
	if out.Reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %v", out.Reasoning)
	}
	if out.Text["verbosity"] != "low" {
		t.Fatalf("text = %v: a knob this codec does not model still belongs to the vendor", out.Text)
	}
}
