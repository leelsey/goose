// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package roundtrip checks a request and its answer across the whole stack.
package roundtrip

import (
	"bytes"
	"encoding/json"
	"fmt"
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
	"github.com/leelsey/goose/wire/openaichat"
)

func proxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","created":1,"model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := openai.New(openai.Options{
		ID: "up", BaseURL: upstream.URL + "/v1", APIKey: "k",
		Dialect: openai.OpenAIDialect(),
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func FuzzOpenAIRequestSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":0.5,"top_p":0.9,"max_tokens":64}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stop":["a","b"],"seed":7,"frequency_penalty":0.25,"presence_penalty":-0.5}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
		`{"model":"m","messages":[{"role":"assistant","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},{"role":"tool","tool_call_id":"1","content":"42"}],"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"a":{"type":"number"}},"required":["a"]}}}]}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}],"tool_choice":"required"}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"s","strict":true,"schema":{"type":"object"}}}}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}`,
	} {
		f.Add(seed)
	}
	front, seen := proxy(f)
	f.Fuzz(func(t *testing.T, body string) {
		if !json.Valid([]byte(body)) {
			return
		}
		resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compare(t, body, *seen, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compare(t testing.TB, sent string, got []byte, reported string) {
	t.Helper()
	in, okIn := foldRead([]byte(sent))
	out, okOut := foldRead(got)
	if !okIn || !okOut {
		return
	}
	normalise(in)
	normalise(out)

	fail := func(what string, want, have any) {
		t.Helper()
		if reported == "" {
			t.Fatalf("%s changed and nothing said so\n sent: %v\n  got: %v\n from: %s", what, want, have, sent)
		}
	}

	for _, field := range []string{
		"temperature", "top_p", "max_tokens", "seed", "stop",
		"frequency_penalty", "presence_penalty", "reasoning_effort",
	} {
		want, had := in[field]
		if !had || empty(want) {
			continue
		}
		if have, kept := out[field]; !kept || !reflect.DeepEqual(want, have) {
			fail(field, want, have)
		}
	}

	if names := toolNames(in["tools"]); len(names) > 0 {
		if have := toolNames(out["tools"]); !reflect.DeepEqual(names, have) {
			fail("tools", names, have)
		}
	}

	for _, field := range []string{"response_format", "tool_choice"} {
		want := in[field]
		if !asks(want) && !isString(want) {
			continue
		}
		if _, kept := out[field]; !kept {
			fail(field, want, nil)
		}
	}

	want, have := roles(in["messages"]), roles(out["messages"])
	if !reflect.DeepEqual(want, have) {
		fail("messages", want, have)
	}
}

func roles(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, entry := range list {
		m, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		out = append(out, role)
	}
	return out
}

// foldRead reads a request the way the decoder under test reads one: a field
// is matched without regard to the case it was written in, and where an object
// spells one field twice the last spelling is the one that lands. A map cannot
// say which of two spellings came last, so the object is read in the order it
// was written rather than folded afterwards.
func foldRead(raw []byte) (map[string]any, bool) {
	v, err := foldValue(json.NewDecoder(bytes.NewReader(raw)))
	if err != nil {
		return nil, false
	}
	body, ok := v.(map[string]any)
	return body, ok
}

// lower folds an object's keys after the fact, for a body already decoded and
// read only for shapes no wire spells twice. Where a field could be spelled
// twice in the one object, read the body with foldRead instead: which spelling
// landed is not something this can say.
func lower(m map[string]any) {
	for k, v := range m {
		if folded := strings.ToLower(k); folded != k {
			if _, taken := m[folded]; !taken {
				m[folded] = v
			}
			delete(m, k)
		}
	}
}

func foldValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, ok := tok.(json.Delim)
	if !ok {
		return tok, nil
	}
	switch delim {
	case '{':
		body := map[string]any{}
		for dec.More() {
			key, err := dec.Token()
			if err != nil {
				return nil, err
			}
			name, _ := key.(string)
			v, err := foldValue(dec)
			if err != nil {
				return nil, err
			}
			body[strings.ToLower(name)] = v
		}
		_, err := dec.Token()
		return body, err
	case '[':
		list := []any{}
		for dec.More() {
			v, err := foldValue(dec)
			if err != nil {
				return nil, err
			}
			list = append(list, v)
		}
		_, err := dec.Token()
		return list, err
	}
	return nil, fmt.Errorf("unexpected %v", delim)
}

func toolNames(v any) []string {
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
		fn, ok := tool["function"].(map[string]any)
		if !ok {
			continue
		}
		if name, ok := fn["name"].(string); ok {
			out = append(out, name)
		}
	}
	return out
}

func asks(v any) bool {
	m, ok := v.(map[string]any)
	if !ok || len(m) == 0 {
		return false
	}
	kind, _ := m["type"].(string)
	return kind != "" && kind != "disabled" && kind != "none"
}

func isString(v any) bool {
	s, ok := v.(string)
	return ok && s != "" && s != "none"
}

func empty(v any) bool {
	switch c := v.(type) {
	case nil:
		return true
	case string:
		return c == ""
	case []any:
		return len(c) == 0
	}
	return false
}

func normalise(body map[string]any) {
	// This wire takes a stop sequence as a string or as a list of them, and a
	// list holding the one string is that string. Reading it as the list it
	// also means is not the request changing under the caller. An empty one is
	// no stop sequence at all, and stays the empty thing it was so that the
	// check below still passes over it.
	if only, ok := body["stop"].(string); ok && only != "" {
		body["stop"] = []any{only}
	}
	msgs, ok := body["messages"].([]any)
	if !ok {
		return
	}
	for len(msgs) > 0 {
		m, ok := msgs[0].(map[string]any)
		if !ok || (m["role"] != "system" && m["role"] != "developer") || !empty(m["content"]) {
			break
		}
		msgs = msgs[1:]
	}
	body["messages"] = msgs
	known := map[string]bool{
		"role": true, "content": true, "name": true,
		"tool_calls": true, "tool_call_id": true, "refusal": true,
	}
	for _, entry := range msgs {
		m, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		for k := range m {
			if !known[k] {
				delete(m, k)
			}
		}
		if parts, ok := m["content"].([]any); ok && len(parts) == 1 {
			if part, ok := parts[0].(map[string]any); ok && part["type"] == "text" {
				m["content"] = part["text"]
			}
		}
		if empty(m["content"]) {
			delete(m, "content")
		}
	}
}

func TestAParticipantNameReachesTheBackend(t *testing.T) {
	front, seen := proxy(t)
	post(t, front, `{"model":"m","messages":[{"role":"user","name":"alice","content":"hi"}]}`)
	if !strings.Contains(string(*seen), `"name":"alice"`) {
		t.Fatalf("the name on a message went missing: %s", *seen)
	}
}

func TestAConversationThatRendersToNothingIsRefused(t *testing.T) {
	front, seen := proxy(t)
	for _, body := range []string{
		`{"model":"m","messages":[{"role":"system"}]}`,
		`{"model":"m","messages":[{"role":"system","content":""}]}`,
		`{"model":"m","messages":[]}`,
		`{"model":"m"}`,
	} {
		status, _ := post(t, front, body)
		if status == http.StatusOK {
			t.Errorf("%s served 200 having sent the backend %s", body, *seen)
		}
	}
}

func TestANamelessToolIsRefusedBeforeAnythingIsSpent(t *testing.T) {
	front, seen := proxy(t)
	*seen = nil
	status, body := post(t, front, `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"parameters":{}}}]}`)
	if status == http.StatusOK {
		t.Fatalf("served 200: %s", body)
	}
	if len(*seen) > 0 {
		t.Fatalf("the request reached the backend: %s", *seen)
	}
}

func post(t *testing.T, front *httptest.Server, body string) (int, string) {
	t.Helper()
	resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(out)
}

// The check is only worth what it reads the request as. Two spellings of one
// field are read the way the decoder under test reads them, and a stop
// sequence that is not one is not a change.
func TestTheCheckReadsARequestTheWayTheDecoderDoes(t *testing.T) {
	for _, tc := range []struct{ what, body string }{
		{"two spellings of a field", `{"messages":[{"role":"000000","Role":"user"}]}`},
		{"a stop sequence that is not one", `{"messages":[{"role":"user","content":"hi"}],"stop":""}`},
	} {
		t.Run(tc.what, func(t *testing.T) {
			front, seen := proxy(t)
			post(t, front, tc.body)
			compare(t, tc.body, *seen, "")
		})
	}
}

// An ordinary streamed call through the commonest provider and wire pair says
// nothing was lost, because nothing was.
func TestAnOrdinaryStreamReportsNoLoss(t *testing.T) {
	chunk := func(body string) string { return "data: " + body + "\n\n" }
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, b := range []string{
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"fp_9","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"fp_9","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","system_fingerprint":"fp_9","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		} {
			_, _ = w.Write([]byte(chunk(b)))
			w.(http.Flusher).Flush()
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer up.Close()

	p, err := openai.New(openai.Options{ID: "up", BaseURL: up.URL + "/v1", APIKey: "k", Dialect: openai.OpenAIDialect()})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{openaichat.New(openaichat.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	defer front.Close()

	resp, err := front.Client().Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if got := resp.Trailer.Get(httpx.DiagnosticsHeader); got != "" {
		t.Errorf("an ordinary stream reported %q\n%s", got, body)
	}
	if got := resp.Header.Get(httpx.DiagnosticsHeader); got != "" {
		t.Errorf("an ordinary stream reported %q\n%s", got, body)
	}
}
