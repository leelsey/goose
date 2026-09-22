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
	"github.com/leelsey/goose/provider/ollama"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/ollamaapi"
)

func ollamaProxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","created_at":"2024-01-01T00:00:00Z",
			"message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop",
			"prompt_eval_count":1,"eval_count":1}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := ollama.New(ollama.Options{ID: "up", BaseURL: upstream.URL})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{ollamaapi.NewChat(ollamaapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func FuzzOllamaRequestSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}`,
		`{"model":"m","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],"stream":false}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"temperature":0.5,"top_p":0.9,"top_k":40,"seed":7,"num_predict":64,"stop":["a"]}}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false,"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"}}}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{}}}]},{"role":"tool","content":"42"}],"stream":false}`,
		`{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false,"think":true}`,
	} {
		f.Add(seed)
	}
	front, seen := ollamaProxy(f)
	f.Fuzz(func(t *testing.T, body string) {
		if !json.Valid([]byte(body)) {
			return
		}
		resp, err := front.Client().Post(front.URL+"/api/chat", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compareOllama(t, body, *seen, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compareOllama(t testing.TB, sent string, got []byte, reported string) {
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

	wantOpts, _ := in["options"].(map[string]any)
	haveOpts, _ := out["options"].(map[string]any)
	lower(wantOpts)
	lower(haveOpts)
	for _, field := range []string{"temperature", "top_p", "top_k", "seed", "num_predict", "stop"} {
		want, had := wantOpts[field]
		if !had || empty(want) {
			continue
		}
		if have, kept := haveOpts[field]; !kept || !reflect.DeepEqual(want, have) {
			fail("options."+field, want, have)
		}
	}
	if names := toolNames(in["tools"]); len(names) > 0 {
		if have := toolNames(out["tools"]); !reflect.DeepEqual(names, have) {
			fail("tools", names, have)
		}
	}
	if want, have := roles(in["messages"]), roles(out["messages"]); !reflect.DeepEqual(want, have) {
		fail("messages", want, have)
	}
}

func TestOllamaNativeOptionsReachTheBackend(t *testing.T) {
	front, seen := ollamaProxy(t)
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false,
		"options":{"temperature":0.5,"num_ctx":8192,"repeat_penalty":1.2,"mirostat":2}}`
	resp, err := front.Client().Post(front.URL+"/api/chat", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	var out struct {
		Options map[string]any `json:"options"`
	}
	if err := json.Unmarshal(*seen, &out); err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]float64{"num_ctx": 8192, "repeat_penalty": 1.2, "mirostat": 2, "temperature": 0.5} {
		if got, ok := out.Options[k].(float64); !ok || got != want {
			t.Fatalf("options[%q] = %v, want %v (sent %s)", k, out.Options[k], want, *seen)
		}
	}
}
