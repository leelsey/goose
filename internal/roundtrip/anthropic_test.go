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
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

func anthropicProxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m",
			"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := anthropic.New(anthropic.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{anthropicmsg.New(anthropicmsg.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func FuzzAnthropicRequestSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","max_tokens":16,"system":"be brief","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"temperature":0.5,"top_p":0.9,"top_k":40}`,
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"stop_sequences":["a"]}`,
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","description":"d","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}`,
		`{"model":"m","max_tokens":16,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"1","name":"f","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"42"}]}]}`,
		`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"}],"thinking":{"type":"enabled","budget_tokens":1024}}`,
	} {
		f.Add(seed)
	}
	front, seen := anthropicProxy(f)
	f.Fuzz(func(t *testing.T, body string) {
		if !json.Valid([]byte(body)) {
			return
		}
		resp, err := front.Client().Post(front.URL+"/v1/messages", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compareAnthropic(t, body, *seen, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compareAnthropic(t testing.TB, sent string, got []byte, reported string) {
	t.Helper()
	in, okIn := foldRead([]byte(sent))
	out, okOut := foldRead(got)
	if !okIn || !okOut {
		return
	}

	fail := func(what string, want, have any) {
		t.Helper()
		if reported == "" {
			t.Fatalf("%s changed and nothing said so\n sent: %v\n  got: %v\n from: %s", what, want, have, sent)
		}
	}
	for _, field := range []string{"temperature", "top_p", "top_k", "stop_sequences", "max_tokens"} {
		want, had := in[field]
		if !had || empty(want) {
			continue
		}
		if have, kept := out[field]; !kept || !reflect.DeepEqual(want, have) {
			fail(field, want, have)
		}
	}
	if names := anthropicToolNames(in["tools"]); len(names) > 0 {
		if have := anthropicToolNames(out["tools"]); !reflect.DeepEqual(names, have) {
			fail("tools", names, have)
		}
	}
	for _, field := range []string{"tool_choice", "thinking"} {
		if !asks(in[field]) {
			continue
		}
		if _, kept := out[field]; !kept {
			fail(field, in[field], nil)
		}
	}
	if want, have := folded(in["messages"]), folded(out["messages"]); !reflect.DeepEqual(want, have) {
		fail("messages", want, have)
	}
}

func anthropicToolNames(v any) []string {
	list, ok := v.([]any)
	if !ok {
		return nil
	}
	var out []string
	for _, entry := range list {
		if tool, ok := entry.(map[string]any); ok {
			if name, ok := tool["name"].(string); ok {
				out = append(out, name)
			}
		}
	}
	return out
}

func folded(v any) []string {
	var out []string
	for _, role := range roles(v) {
		if len(out) == 0 || out[len(out)-1] != role {
			out = append(out, role)
		}
	}
	return out
}

func TestAnUnmodelledBlockRoundTripsToItsOwnVendor(t *testing.T) {
	front, seen := anthropicProxy(t)
	body := `{"model":"m","max_tokens":16,"messages":[
		{"role":"assistant","content":[
			{"type":"text","text":"before"},
			{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"x"}},
			{"type":"text","text":"after"}]},
		{"role":"user","content":"go on"}]}`
	resp, err := front.Client().Post(front.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	got := string(*seen)
	if !strings.Contains(got, `"server_tool_use"`) || !strings.Contains(got, `"srvtoolu_1"`) {
		t.Fatalf("the block was dropped at parse: %s", got)
	}
	before := strings.Index(got, `"before"`)
	block := strings.Index(got, `"server_tool_use"`)
	after := strings.Index(got, `"after"`)
	if !(before < block && block < after) {
		t.Fatalf("the block moved: %s", got)
	}
}

func TestStrictRefusesRatherThanDeliverALoss(t *testing.T) {
	front, _ := crossProxyStrict(t)
	body := `{"model":"m","max_tokens":16,"messages":[
		{"role":"assistant","content":[{"type":"redacted_thinking","data":"b3BhcXVl"}]},
		{"role":"user","content":"go on"}]}`
	resp, err := front.Client().Post(front.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("a lossy translation was delivered: %s", out)
	}
}
