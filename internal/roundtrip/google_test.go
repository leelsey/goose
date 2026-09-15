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
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/geminiapi"
)

func googleProxy(t testing.TB) (*httptest.Server, *[]byte) {
	t.Helper()
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},
			"finishReason":"STOP","index":0}],
			"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}`))
	}))
	t.Cleanup(upstream.Close)

	p, err := google.New(google.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{geminiapi.New(geminiapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, &seen
}

func FuzzGoogleRequestSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`,
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"systemInstruction":{"parts":[{"text":"be brief"}]}}`,
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"temperature":0.5,"topP":0.9,"topK":40,"maxOutputTokens":64,"stopSequences":["a"]}}`,
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"tools":[{"functionDeclarations":[{"name":"f","description":"d","parameters":{"type":"object"}}]}]}`,
		`{"contents":[{"role":"model","parts":[{"functionCall":{"name":"f","args":{}}}]},{"role":"user","parts":[{"functionResponse":{"name":"f","response":{"r":1}}}]}]}`,
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"responseMimeType":"application/json"}}`,
	} {
		f.Add(seed)
	}
	front, seen := googleProxy(f)
	f.Fuzz(func(t *testing.T, body string) {
		if !json.Valid([]byte(body)) {
			return
		}
		resp, err := front.Client().Post(front.URL+"/v1beta/models/m:generateContent", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compareGoogle(t, body, *seen, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compareGoogle(t testing.TB, sent string, got []byte, reported string) {
	t.Helper()
	var in, out map[string]any
	if json.Unmarshal([]byte(sent), &in) != nil || json.Unmarshal(got, &out) != nil {
		return
	}
	lower(in)
	lower(out)
	fail := func(what string, want, have any) {
		t.Helper()
		if reported == "" {
			t.Fatalf("%s changed and nothing said so\n sent: %v\n  got: %v\n from: %s", what, want, have, sent)
		}
	}

	wantCfg, _ := in["generationconfig"].(map[string]any)
	haveCfg, _ := out["generationconfig"].(map[string]any)
	lower(wantCfg)
	lower(haveCfg)
	for _, field := range []string{"temperature", "topp", "topk", "maxoutputtokens", "stopsequences"} {
		want, had := wantCfg[field]
		if !had || empty(want) {
			continue
		}
		if have, kept := haveCfg[field]; !kept || !reflect.DeepEqual(want, have) {
			fail("generationConfig."+field, want, have)
		}
	}
	if names := declaredTools(in["tools"]); len(names) > 0 {
		if have := declaredTools(out["tools"]); !reflect.DeepEqual(names, have) {
			fail("tools", names, have)
		}
	}
	if want, have := contentRoles(in["contents"]), contentRoles(out["contents"]); !reflect.DeepEqual(want, have) {
		fail("contents", want, have)
	}
}

func declaredTools(v any) []string {
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
		lower(tool)
		decls, ok := tool["functiondeclarations"].([]any)
		if !ok {
			continue
		}
		for _, d := range decls {
			if fn, ok := d.(map[string]any); ok {
				if name, ok := fn["name"].(string); ok {
					out = append(out, name)
				}
			}
		}
	}
	return out
}

func contentRoles(v any) []string {
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
		lower(m)
		role, _ := m["role"].(string)
		if role == "" {
			role = "user"
		}
		if len(out) == 0 || out[len(out)-1] != role {
			out = append(out, role)
		}
	}
	return out
}

func TestGeminiKeepsItsOwnUnmodelledFields(t *testing.T) {
	front, seen := googleProxy(t)
	body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
		"safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT","threshold":"BLOCK_NONE"}],
		"cachedContent":"projects/p/locations/l/cachedContents/1"}`
	resp, err := front.Client().Post(front.URL+"/v1beta/models/m:generateContent", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d", resp.StatusCode)
	}
	for _, want := range []string{"safetySettings", "BLOCK_NONE", "cachedContent"} {
		if !strings.Contains(string(*seen), want) {
			t.Fatalf("%s did not reach the backend that has a field for it: %s", want, *seen)
		}
	}
}

func TestGeminiNativeConfigReachesTheBackend(t *testing.T) {
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	defer upstream.Close()

	p, err := google.New(google.Options{ID: "up", BaseURL: upstream.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{geminiapi.New(geminiapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	defer front.Close()

	resp, err := front.Client().Post(front.URL+"/v1beta/models/m:generateContent", "application/json",
		strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}],
			"generationConfig":{"temperature":0.5,"responseLogprobs":true,"mediaResolution":"MEDIA_RESOLUTION_LOW"}}`))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	var out struct {
		GenerationConfig map[string]any `json:"generationConfig"`
	}
	if err := json.Unmarshal(seen, &out); err != nil {
		t.Fatal(err)
	}
	if out.GenerationConfig["responseLogprobs"] != true {
		t.Fatalf("config = %v: a knob this codec does not model still belongs to the vendor", out.GenerationConfig)
	}
	if out.GenerationConfig["mediaResolution"] != "MEDIA_RESOLUTION_LOW" {
		t.Fatalf("config = %v", out.GenerationConfig)
	}
	if out.GenerationConfig["temperature"] != 0.5 {
		t.Fatalf("config = %v: the one it does model must survive too", out.GenerationConfig)
	}
}
