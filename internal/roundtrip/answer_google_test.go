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
	"sync"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/geminiapi"
)

func answeringGoogle(t testing.TB) (*httptest.Server, func(string)) {
	t.Helper()
	var mu sync.Mutex
	reply := `{}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		mu.Lock()
		body := reply
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
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
	return front, func(body string) {
		mu.Lock()
		reply = body
		mu.Unlock()
	}
}

func FuzzGoogleAnswerSurvivesTheProxy(f *testing.F) {
	for _, seed := range []string{
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5}}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"a":1}}}]},"finishReason":"STOP","index":0}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"t","thought":true},{"text":"answer"}]},"finishReason":"STOP","index":0}]}`,
		`{"candidates":[{"finishReason":"SAFETY","index":0}],"promptFeedback":{"blockReason":"SAFETY"}}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AAAA"}}]},"finishReason":"STOP","index":0}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"a"}]},"finishReason":"MAX_TOKENS","index":0}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4,"thoughtsTokenCount":2,"cachedContentTokenCount":7,"totalTokenCount":13}}`,
	} {
		f.Add(seed)
	}
	front, answer := answeringGoogle(f)
	f.Fuzz(func(t *testing.T, reply string) {
		if !json.Valid([]byte(reply)) {
			return
		}
		answer(reply)
		resp, err := front.Client().Post(front.URL+"/v1beta/models/m:generateContent", "application/json",
			strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
		if err != nil {
			t.Fatal(err)
		}
		out, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return
		}
		compareGoogleAnswer(t, reply, out, resp.Header.Get(httpx.DiagnosticsHeader))
	})
}

func compareGoogleAnswer(t testing.TB, sent string, got []byte, reported string) {
	t.Helper()
	var in, out map[string]any
	if json.Unmarshal([]byte(sent), &in) != nil || json.Unmarshal(got, &out) != nil {
		return
	}
	fail := func(what string, want, have any) {
		t.Helper()
		if reported == "" {
			t.Fatalf("%s changed and nothing said so\n backend: %v\n  client: %v\n    from: %s", what, want, have, sent)
		}
	}
	wantText, wantCalls := geminiParts(in)
	haveText, haveCalls := geminiParts(out)
	if wantText != "" && haveText != wantText {
		fail("text", wantText, haveText)
	}
	if !reflect.DeepEqual(wantCalls, haveCalls) {
		fail("functionCall", wantCalls, haveCalls)
	}
	lower(in)
	lower(out)
	wantUsage, _ := in["usagemetadata"].(map[string]any)
	haveUsage, _ := out["usagemetadata"].(map[string]any)
	lower(wantUsage)
	lower(haveUsage)
	for _, field := range []string{"prompttokencount", "candidatestokencount"} {
		want, had := wantUsage[field]
		if !had || want == float64(0) {
			continue
		}
		if have := haveUsage[field]; have != want {
			fail("usageMetadata."+field, want, have)
		}
	}
}

func geminiParts(body map[string]any) (string, []string) {
	lower(body)
	cands, _ := body["candidates"].([]any)
	if len(cands) == 0 {
		return "", nil
	}
	c, _ := cands[0].(map[string]any)
	lower(c)
	content, _ := c["content"].(map[string]any)
	lower(content)
	parts, _ := content["parts"].([]any)
	var text strings.Builder
	var calls []string
	for _, entry := range parts {
		p, ok := entry.(map[string]any)
		if !ok {
			continue
		}
		lower(p)
		if thought, _ := p["thought"].(bool); thought {
			continue
		}
		if s, ok := p["text"].(string); ok {
			text.WriteString(s)
		}
		if fc, ok := p["functioncall"].(map[string]any); ok {
			lower(fc)
			if name, _ := fc["name"].(string); name != "" {
				calls = append(calls, name)
			}
		}
	}
	return text.String(), calls
}
