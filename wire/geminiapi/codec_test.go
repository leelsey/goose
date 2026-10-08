// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"encoding/json"
	"net/url"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func TestDecodeReadsModelAndTransportFromThePath(t *testing.T) {
	in := &wire.Inbound{
		Method: "POST",
		Path:   "/v1beta/models/gemini-2.5-pro:streamGenerateContent",
		Query:  url.Values{"alt": []string{"sse"}, "key": []string{"AIza-test"}},
		Body:   []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
	}
	r, opts, _, err := New(Options{}).Decode(in)
	if err != nil {
		t.Fatal(err)
	}
	if r.Model != "gemini-2.5-pro" {
		t.Fatalf("model = %q: it lives in the path on this wire", r.Model)
	}
	if !opts.Stream {
		t.Fatal("streaming is chosen by the path and the query, not the body")
	}
	if opts.Key != "AIza-test" {
		t.Fatalf("key = %q: this wire carries the credential in the query", opts.Key)
	}
}

func TestDecodePartsAndSystemInstruction(t *testing.T) {
	body := `{"systemInstruction":{"parts":[{"text":"be brief"}]},
		"contents":[
			{"role":"user","parts":[{"text":"look"},{"inlineData":{"mimeType":"image/png","data":"aGk="}}]},
			{"role":"model","parts":[{"text":"thinking","thought":true,"thoughtSignature":"sig"},{"functionCall":{"name":"f","args":{"a":1}}}]},
			{"role":"user","parts":[{"functionResponse":{"name":"f","response":{"ok":true}}}]}
		],
		"generationConfig":{"temperature":0.3,"maxOutputTokens":99,"candidateCount":2,"thinkingConfig":{"thinkingBudget":2048}},
		"tools":[{"functionDeclarations":[{"name":"f","description":"d","parameters":{"type":"object"}}]}],
		"toolConfig":{"functionCallingConfig":{"mode":"ANY","allowedFunctionNames":["f"]}}}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(r.System) != "be brief" {
		t.Fatalf("system = %q", llm.TextOf(r.System))
	}
	if img, ok := r.Messages[0].Content[1].(*llm.Image); !ok || string(img.Data) != "hi" {
		t.Fatalf("image = %+v", r.Messages[0].Content)
	}
	if r.Messages[1].Role != llm.RoleAssistant {
		t.Fatalf("role = %q: model means assistant", r.Messages[1].Role)
	}
	th, ok := r.Messages[1].Content[0].(*llm.Thinking)
	if !ok || th.Signature != "sig" {
		t.Fatalf("thought part = %+v", r.Messages[1].Content[0])
	}
	tu := llm.ToolUses(r.Messages[1].Content)
	if len(tu) != 1 || tu[0].ID != "call_f_0" {
		t.Fatalf("tool call = %+v: this wire has no id, so one is minted per name", tu)
	}
	tr, ok := r.Messages[2].Content[0].(*llm.ToolResult)
	if !ok || tr.ID != tu[0].ID {
		t.Fatalf("tool result = %+v: a minted id is only useful if it pairs the answer with its call", r.Messages[2].Content[0])
	}
	if r.Candidates != 2 || *r.MaxOutputTokens != 99 || *r.Temperature != 0.3 {
		t.Fatalf("generation config = %+v", r)
	}
	if r.Thinking == nil || r.Thinking.Budget != 2048 {
		t.Fatalf("thinking = %+v", r.Thinking)
	}
	if r.ToolChoice.Mode != llm.ToolSpecific || r.ToolChoice.Name != "f" {
		t.Fatalf("tool choice = %+v", r.ToolChoice)
	}
}

func TestUsageConventionRoundTrip(t *testing.T) {
	u := DecodeUsage(usageMetadata{
		PromptTokenCount: 1000, CandidatesTokenCount: 30, ThoughtsTokenCount: 70,
		CachedContentTokenCount: 400, ToolUsePromptTokenCount: 50,
	})
	if u.Prompt != 1050 {
		t.Fatalf("prompt = %d, want the tool prompt added and the cache left inside", u.Prompt)
	}
	if u.Completion != 100 || u.Reasoning != 70 {
		t.Fatalf("completion = %d reasoning = %d", u.Completion, u.Reasoning)
	}
	if u.CacheRead != 400 {
		t.Fatalf("cache read = %d", u.CacheRead)
	}
	back := encodeUsage(u)
	if back.CandidatesTokenCount != 30 || back.ThoughtsTokenCount != 70 {
		t.Fatalf("re-encoded = %+v: thinking must come back out of the candidate count", back)
	}
	if back.TotalTokenCount != 1150 {
		t.Fatalf("total = %d", back.TotalTokenCount)
	}
}

func TestEncodePromptBlockKeepsItsOwnShape(t *testing.T) {
	resp := &llm.Response{Block: &llm.Block{Reason: "SAFETY"}, Stop: llm.StopContentFilter}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out.Body, &got); err != nil {
		t.Fatal(err)
	}
	if _, ok := got["candidates"]; ok {
		t.Fatalf("body = %s: a blocked prompt has no candidate", out.Body)
	}
	if got["promptFeedback"].(map[string]any)["blockReason"] != "SAFETY" {
		t.Fatalf("body = %s", out.Body)
	}
}

func chunks(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(block, "data: ")), &m); err != nil {
			t.Fatalf("chunk %q: %v", block, err)
		}
		out = append(out, m)
	}
	return out
}

func TestStreamBuffersToolCallsAndCarriesUsageOnTheFinish(t *testing.T) {
	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{Model: "gemini"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "one"}, &llm.ToolUse{Index: &i0, Name: "f", Args: json.RawMessage(`{"a":`)}}},
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Args: json.RawMessage(`1}`)}}},
		{Stop: llm.StopToolUse},
		{Usage: &llm.Usage{Prompt: 90, Completion: 12, Reasoning: 4, Reported: true}, TurnEnd: true},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	got := chunks(t, w.Body.String())
	first := got[0]["candidates"].([]any)[0].(map[string]any)
	parts := first["content"].(map[string]any)["parts"].([]any)
	if parts[0].(map[string]any)["text"] != "one" {
		t.Fatalf("text sharing a chunk with a tool fragment must still be emitted: %v", parts)
	}
	last := got[len(got)-1]
	u := last["usageMetadata"].(map[string]any)
	if u["candidatesTokenCount"].(float64) != 8 || u["thoughtsTokenCount"].(float64) != 4 {
		t.Fatalf("usage = %v", u)
	}
	cand := last["candidates"].([]any)[0].(map[string]any)
	if cand["finishReason"] != "STOP" {
		t.Fatalf("finish = %v", cand)
	}
	fc := cand["content"].(map[string]any)["parts"].([]any)[0].(map[string]any)["functionCall"].(map[string]any)
	if fc["args"].(map[string]any)["a"].(float64) != 1 {
		t.Fatalf("arguments = %v: fragments are assembled before they reach the wire", fc)
	}
}

func TestStreamAlwaysEndsWithAFinishChunk(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: "x"}}}})
	w := &wire.BufferWriter{}
	_, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if e := llm.AsError(err); e == nil || e.Code != llm.WireNoFinish {
		t.Fatalf("error = %v, want %s", err, llm.WireNoFinish)
	}
	got := chunks(t, w.Body.String())
	if len(got) < 2 {
		t.Fatalf("chunks = %v: a stream that stops without a finish still needs a terminal chunk", got)
	}
	if !strings.Contains(w.Body.String(), "finish reason") {
		t.Fatalf("body = %s: the truncation must be told in band", w.Body.String())
	}
}

func TestAnActionThisProxyDoesNotServeIsRefused(t *testing.T) {
	for _, action := range []string{"countTokens", "embedContent", "batchEmbedContents", "predict"} {
		_, _, _, err := New(Options{}).Decode(&wire.Inbound{
			Method: "POST", Path: "/v1beta/models/gemini-2.5-pro:" + action,
			Body: []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
		})
		if err == nil {
			t.Errorf("%s was served as generation", action)
			continue
		}
		if llm.AsError(err).Kind != llm.KindNotFound {
			t.Errorf("%s: kind = %s", action, llm.AsError(err).Kind)
		}
	}
	for _, action := range []string{"generateContent", "streamGenerateContent"} {
		_, opts, _, err := New(Options{}).Decode(&wire.Inbound{
			Method: "POST", Path: "/v1beta/models/gemini-2.5-pro:" + action,
			Body: []byte(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`),
		})
		if err != nil {
			t.Fatalf("%s: %v", action, err)
		}
		if want := action == "streamGenerateContent"; opts.Stream != want {
			t.Errorf("%s: stream = %v", action, opts.Stream)
		}
	}
}

func TestRepeatedCallsOfOneNamePairWithTheirOwnResults(t *testing.T) {
	body := `{"contents":[
		{"role":"model","parts":[{"functionCall":{"name":"f","args":{"i":1}}},{"functionCall":{"name":"f","args":{"i":2}}}]},
		{"role":"user","parts":[{"functionResponse":{"name":"f","response":{"r":1}}},{"functionResponse":{"name":"f","response":{"r":2}}}]}
	]}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	calls := llm.ToolUses(r.Messages[0].Content)
	if len(calls) != 2 {
		t.Fatalf("calls = %+v", calls)
	}
	var results []*llm.ToolResult
	for _, p := range r.Messages[1].Content {
		if tr, ok := p.(*llm.ToolResult); ok {
			results = append(results, tr)
		}
	}
	if len(results) != 2 {
		t.Fatalf("results = %+v", results)
	}
	for i := range calls {
		if calls[i].ID != results[i].ID {
			t.Fatalf("call %d = %q, result = %q: each answer belongs to one call", i, calls[i].ID, results[i].ID)
		}
	}
	if calls[0].ID == calls[1].ID {
		t.Fatalf("both calls minted %q: two calls are not one", calls[0].ID)
	}
}

func TestAThoughtSignatureOnACallOrTextTravelsBothWays(t *testing.T) {
	body := `{"contents":[
		{"role":"user","parts":[{"text":"weather?"}]},
		{"role":"model","parts":[
			{"text":"Let me look.","thoughtSignature":"SIG-text"},
			{"functionCall":{"name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"SIG-call"},
			{"text":"","thoughtSignature":"SIG-alone"}]}
	]}`
	r, _, diags, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if len(diags) != 0 {
		t.Fatalf("diagnostics = %+v: a signed part is an ordinary part on this wire", diags)
	}
	parts := r.Messages[1].Content
	if tx, ok := parts[0].(*llm.Text); !ok || tx.Signature != "SIG-text" {
		t.Fatalf("text = %+v", parts[0])
	}
	if tu, ok := parts[1].(*llm.ToolUse); !ok || tu.Signature != "SIG-call" {
		t.Fatalf("call = %+v: the backend refuses a replayed call that has lost its signature", parts[1])
	}
	if tx, ok := parts[2].(*llm.Text); !ok || tx.Text != "" || tx.Signature != "SIG-alone" {
		t.Fatalf("signature-only part = %+v", parts[2])
	}

	out, diags := EncodeParts(parts)
	if len(diags) != 0 {
		t.Fatalf("diagnostics = %+v", diags)
	}
	want := []string{
		`{"text":"Let me look.","thoughtSignature":"SIG-text"}`,
		`{"functionCall":{"args":{"city":"Paris"},"name":"get_weather"},"thoughtSignature":"SIG-call"}`,
		`{"text":"","thoughtSignature":"SIG-alone"}`,
	}
	for i, w := range want {
		if string(out[i]) != w {
			t.Fatalf("part %d = %s, want %s", i, out[i], w)
		}
	}

	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Name: "get_weather", Args: json.RawMessage(`{"city":`), Signature: "SIG-call"}}},
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Args: json.RawMessage(`"Paris"}`)}}},
		{Delta: []llm.Part{&llm.Text{Signature: "SIG-alone"}}, Stop: llm.StopToolUse},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	streamed := w.Body.String()
	for _, sig := range []string{"SIG-call", "SIG-alone"} {
		if !strings.Contains(streamed, `"thoughtSignature":"`+sig+`"`) {
			t.Fatalf("stream = %s: %s did not reach the client", streamed, sig)
		}
	}
}

// Only what the client said travels beside the neutral request: a field it
// never sent, carried to another backend, would be reported as one it lost.
func TestThinkingLevelAndHiddenThoughtsTravelToThisWire(t *testing.T) {
	cases := []struct {
		config  string
		carried string
		hidden  bool
	}{
		{`{"thinkingLevel":"low"}`, `{"thinkingConfig":{"thinkingLevel":"low"}}`, true},
		{`{"thinkingBudget":1024}`, ``, true},
		{`{"thinkingBudget":1024,"includeThoughts":false}`, ``, true},
		{`{"thinkingLevel":"high","includeThoughts":true}`, `{"thinkingConfig":{"thinkingLevel":"high"}}`, false},
		{`{"thinkingBudget":1024,"includeThoughts":true}`, ``, false},
		{`{"thinkingBudget":0}`, ``, false},
	}
	for _, c := range cases {
		body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}],"generationConfig":{"thinkingConfig":` + c.config + `}}`
		r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
		if err != nil {
			t.Fatalf("%s: %v", c.config, err)
		}
		if r.Thinking == nil {
			t.Fatalf("%s: thinking was not read", c.config)
		}
		if got := string(r.Extra["generationConfig"]); got != c.carried {
			t.Errorf("%s: carried = %s, want %s", c.config, got, c.carried)
		}
		if r.Thinking.Hidden != c.hidden {
			t.Errorf("%s: hidden = %v, want %v: this wire hands thoughts back only when asked", c.config, r.Thinking.Hidden, c.hidden)
		}
	}
}

func TestArgumentsThatDidNotSurviveAreReported(t *testing.T) {
	broken := &llm.ToolUse{Name: "f", Args: json.RawMessage(`{"a":`)}
	if _, diags := EncodeParts([]llm.Part{broken}); !diags.Has(llm.DropToolCall) {
		t.Fatalf("diagnostics = %+v: a call sent with none of its arguments is a call that changed", diags)
	}
	i0 := 0
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Name: "f", Args: json.RawMessage(`{"a":`)}}, Stop: llm.StopToolUse},
	})
	diags, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
	if err != nil {
		t.Fatal(err)
	}
	if !diags.Has(llm.DropToolCall) {
		t.Fatalf("diagnostics = %+v: the stream loses the arguments as surely as the answer does", diags)
	}
}

func TestARefusalFlattenedToTextIsReportedOnce(t *testing.T) {
	if _, diags := EncodeParts([]llm.Part{&llm.Refusal{Text: "I can't help with that."}}); !diags.Has(llm.DropRefusal) {
		t.Fatalf("diagnostics = %+v: a refusal read as an answer is an answer the client will act on", diags)
	}
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Refusal{Text: "I can't "}}},
		{Delta: []llm.Part{&llm.Refusal{Text: "help with that."}}, Stop: llm.StopEnd},
	})
	diags, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
	if err != nil {
		t.Fatal(err)
	}
	if n := len(diags.Find(llm.DropRefusal)); n != 1 {
		t.Fatalf("drop.refusal noted %d times: one refusal in two deltas is one refusal", n)
	}
}
