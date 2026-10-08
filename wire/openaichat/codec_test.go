// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func decode(t *testing.T, body string) (*llm.Request, wire.Options, llm.Diagnostics) {
	t.Helper()
	req, opts, diags, err := New(Options{}).Decode(&wire.Inbound{
		Method: "POST", Path: "/v1/chat/completions",
		Header: http.Header{"Authorization": []string{"Bearer sk-test"}},
		Body:   []byte(body),
	})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return req, opts, diags
}

func decodeErr(t *testing.T, body string) error {
	t.Helper()
	_, _, _, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Body: []byte(body)})
	if err == nil {
		t.Fatal("expected a rejection")
	}
	return err
}

func TestDecodeFoldsBothCeilingSpellings(t *testing.T) {
	r, _, _ := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":256}`)
	if r.MaxOutputTokens == nil || *r.MaxOutputTokens != 256 {
		t.Fatalf("ceiling = %v: a modern SDK sends only the newer spelling", r.MaxOutputTokens)
	}
	r, _, _ = decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":256}`)
	if *r.MaxOutputTokens != 10 {
		t.Fatalf("ceiling = %d, want the explicit max_tokens to win", *r.MaxOutputTokens)
	}
}

func TestDecodeRejectsFractionalCandidateCount(t *testing.T) {
	r, _, _ := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"n":2.0}`)
	if r.Candidates != 2 {
		t.Fatalf("candidates = %d, want 2", r.Candidates)
	}
	if err := decodeErr(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"n":1.5}`); err == nil {
		t.Fatal("a fractional n must be refused")
	}
}

func TestDecodeKeepsUnmodelledFields(t *testing.T) {
	r, _, _ := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"logit_bias":{"5":-100},"parallel_tool_calls":false}`)
	if string(r.Extra["logit_bias"]) != `{"5":-100}` {
		t.Fatalf("extra = %v: an unmodelled field must not vanish at parse", r.Extra)
	}
	if r.ParallelTools == nil || *r.ParallelTools {
		t.Fatalf("parallel = %v: the client asked for one call at a time", r.ParallelTools)
	}
	if _, ok := r.Extra["parallel_tool_calls"]; ok {
		t.Fatal("a field this codec reads must not also ride as an extra")
	}
}

func TestDecodeContentParts(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"look"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,aGk="}},
		{"type":"video_url","video_url":{"url":"x"}}
	]}]}`
	r, _, diags := decode(t, body)
	parts := r.Messages[0].Content
	if len(parts) != 3 {
		t.Fatalf("parts = %d", len(parts))
	}
	img, ok := parts[1].(*llm.Image)
	if !ok || img.MIME != "image/png" || string(img.Data) != "hi" {
		t.Fatalf("image = %+v: a data URL must be decoded once, here", parts[1])
	}
	raw, ok := parts[2].(*llm.Raw)
	if !ok || raw.Type != "video_url" {
		t.Fatalf("unknown block = %+v: it must be carried, not dropped", parts[2])
	}
	if len(diags) == 0 {
		t.Fatal("carrying an unknown block must be recorded")
	}
}

func TestDecodeToolTurns(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":1}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"result text"}
	]}`
	r, _, _ := decode(t, body)
	tu := llm.ToolUses(r.Messages[0].Content)
	if len(tu) != 1 || tu[0].ID != "call_1" || string(tu[0].Args) != `{"q":1}` {
		t.Fatalf("tool call = %+v", tu)
	}
	tr, ok := r.Messages[1].Content[0].(*llm.ToolResult)
	if !ok || tr.ID != "call_1" || llm.TextOf(tr.Content) != "result text" {
		t.Fatalf("tool result = %+v", r.Messages[1].Content[0])
	}
}

func TestDecodeRefusalAndDeveloperRole(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"developer","content":"be careful"},
		{"role":"assistant","content":null,"refusal":"I won't"}
	]}`
	r, _, _ := decode(t, body)
	if r.Messages[0].Role != llm.RoleDeveloper {
		t.Fatalf("role = %q: the developer role must survive", r.Messages[0].Role)
	}
	if llm.RefusalOf(r.Messages[1].Content) != "I won't" {
		t.Fatal("refusal lost at decode")
	}
	system, rest := r.SystemAndMessages()
	if llm.TextOf(system) != "be careful" || len(rest) != 1 {
		t.Fatalf("split = %q / %d", llm.TextOf(system), len(rest))
	}
}

func TestDecodeRejectsBuiltInTools(t *testing.T) {
	err := decodeErr(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search"}]}`)
	if llm.AsError(err).Kind != llm.KindUnsupported {
		t.Fatalf("err = %v: a non-portable tool must be refused, not ignored", err)
	}
}

func TestDecodeReadsClientKeyAndStreamOptions(t *testing.T) {
	_, opts, _ := decode(t, `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`)
	if !opts.Stream || !opts.StreamUsage || opts.Key != "sk-test" {
		t.Fatalf("options = %+v", opts)
	}
}

func TestEncodeResponseUsageShape(t *testing.T) {
	resp := &llm.Response{
		Model:   "served-model",
		Choices: []llm.Choice{{Index: 0, Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hello"}}}, Stop: llm.StopEnd}},
		Usage:   llm.Usage{Prompt: 1000, Completion: 20, CacheRead: 800, CacheWrite: 100, Reasoning: 5, Reported: true},
	}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{RequestModel: "asked-for"})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out.Body, &got); err != nil {
		t.Fatal(err)
	}
	usage := got["usage"].(map[string]any)
	if usage["prompt_tokens"].(float64) != 1000 {
		t.Fatalf("prompt_tokens = %v: it stays inclusive of the cache figures", usage["prompt_tokens"])
	}
	pd := usage["prompt_tokens_details"].(map[string]any)
	if pd["cached_tokens"].(float64) != 800 || pd["cache_write_tokens"].(float64) != 100 {
		t.Fatalf("details = %v: every SDK reads the nested spelling", pd)
	}
	cd := usage["completion_tokens_details"].(map[string]any)
	if cd["reasoning_tokens"].(float64) != 5 {
		t.Fatalf("reasoning = %v", cd)
	}
	if got["model"] != "asked-for" {
		t.Fatalf("model = %v: the client is told the name it used", got["model"])
	}
}

func TestEncodeResponseKeepsRefusal(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Refusal{Text: "no"}}},
		Stop:    llm.StopRefusal,
	}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"refusal":"no"`) {
		t.Fatalf("body = %s: a refusal must not ship as a blank answer", out.Body)
	}
}

func TestEncodeResponseReportsDroppedThinking(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.Thinking{Text: "reasoned"}, &llm.Text{Text: "answer"}}},
		Stop:    llm.StopEnd,
	}}}
	_, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !hasCode(diags, llm.DropThinking) {
		t.Fatalf("diagnostics = %+v: a wire with no reasoning field must report the drop", diags)
	}

	out, diags, err := New(Options{ReasoningField: "reasoning_content"}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if hasCode(diags, llm.DropThinking) {
		t.Fatal("configured reasoning field still reported a drop")
	}
	if !strings.Contains(string(out.Body), `"reasoning_content":"reasoned"`) {
		t.Fatalf("body = %s", out.Body)
	}
}

func TestEncodeUnknownStopReasonIsReported(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{Stop: llm.StopUnknown, Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "x"}}}}}}
	out, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"finish_reason":"stop"`) {
		t.Fatalf("body = %s", out.Body)
	}
	if len(diags) == 0 {
		t.Fatal("flattening an unknown stop reason must leave a trace")
	}
}

func hasCode(ds llm.Diagnostics, code string) bool {
	for _, d := range ds {
		if d.Code == code {
			return true
		}
	}
	return false
}

func streamOf(t *testing.T, opts wire.Options, frames ...*llm.Frame) (*wire.BufferWriter, llm.Diagnostics, error) {
	t.Helper()
	s := llm.StaticStream(llm.StreamMeta{ID: "chatcmpl-1", Model: "m"}, frames)
	w := &wire.BufferWriter{}
	diags, err := New(Options{}).EncodeStream(context.Background(), s, opts, w)
	return w, diags, err
}

func chunks(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		data := strings.TrimPrefix(block, "data: ")
		if data == "[DONE]" {
			out = append(out, map[string]any{"done": true})
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatalf("chunk %q: %v", data, err)
		}
		out = append(out, m)
	}
	return out
}

func TestEncodeStreamShape(t *testing.T) {
	w, _, err := streamOf(t, wire.Options{Stream: true, StreamUsage: true},
		&llm.Frame{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "he"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "llo"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 5, Completion: 2, Reported: true}, TurnEnd: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	got := chunks(t, w.Body.String())
	last := got[len(got)-1]
	if last["done"] != true {
		t.Fatalf("last frame = %v: clients block until the sentinel", last)
	}
	usage := got[len(got)-2]["usage"].(map[string]any)
	if usage["completion_tokens"].(float64) != 2 {
		t.Fatalf("usage chunk = %v: usage comes after the finish", usage)
	}
	if w.Flushes < 4 {
		t.Fatalf("flushes = %d: a buffering proxy is not a streaming one", w.Flushes)
	}
	if w.Head.Get("Content-Type") != "text/event-stream" {
		t.Fatalf("content type = %q", w.Head.Get("Content-Type"))
	}
}

func TestEncodeStreamKeepsRefusalDelta(t *testing.T) {
	w, _, err := streamOf(t, wire.Options{Stream: true},
		&llm.Frame{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Refusal{Text: "I can't"}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(w.Body.String(), `"refusal":"I can't"`) {
		t.Fatalf("body = %s: a refusal-only delta is not an empty chunk", w.Body.String())
	}
}

func TestEncodeStreamIndexesToolCalls(t *testing.T) {
	i0, i1 := 0, 1
	w, _, err := streamOf(t, wire.Options{Stream: true},
		&llm.Frame{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.ToolUse{Index: &i0, ID: "a", Name: "one", Args: json.RawMessage(`{}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i1, ID: "b", Name: "two", Args: json.RawMessage(`{}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	if err != nil {
		t.Fatal(err)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"index":0`) || !strings.Contains(body, `"index":1`) {
		t.Fatalf("body = %s: a client reassembles parallel calls by index", body)
	}
}

// Anthropic and Bedrock number a call by its block, which counts the text and
// thinking ahead of it; this wire numbers a choice's calls from nought.
func TestEncodeStreamNumbersCallsByTheirPlaceAmongCalls(t *testing.T) {
	b1, b2 := 1, 2
	w, _, err := streamOf(t, wire.Options{Stream: true},
		&llm.Frame{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "Let me check."}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &b1, ID: "toolu_1", Name: "get_weather"}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &b1, Args: json.RawMessage(`{"city":"Paris"}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &b2, ID: "toolu_2", Name: "get_time"}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &b2, Args: json.RawMessage(`{}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	if err != nil {
		t.Fatal(err)
	}
	var got []float64
	for _, ch := range chunks(t, w.Body.String()) {
		choices, _ := ch["choices"].([]any)
		for _, c := range choices {
			delta, _ := c.(map[string]any)["delta"].(map[string]any)
			calls, _ := delta["tool_calls"].([]any)
			for _, tc := range calls {
				got = append(got, tc.(map[string]any)["index"].(float64))
			}
		}
	}
	want := []float64{0, 0, 1, 1}
	if len(got) != len(want) {
		t.Fatalf("indices = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("indices = %v, want %v: the first call a client sees is call nought", got, want)
		}
	}
}

func TestEncodeStreamReportsUpstreamFailureInBand(t *testing.T) {
	w, diags, err := streamOf(t, wire.Options{Stream: true},
		&llm.Frame{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "half"}}},
		&llm.Frame{Err: &llm.Error{Kind: llm.KindOverloaded, Message: "overloaded"}},
	)
	if err == nil {
		t.Fatal("the failure must reach the caller")
	}
	body := w.Body.String()
	if !strings.Contains(body, `"error"`) {
		t.Fatalf("body = %s: after the headers the only way to say so is in band", body)
	}
	if !strings.HasSuffix(strings.TrimSpace(body), "data: [DONE]") {
		t.Fatalf("body = %s: the sentinel must close even a failed stream", body)
	}
	if !hasCode(diags, llm.WireErrorInBody) {
		t.Fatalf("diagnostics = %+v", diags)
	}
}

func TestEncodeStreamOmitsUsageUnlessAsked(t *testing.T) {
	w, _, err := streamOf(t, wire.Options{Stream: true},
		&llm.Frame{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "x"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 5, Reported: true}, TurnEnd: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(w.Body.String(), `"usage"`) {
		t.Fatal("usage was sent to a client that did not ask for it")
	}
}

func TestTruncatedStreamIsToldInBand(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "chatcmpl-1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	})
	w := &wire.BufferWriter{}
	diags, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if e := llm.AsError(err); e == nil || e.Code != llm.WireNoFinish {
		t.Fatalf("err = %v, want %s", err, llm.WireNoFinish)
	}
	body := w.Body.String()
	if !strings.Contains(body, `"error"`) || !strings.Contains(body, "finish reason") {
		t.Fatalf("body = %s: the truncation must reach the client", body)
	}
	if !strings.Contains(body, "[DONE]") {
		t.Fatalf("body = %s: a client blocks on the sentinel", body)
	}
	if !hasCode(diags, llm.WireErrorInBody) {
		t.Fatalf("diagnostics = %+v", diags)
	}
}

func TestOnlyTheFiguresThisWireCannotWriteAreReported(t *testing.T) {
	u := llm.Usage{Prompt: 100, CacheRead: 40, CacheWrite: 10, Completion: 20, Reasoning: 5, Reported: true}
	u.Cost = &llm.Cost{Amount: 0.02, Currency: "USD"}
	resp := &llm.Response{
		Model:   "m",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hi"}}}, Stop: llm.StopEnd}},
		Usage:   u,
	}
	_, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var fields []string
	for _, d := range diags {
		if d.Code == llm.AcctFigureUncarried {
			fields = append(fields, d.Field)
		}
	}
	if len(fields) != 1 || fields[0] != llm.FigureCost {
		t.Fatalf("uncarried = %v: this wire writes the cache and reasoning columns", fields)
	}
}

// A frame that puts nothing on the wire can still carry fields this wire does
// not model, and they wait for the next chunk that does. Two such frames are
// two sets of fields, not one replacing the other.
func TestFieldsHeldBackFromTwoQuietFramesBothTravel(t *testing.T) {
	// The first frame of a stream always opens a chunk, so the two quiet ones
	// have to come after something has already been said.
	w, _, err := streamOf(t, wire.Options{Stream: true},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		&llm.Frame{Extra: map[string]json.RawMessage{"first": json.RawMessage(`1`)}},
		&llm.Frame{Extra: map[string]json.RawMessage{"second": json.RawMessage(`2`)}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: " there"}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	if err != nil {
		t.Fatal(err)
	}
	body := w.Body.String()
	for _, want := range []string{`"first":1`, `"second":2`} {
		if !strings.Contains(body, want) {
			t.Errorf("%s did not survive: a quiet frame's unmodelled fields were dropped\n%s", want, body)
		}
	}
}

// How many frames carry nothing but fields this wire does not model is the
// upstream's to choose, so what waits for a chunk to carry them is bounded and
// the ceiling is said rather than the rest quietly going along.
func TestFieldsHeldBackAreBoundedAndTheCeilingIsSaid(t *testing.T) {
	const quiet = 20000
	frames := []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: "hi"}}}}
	for i := range quiet {
		frames = append(frames, &llm.Frame{Extra: map[string]json.RawMessage{
			fmt.Sprintf("field%d", i): json.RawMessage(`"some value or other"`),
		}})
	}
	frames = append(frames,
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: " there"}}},
		&llm.Frame{Stop: llm.StopEnd})

	w, diags, err := streamOf(t, wire.Options{Stream: true}, frames...)
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(w.Body.String(), `"field`); n >= quiet {
		t.Errorf("one chunk carried %d held fields: what waits is the upstream's to choose", n)
	}
	if !hasCode(diags, llm.DropExtraField) {
		t.Error("fields were left behind and nothing said so")
	}
}
