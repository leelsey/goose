// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollamaapi

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func TestChatStreamDefaultsToTrue(t *testing.T) {
	_, opts, _, err := NewChat(Options{}).Decode(&wire.Inbound{Body: []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`)})
	if err != nil {
		t.Fatal(err)
	}
	if !opts.Stream {
		t.Fatal("absent stream must mean streaming on this wire")
	}
	_, opts, _, err = NewChat(Options{}).Decode(&wire.Inbound{Body: []byte(`{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}`)})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Stream {
		t.Fatal("explicit false ignored")
	}
}

func TestChatDecodeMessages(t *testing.T) {
	body := `{"model":"m","messages":[
		{"role":"user","content":"look","images":["aGk="]},
		{"role":"assistant","content":"","thinking":"hmm","tool_calls":[{"function":{"name":"f","arguments":{"a":1}}}]},
		{"role":"tool","tool_name":"f","content":"42"}
	],"options":{"temperature":0.2,"num_predict":128,"stop":["END"]},"format":"json","think":true}`
	r, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if img, ok := r.Messages[0].Content[1].(*llm.Image); !ok || string(img.Data) != "hi" {
		t.Fatalf("image = %+v", r.Messages[0].Content)
	}
	if th, ok := r.Messages[1].Content[0].(*llm.Thinking); !ok || th.Text != "hmm" {
		t.Fatalf("thinking = %+v", r.Messages[1].Content)
	}
	if tu := llm.ToolUses(r.Messages[1].Content); len(tu) != 1 || tu[0].Name != "f" {
		t.Fatalf("tool call = %+v", tu)
	}
	if tr, ok := r.Messages[2].Content[0].(*llm.ToolResult); !ok || llm.TextOf(tr.Content) != "42" {
		t.Fatalf("tool result = %+v", r.Messages[2].Content)
	}
	if *r.Temperature != 0.2 || *r.MaxOutputTokens != 128 || r.Stop[0] != "END" {
		t.Fatalf("options not applied: %+v", r)
	}
	if r.ResponseFormat == nil || r.ResponseFormat.Type != "json_object" {
		t.Fatalf("format = %+v", r.ResponseFormat)
	}
	if r.Thinking == nil || !r.Thinking.Enabled {
		t.Fatal("think flag lost")
	}
}

func frames(t *testing.T, c wire.Codec, fs ...*llm.Frame) []map[string]any {
	t.Helper()
	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, fs)
	w := &wire.BufferWriter{}
	if _, err := c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatalf("encode: %v", err)
	}
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(w.Body.String()), "\n") {
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			t.Fatalf("line %q: %v", line, err)
		}
		out = append(out, m)
	}
	return out
}

func TestChatStreamHoldsTerminalFrame(t *testing.T) {
	got := frames(t, NewChat(Options{}),
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "he"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "llo"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 20, Completion: 5, Reported: true}, TurnEnd: true},
	)
	last := got[len(got)-1]
	if last["done"] != true || last["done_reason"] != "stop" {
		t.Fatalf("terminal frame = %v", last)
	}
	if last["prompt_eval_count"].(float64) != 20 || last["eval_count"].(float64) != 5 {
		t.Fatalf("counters = %v: the terminal frame carries the final figures", last)
	}
	for _, f := range got[:len(got)-1] {
		if f["done"] == true {
			t.Fatalf("an early frame claimed the turn was done: %v", f)
		}
	}
}

func TestChatStreamEmitsBufferedToolCallOnTheTerminalFrame(t *testing.T) {
	i0 := 0
	got := frames(t, NewChat(Options{}),
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Name: "search", Args: json.RawMessage(`{"q":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Args: json.RawMessage(`"go"}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	last := got[len(got)-1]
	msg := last["message"].(map[string]any)
	calls := msg["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("tool calls = %v", calls)
	}
	args := calls[0].(map[string]any)["function"].(map[string]any)["arguments"].(map[string]any)
	if args["q"] != "go" {
		t.Fatalf("arguments = %v: fragments must be assembled before they reach the wire", args)
	}
}

func TestChatFoldsRefusalIntoContent(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.Refusal{Text: "I can't"}}},
		Stop:    llm.StopRefusal,
	}}}
	out, _, err := NewChat(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"I can't"`) {
		t.Fatalf("body = %s: a refusal must be visible, not blank", out.Body)
	}
}

func TestGenerateKeepsThinking(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.Thinking{Text: "reasoned"}, &llm.Text{Text: "answer"}}},
		Stop:    llm.StopEnd,
	}}}
	out, diags, err := NewGenerate(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"thinking":"reasoned"`) {
		t.Fatalf("body = %s: this endpoint has a thinking field", out.Body)
	}
	if diags.Worst() >= llm.SeverityLoss {
		t.Fatalf("diagnostics = %+v", diags)
	}
}

func TestGenerateRefusesToDeliverAToolOnlyTurnBlank(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.ToolUse{Name: "f", Args: json.RawMessage(`{}`)}}},
		Stop:    llm.StopToolUse,
	}}}
	_, diags, err := NewGenerate(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if err := (llm.LossPolicy{}).Check(diags); err == nil {
		t.Fatal("an answer that is only a tool call cannot be delivered as an empty string")
	}
}

func TestGenerateDecodesSystemAndPrompt(t *testing.T) {
	r, opts, _, err := NewGenerate(Options{}).Decode(&wire.Inbound{Body: []byte(`{"model":"m","system":"be brief","prompt":"why?","stream":false}`)})
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(r.System) != "be brief" || llm.TextOf(r.Messages[0].Content) != "why?" {
		t.Fatalf("request = %+v", r)
	}
	if opts.Stream {
		t.Fatal("explicit stream:false ignored")
	}
}

func TestTruncatedStreamIsToldInBand(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	})
	w := &wire.BufferWriter{}
	_, err := NewChat(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if e := llm.AsError(err); e == nil || e.Code != llm.WireNoFinish {
		t.Fatalf("err = %v, want %s", err, llm.WireNoFinish)
	}
	if !strings.Contains(w.Body.String(), "finish reason") {
		t.Fatalf("body = %s", w.Body.String())
	}
}

func TestThinkTakesALevelAsWellAsASwitch(t *testing.T) {
	for _, tc := range []struct {
		body    string
		enabled bool
		effort  llm.Effort
	}{
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"think":true}`, true, ""},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"think":false}`, false, ""},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"think":"high"}`, true, llm.EffortHigh},
		{`{"model":"m","messages":[{"role":"user","content":"hi"}],"think":"low"}`, true, llm.EffortLow},
	} {
		req, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{Body: []byte(tc.body)})
		if err != nil {
			t.Fatalf("%s: %v", tc.body, err)
		}
		if req.Thinking == nil || req.Thinking.Enabled != tc.enabled {
			t.Errorf("%s: thinking = %+v", tc.body, req.Thinking)
		}
		if req.Effort != tc.effort {
			t.Errorf("%s: effort = %q", tc.body, req.Effort)
		}
	}
	if _, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{
		Body: []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"think":"enormous"}`),
	}); err == nil {
		t.Error("a level this endpoint does not spell was accepted")
	}
}

func TestNativeFieldsTravelRatherThanVanish(t *testing.T) {
	req, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{
		Body: []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"keep_alive":"10m"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if string(req.Extra["keep_alive"]) != `"10m"` {
		t.Fatalf("keep_alive = %s", req.Extra["keep_alive"])
	}
	gen, _, _, err := NewGenerate(Options{}).Decode(&wire.Inbound{
		Body: []byte(`{"model":"m","prompt":"hi","context":[1,2,3],"raw":true,"template":"t","suffix":"s"}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"context", "raw", "template", "suffix"} {
		if _, ok := gen.Extra[key]; !ok {
			t.Errorf("%s was dropped at parse", key)
		}
	}
}

func TestDoubleEncodedArgumentsAreRead(t *testing.T) {
	req, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{
		Body: []byte(`{"model":"m","messages":[{"role":"assistant","tool_calls":[
			{"function":{"name":"f","arguments":"{\"q\":\"x\"}"}}]}]}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	calls := llm.ToolUses(req.Messages[0].Content)
	if len(calls) != 1 {
		t.Fatalf("calls = %d", len(calls))
	}
	if got := string(calls[0].Args); got != `{"q":"x"}` {
		t.Fatalf("arguments = %s", got)
	}

	plain, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{
		Body: []byte(`{"model":"m","messages":[{"role":"assistant","tool_calls":[
			{"function":{"name":"f","arguments":"hello"}}]}]}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := string(llm.ToolUses(plain.Messages[0].Content)[0].Args); got != `"hello"` {
		t.Fatalf("arguments = %s", got)
	}
}

func TestAToolResultWithNoNameIsNotGivenAnIDThatMatchesNothing(t *testing.T) {
	req, _, _, err := NewChat(Options{}).Decode(&wire.Inbound{
		Body: []byte(`{"model":"m","messages":[{"role":"tool","content":"42"}]}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	result := req.Messages[0].Content[0].(*llm.ToolResult)
	if result.ID != "" || result.Name != "" {
		t.Fatalf("result = %+v", result)
	}
}

func TestTheMessageKeyIsAlwaysPresent(t *testing.T) {
	out, _, err := NewChat(Options{}).EncodeResponse(&llm.Response{Model: "m"}, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	if _, ok := body["message"]; !ok {
		t.Fatalf("no message key: %s", out.Body)
	}
}

func TestFiguresThisWireCannotCarryAreReported(t *testing.T) {
	u := llm.Usage{Prompt: 100, CacheRead: 40, CacheWrite: 10, Completion: 20, Reasoning: 5, Reported: true}
	resp := &llm.Response{
		Model:   "m",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hi"}}}, Stop: llm.StopEnd}},
		Usage:   u,
	}
	_, diags, err := NewChat(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	fields := map[string]bool{}
	for _, d := range diags {
		if d.Code == llm.AcctFigureUncarried {
			fields[d.Field] = true
		}
	}
	for _, want := range []string{llm.FigureCacheRead, llm.FigureCacheWrite, llm.FigureReasoning} {
		if !fields[want] {
			t.Fatalf("diagnostics = %+v: %q went missing without a word", diags, want)
		}
	}

	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopEnd},
		{Usage: &u, TurnEnd: true},
	})
	streamed, err := NewChat(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
	if err != nil {
		t.Fatal(err)
	}
	if !streamed.Has(llm.AcctFigureUncarried) {
		t.Fatalf("streamed diagnostics = %+v: the streamed path loses the same columns", streamed)
	}
}

func TestEveryStreamedFrameIsStamped(t *testing.T) {
	for _, c := range []wire.Codec{NewChat(Options{}), NewGenerate(Options{})} {
		s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
			{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
			{Stop: llm.StopEnd},
		})
		w := &wire.BufferWriter{}
		if _, err := c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(strings.TrimSpace(w.Body.String()), "\n") {
			var frame map[string]any
			if err := json.Unmarshal([]byte(line), &frame); err != nil {
				t.Fatalf("%s: %v", c.Name(), err)
			}
			if frame["created_at"] == nil || frame["created_at"] == "" {
				t.Fatalf("%s: frame %s carries no stamp", c.Name(), line)
			}
		}
	}
}

func TestANegativeCeilingIsReported(t *testing.T) {
	for _, c := range []wire.Codec{NewChat(Options{}), NewGenerate(Options{})} {
		path := "/api/chat"
		body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":-1}}`
		if c.Name() == NameGenerate {
			path, body = "/api/generate", `{"model":"m","prompt":"hi","options":{"num_predict":-1}}`
		}
		req, _, diags, err := c.Decode(&wire.Inbound{Method: "POST", Path: path, Body: []byte(body)})
		if err != nil {
			t.Fatalf("%s: %v", c.Name(), err)
		}
		if req.MaxOutputTokens != nil {
			t.Fatalf("%s: ceiling = %d: a negative one is no ceiling", c.Name(), *req.MaxOutputTokens)
		}
		if !diags.Has(llm.DropExtraField) {
			t.Fatalf("%s: diagnostics = %+v: the setting went missing without a word", c.Name(), diags)
		}
	}

	req, _, diags, err := NewChat(Options{}).Decode(&wire.Inbound{
		Method: "POST", Path: "/api/chat",
		Body: []byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":0}}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if req.MaxOutputTokens == nil || *req.MaxOutputTokens != 0 {
		t.Fatalf("ceiling = %v: zero is a ceiling of zero", req.MaxOutputTokens)
	}
	if diags.Has(llm.DropExtraField) {
		t.Fatalf("diagnostics = %+v: nothing was dropped", diags)
	}
}

// A frame that writes no line can still carry fields this wire has no place
// for, and they wait for the next line that can carry them. Two such frames are
// two sets of fields, and how many wait is the upstream's to choose, so what
// waits is bounded.
func TestFieldsHeldBackFromQuietFramesTravelAndAreBounded(t *testing.T) {
	t.Run("two of them both travel", func(t *testing.T) {
		out := frames(t, NewChat(Options{}),
			&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
			&llm.Frame{Extra: map[string]json.RawMessage{"first": json.RawMessage(`1`)}},
			&llm.Frame{Extra: map[string]json.RawMessage{"second": json.RawMessage(`2`)}},
			&llm.Frame{Delta: []llm.Part{&llm.Text{Text: " there"}}},
			&llm.Frame{Stop: llm.StopEnd},
		)
		var held int
		for _, line := range out {
			for _, name := range []string{"first", "second"} {
				if _, ok := line[name]; ok {
					held++
				}
			}
		}
		if held != 2 {
			t.Errorf("%d of the two fields held back travelled: %v", held, out)
		}
	})

	t.Run("and a run of them is held to a line", func(t *testing.T) {
		const quiet = 20000
		fs := []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: "hi"}}}}
		for i := range quiet {
			fs = append(fs, &llm.Frame{Extra: map[string]json.RawMessage{
				fmt.Sprintf("field%d", i): json.RawMessage(`"some value or other"`),
			}})
		}
		fs = append(fs, &llm.Frame{Stop: llm.StopEnd})
		s := llm.StaticStream(llm.StreamMeta{Model: "m"}, fs)
		w := &wire.BufferWriter{}
		diags, err := NewChat(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
		if err != nil {
			t.Fatal(err)
		}
		// The ceiling is on bytes, so the body is what says whether it held:
		// twenty thousand of these fields are better than half a megabyte.
		if n := w.Body.Len(); n > 1<<18 {
			t.Errorf("one line came to %d bytes of held fields: what waits is the upstream's to choose", n)
		}
		var said bool
		for _, d := range diags {
			if d.Code == llm.DropExtraField {
				said = true
			}
		}
		if !said {
			t.Error("fields were left behind and nothing said so")
		}
	})
}

// A field this wire has no place for survives an unstreamed answer on this
// endpoint, so it survives a streamed one too. A backend's own decoration
// should not depend on which of the two a caller asked for.
func TestGenerateCarriesFieldsItHasNoPlaceForEitherWay(t *testing.T) {
	extra := map[string]json.RawMessage{"x_backend_nonce": json.RawMessage(`"aaa"`)}

	t.Run("a whole answer", func(t *testing.T) {
		resp := &llm.Response{
			Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "hi"}}}, Stop: llm.StopEnd}},
			Extra:   extra,
		}
		out, _, err := NewGenerate(Options{}).EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out.Body), `"x_backend_nonce":"aaa"`) {
			t.Fatalf("the field did not survive: %s", out.Body)
		}
	})

	t.Run("and a streamed one", func(t *testing.T) {
		s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
			{Delta: []llm.Part{&llm.Text{Text: "hi"}}, Extra: extra},
			{Stop: llm.StopEnd},
		})
		w := &wire.BufferWriter{}
		if _, err := NewGenerate(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(w.Body.String(), `"x_backend_nonce":"aaa"`) {
			t.Errorf("the field vanished on the streamed path:\n%s", w.Body.String())
		}
	})
}

// Neither endpoint has a field for a refusal, so one read as an answer is
// said once, whole or streamed, on both of them.
func TestARefusalFlattenedToTextIsReportedOnce(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.Refusal{Text: "I can't help with that."}}},
		Stop:    llm.StopEnd,
	}}}
	for _, c := range []wire.Codec{NewChat(Options{}), NewGenerate(Options{})} {
		_, diags, err := c.EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if n := len(diags.Find(llm.DropRefusal)); n != 1 {
			t.Errorf("%s: drop.refusal noted %d times on the whole answer", c.Name(), n)
		}
		s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
			{Delta: []llm.Part{&llm.Refusal{Text: "I can't "}}},
			{Delta: []llm.Part{&llm.Refusal{Text: "help with that."}}},
			{Stop: llm.StopEnd},
		})
		diags, err = c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
		if err != nil {
			t.Fatal(err)
		}
		if n := len(diags.Find(llm.DropRefusal)); n != 1 {
			t.Errorf("%s: drop.refusal noted %d times on the stream", c.Name(), n)
		}
	}
}

func TestArgumentsThatDidNotSurviveAreReported(t *testing.T) {
	broken := func() *llm.ToolUse {
		i0 := 0
		return &llm.ToolUse{Index: &i0, Name: "f", Args: json.RawMessage(`{"a":`)}
	}
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{broken()}},
		Stop:    llm.StopToolUse,
	}}}
	_, diags, err := NewChat(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !diags.Has(llm.DropToolCall) {
		t.Fatalf("diagnostics = %+v: a call sent with none of its arguments is a call that changed", diags)
	}
	s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{broken()}},
		{Stop: llm.StopToolUse},
	})
	diags, err = NewChat(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
	if err != nil {
		t.Fatal(err)
	}
	if !diags.Has(llm.DropToolCall) {
		t.Fatalf("diagnostics = %+v: the stream loses the arguments as surely as the answer does", diags)
	}
}
