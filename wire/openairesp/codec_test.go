// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"context"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func decode(t *testing.T, body string) (*llm.Request, wire.Options) {
	t.Helper()
	r, o, _, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Path: "/v1/responses", Body: []byte(body)})
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	return r, o
}

func TestDecodeInputItems(t *testing.T) {
	body := `{"model":"gpt-5","instructions":"be brief","previous_response_id":"resp_prev","input":[
		{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},
		{"type":"reasoning","summary":[{"type":"summary_text","text":"thought"}],"encrypted_content":"enc"},
		{"type":"function_call","call_id":"c1","name":"f","arguments":"{\"a\":1}"},
		{"type":"function_call_output","call_id":"c1","output":"42"}
	],"reasoning":{"effort":"high"},"max_output_tokens":100}`
	r, opts := decode(t, body)
	if llm.TextOf(r.System) != "be brief" {
		t.Fatalf("instructions = %q", llm.TextOf(r.System))
	}
	if opts.Session != "resp_prev" {
		t.Fatalf("session = %q: it must travel beside the request, not inside it", opts.Session)
	}
	if r.Effort != llm.EffortHigh || *r.MaxOutputTokens != 100 {
		t.Fatalf("request = %+v", r)
	}
	th, ok := r.Messages[1].Content[0].(*llm.Thinking)
	if !ok || th.Text != "thought" || th.Signature != "enc" {
		t.Fatalf("reasoning item = %+v", r.Messages[1].Content[0])
	}
	if tu := llm.ToolUses(r.Messages[1].Content); len(tu) != 1 || tu[0].ID != "c1" {
		t.Fatalf("function call = %+v", tu)
	}
	tr, ok := r.Messages[2].Content[0].(*llm.ToolResult)
	if !ok || tr.ID != "c1" || llm.TextOf(tr.Content) != "42" {
		t.Fatalf("function output = %+v", r.Messages[2].Content[0])
	}
}

func TestDecodeRejectsProviderOnlyFeatures(t *testing.T) {
	for _, body := range []string{
		`{"model":"m","input":"hi","tools":[{"type":"web_search"}]}`,
		`{"model":"m","input":[{"type":"input_file","file_id":"f"}]}`,
		`{"model":"m","input":"hi","conversation":{"id":"c"}}`,
	} {
		_, _, _, err := New(Options{}).Decode(&wire.Inbound{Body: []byte(body)})
		if err == nil {
			t.Fatalf("expected a rejection for %s", body)
		}
		if llm.AsError(err).Kind != llm.KindUnsupported {
			t.Fatalf("err = %v", err)
		}
	}
}

func TestEncodeSummaryIsAlwaysAnArray(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Content: []llm.Part{&llm.Thinking{Text: "reasoned"}, &llm.Text{Text: "answer"}}},
		Stop:    llm.StopEnd,
	}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(out.Body, &got); err != nil {
		t.Fatal(err)
	}
	items := got["output"].([]any)
	first := items[0].(map[string]any)
	if first["type"] != "reasoning" {
		t.Fatalf("first item = %v: reasoning comes before the message", first)
	}
	summary, ok := first["summary"].([]any)
	if !ok || len(summary) != 1 {
		t.Fatalf("summary = %v: a bare string here is the shape SDKs crash on", first["summary"])
	}
	if summary[0].(map[string]any)["type"] != "summary_text" {
		t.Fatalf("summary part = %v", summary[0])
	}
}

func TestEncodeStatusFollowsTheStopReason(t *testing.T) {
	cases := map[llm.StopReason]string{
		llm.StopEnd:     "completed",
		llm.StopLength:  "incomplete",
		llm.StopPause:   "incomplete",
		llm.StopToolUse: "completed",
	}
	for stop, want := range cases {
		resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "x"}}}, Stop: stop}}}
		out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		var got map[string]any
		_ = json.Unmarshal(out.Body, &got)
		if got["status"] != want {
			t.Fatalf("stop %q gave status %v, want %v", stop, got["status"], want)
		}
	}
}

func TestEncodeUsageDetails(t *testing.T) {
	resp := &llm.Response{
		Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "x"}}}, Stop: llm.StopEnd}},
		Usage:   llm.Usage{Prompt: 500, Completion: 60, CacheRead: 400, Reasoning: 20, Reported: true},
	}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	_ = json.Unmarshal(out.Body, &got)
	u := got["usage"].(map[string]any)
	if u["input_tokens"].(float64) != 500 || u["total_tokens"].(float64) != 560 {
		t.Fatalf("usage = %v", u)
	}
	if u["input_tokens_details"].(map[string]any)["cached_tokens"].(float64) != 400 {
		t.Fatalf("cached tokens lost: %v", u)
	}
	if u["output_tokens_details"].(map[string]any)["reasoning_tokens"].(float64) != 20 {
		t.Fatalf("reasoning tokens lost: %v", u)
	}
}

type ev struct {
	name string
	body map[string]any
}

func events(t *testing.T, frames ...*llm.Frame) []ev {
	t.Helper()
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "gpt-5"}, frames)
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatalf("encode: %v", err)
	}
	var out []ev
	for _, block := range strings.Split(w.Body.String(), "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var e ev
		for _, line := range strings.Split(block, "\n") {
			switch {
			case strings.HasPrefix(line, "event: "):
				e.name = strings.TrimPrefix(line, "event: ")
			case strings.HasPrefix(line, "data: "):
				if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &e.body); err != nil {
					t.Fatalf("event %q: %v", line, err)
				}
			}
		}
		out = append(out, e)
	}
	return out
}

func names(evs []ev) []string {
	out := make([]string, len(evs))
	for i, e := range evs {
		out[i] = e.name
	}
	return out
}

func TestStreamClosesReasoningBeforeAMessageOpens(t *testing.T) {
	evs := events(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "think"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "answer"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 4, Completion: 2, Reported: true}, TurnEnd: true},
	)
	got := names(evs)
	reasonDone, msgAdded := -1, -1
	for i, n := range got {
		if n == "response.output_item.done" && reasonDone < 0 {
			reasonDone = i
		}
		if n == "response.output_item.added" && i > 2 && msgAdded < 0 {
			item := evs[i].body["item"].(map[string]any)
			if item["type"] == "message" {
				msgAdded = i
			}
		}
	}
	if reasonDone < 0 || msgAdded < 0 || reasonDone > msgAdded {
		t.Fatalf("events = %v: the reasoning item must close before a message opens", got)
	}
	if got[len(got)-1] != "response.completed" {
		t.Fatalf("last event = %s", got[len(got)-1])
	}
}

func TestStreamJoinsAToolCallSplitAcrossIndexlessFrames(t *testing.T) {
	evs := events(t,
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "call_a", Name: "search"}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Args: json.RawMessage(`{"q":"go"}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
		&llm.Frame{Usage: &llm.Usage{Prompt: 4, Completion: 2, Reported: true}, TurnEnd: true},
	)
	last := evs[len(evs)-1]
	resp := last.body["response"].(map[string]any)
	items := resp["output"].([]any)
	if len(items) != 1 {
		t.Fatalf("output = %v: the name and the arguments belong to one call", items)
	}
	call := items[0].(map[string]any)
	if call["name"] != "search" || call["arguments"] != `{"q":"go"}` {
		t.Fatalf("call = %v", call)
	}
}

func TestStreamTerminalEventCarriesEverything(t *testing.T) {
	evs := events(t,
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: "but no"}}},
		&llm.Frame{Stop: llm.StopEnd},
		&llm.Frame{Usage: &llm.Usage{Prompt: 10, Completion: 3, Reported: true}, TurnEnd: true},
	)
	last := evs[len(evs)-1]
	resp := last.body["response"].(map[string]any)
	items := resp["output"].([]any)
	if len(items) != 1 {
		t.Fatalf("output = %v", items)
	}
	content := items[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content = %v: the refusal must be flushed into the terminal response", content)
	}
	if content[1].(map[string]any)["refusal"] != "but no" {
		t.Fatalf("refusal lost: %v", content)
	}
	if resp["usage"].(map[string]any)["output_tokens"].(float64) != 3 {
		t.Fatalf("usage = %v", resp["usage"])
	}
}

func TestStreamIncompleteWhenTruncated(t *testing.T) {
	evs := events(t,
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "half"}}},
		&llm.Frame{Stop: llm.StopLength},
	)
	last := evs[len(evs)-1]
	if last.name != "response.incomplete" {
		t.Fatalf("last event = %s: a truncated turn is not completed", last.name)
	}
	resp := last.body["response"].(map[string]any)
	if resp["incomplete_details"].(map[string]any)["reason"] != "max_output_tokens" {
		t.Fatalf("details = %v", resp["incomplete_details"])
	}
}

func TestStreamEmitsCompleteToolArguments(t *testing.T) {
	i0, i1 := 0, 1
	evs := events(t,
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i0, ID: "c1", Name: "one", Args: json.RawMessage(`{"a":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i1, ID: "c2", Name: "two", Args: json.RawMessage(`{"b":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i0, Args: json.RawMessage(`1}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &i1, Args: json.RawMessage(`2}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	var args []string
	for _, e := range evs {
		if e.name == "response.function_call_arguments.done" {
			args = append(args, e.body["arguments"].(string))
		}
	}
	if len(args) != 2 || args[0] != `{"a":1}` || args[1] != `{"b":2}` {
		t.Fatalf("arguments = %v: interleaved fragments must not cross calls", args)
	}
}

func TestStreamFailureUsesTheFailedTerminal(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half"}}},
		{Err: &llm.Error{Kind: llm.KindOverloaded, Message: "overloaded"}},
	})
	w := &wire.BufferWriter{}
	_, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if err == nil {
		t.Fatal("the failure must reach the caller")
	}
	if !strings.Contains(w.Body.String(), "event: response.failed") {
		t.Fatalf("body = %s", w.Body.String())
	}
	if !strings.Contains(w.Body.String(), `"half"`) {
		t.Fatal("the partial answer was thrown away")
	}
}

func TestTruncatedStreamIsToldInBand(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "gpt-5"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	})
	w := &wire.BufferWriter{}
	_, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w)
	if e := llm.AsError(err); e == nil || e.Code != llm.WireNoFinish {
		t.Fatalf("err = %v, want %s", err, llm.WireNoFinish)
	}
	if !strings.Contains(w.Body.String(), "finish reason") {
		t.Fatalf("body = %s", w.Body.String())
	}
}

func TestTheStreamedObjectIsStampedLikeTheUnaryOne(t *testing.T) {
	made := time.Date(2024, 3, 4, 5, 6, 7, 0, time.UTC)
	s := llm.StaticStream(llm.StreamMeta{Model: "m", Created: made}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopEnd},
	})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(w.Body.String(), `"created_at":`+strconv.FormatInt(made.Unix(), 10)) {
		t.Fatalf("body = %s", w.Body.String())
	}

	quiet := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{{Stop: llm.StopEnd}})
	w = &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), quiet, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(w.Body.String(), `"created_at":0`) {
		t.Fatalf("body = %s: 1970 is not when this answer was made", w.Body.String())
	}
}

// The Responses schema makes these fields required, so a client that validates
// what it is handed cannot parse an answer that leaves them out.
func TestRequiredFieldsAreAlwaysWritten(t *testing.T) {
	resp := &llm.Response{
		ID: "r", Model: "m",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{Text: ""},
			&llm.Text{Text: "hello"},
			&llm.Refusal{Text: ""},
			&llm.ToolUse{Name: "f", Args: json.RawMessage(`{"a":1}`)},
		}}, Stop: llm.StopEnd}},
	}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil || out == nil {
		t.Fatalf("encode: %v", err)
	}
	var body struct {
		Output []map[string]json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	for _, entry := range body.Output {
		var kind string
		if err := json.Unmarshal(entry["type"], &kind); err != nil {
			t.Fatal(err)
		}
		switch kind {
		case "reasoning":
			if _, ok := entry["summary"]; !ok {
				t.Errorf("reasoning item has no summary: %v", keysOf(entry))
			}
		case "function_call":
			for _, want := range []string{"call_id", "name", "arguments"} {
				if _, ok := entry[want]; !ok {
					t.Errorf("function_call has no %s: %v", want, keysOf(entry))
				}
			}
		case "message":
			var parts []map[string]json.RawMessage
			if err := json.Unmarshal(entry["content"], &parts); err != nil {
				t.Fatal(err)
			}
			for _, part := range parts {
				var pk string
				if err := json.Unmarshal(part["type"], &pk); err != nil {
					t.Fatal(err)
				}
				switch pk {
				case "output_text":
					for _, want := range []string{"text", "annotations"} {
						if _, ok := part[want]; !ok {
							t.Errorf("output_text has no %s: %v", want, keysOf(part))
						}
					}
				case "refusal":
					if _, ok := part["refusal"]; !ok {
						t.Errorf("refusal has no refusal: %v", keysOf(part))
					}
				}
			}
		}
	}
}

func keysOf(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Every part the stream announces is a part the closing item carries, at the
// index the stream gave it. A client that followed the events and a client
// that read the final object must end up with the same answer.
func TestTheClosingItemCarriesEveryPartTheStreamAnnounced(t *testing.T) {
	for _, tc := range []struct {
		name   string
		frames []*llm.Frame
	}{
		{"refusal only", []*llm.Frame{
			{Delta: []llm.Part{&llm.Refusal{Text: "I can't help with that."}}},
			{Stop: llm.StopRefusal, TurnEnd: true},
		}},
		{"text then refusal", []*llm.Frame{
			{Delta: []llm.Part{&llm.Text{Text: "hm"}}},
			{Delta: []llm.Part{&llm.Refusal{Text: "no"}}},
			{Stop: llm.StopRefusal, TurnEnd: true},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := llm.StaticStream(llm.StreamMeta{Model: "m"}, tc.frames)
			w := &wire.BufferWriter{}
			if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
				t.Fatal(err)
			}
			announced := map[float64]bool{}
			var final []map[string]any
			for _, line := range strings.Split(w.Body.String(), "\n") {
				raw, ok := strings.CutPrefix(line, "data: ")
				if !ok {
					continue
				}
				var ev map[string]any
				if json.Unmarshal([]byte(raw), &ev) != nil {
					continue
				}
				switch ev["type"] {
				case "response.content_part.added":
					announced[ev["content_index"].(float64)] = true
				case "response.output_item.done":
					it, _ := ev["item"].(map[string]any)
					if it["type"] == "message" {
						parts, _ := it["content"].([]any)
						final = nil
						for _, p := range parts {
							final = append(final, p.(map[string]any))
						}
					}
				}
			}
			if len(final) != len(announced) {
				t.Fatalf("announced %d parts, the closing item carries %d: %v", len(announced), len(final), final)
			}
		})
	}
}
