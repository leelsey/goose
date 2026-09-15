// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
)

func richResponse() *llm.Response {
	return &llm.Response{
		ID:       "resp_1",
		Provider: "stub",
		Model:    "m",
		Choices: []llm.Choice{{
			Index: 0,
			Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
				&llm.Thinking{Text: "first thought", Signature: "sig-1"},
				&llm.ToolUse{ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":"go"}`)},
				&llm.Thinking{Text: "second thought", Signature: "sig-2"},
				&llm.Text{Text: "here is the answer"},
				&llm.ToolUse{ID: "call_b", Name: "fetch", Args: json.RawMessage(`{"url":"x"}`)},
				&llm.Refusal{Text: "but not that part"},
				&llm.Image{MIME: "image/png", Data: []byte("png-bytes")},
			}},
			Stop: llm.StopToolUse,
		}},
		Usage: llm.Usage{Prompt: 120, Completion: 45, CacheRead: 100, Reported: true, Turns: 1},
	}
}

func TestSynthesiseFoldRoundTrip(t *testing.T) {
	orig := richResponse()
	got, err := Fold(Synthesise(orig), FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Choices) != 1 {
		t.Fatalf("choices = %d", len(got.Choices))
	}
	want := orig.Choices[0].Message.Content
	have := got.Choices[0].Message.Content
	if len(have) != len(want) {
		t.Fatalf("parts = %d, want %d", len(have), len(want))
	}
	for i := range want {
		if have[i].Kind() != want[i].Kind() {
			t.Fatalf("part %d = %s, want %s", i, have[i].Kind(), want[i].Kind())
		}
	}
	if have[0].(*llm.Thinking).Signature != "sig-1" || have[2].(*llm.Thinking).Signature != "sig-2" {
		t.Fatal("per-block signatures crossed or lost in the round trip")
	}
	if llm.RefusalOf(have) != "but not that part" {
		t.Fatal("refusal lost in the round trip")
	}
	tools := llm.ToolUses(have)
	if len(tools) != 2 || string(tools[1].Args) != `{"url":"x"}` {
		t.Fatalf("tool calls = %+v", tools)
	}
	if string(have[6].(*llm.Image).Data) != "png-bytes" {
		t.Fatal("image lost in the round trip")
	}
	if got.Choices[0].Stop != llm.StopToolUse {
		t.Fatalf("stop = %q", got.Choices[0].Stop)
	}
	if got.Usage.Prompt != 120 || got.Usage.Completion != 45 || got.Usage.CacheRead != 100 {
		t.Fatalf("usage = %+v", got.Usage)
	}
}

func TestSynthesiseIndexesToolCalls(t *testing.T) {
	frames := SynthesiseFrames(richResponse())
	var seen []int
	for _, f := range frames {
		for _, p := range f.Delta {
			if tu, ok := p.(*llm.ToolUse); ok {
				if tu.Index == nil {
					t.Fatal("a streamed tool call without an index cannot be reassembled")
				}
				seen = append(seen, *tu.Index)
			}
		}
	}
	if len(seen) != 2 || seen[0] != 0 || seen[1] != 1 {
		t.Fatalf("indices = %v", seen)
	}
}

func TestSynthesiseDoesNotMutateTheResponse(t *testing.T) {
	orig := richResponse()
	SynthesiseFrames(orig)
	for _, p := range orig.Choices[0].Message.Content {
		if tu, ok := p.(*llm.ToolUse); ok && tu.Index != nil {
			t.Fatal("synthesis wrote an index back into the caller's response")
		}
	}
}

func TestSynthesiseEmitsUsageLast(t *testing.T) {
	frames := SynthesiseFrames(richResponse())
	last := frames[len(frames)-1]
	if last.Usage == nil || !last.TurnEnd {
		t.Fatalf("last frame = %+v: usage must come after the finish", last)
	}
	for _, f := range frames[:len(frames)-1] {
		if f.Usage != nil {
			t.Fatal("usage emitted before the finish event")
		}
	}
}

func TestSynthesiseCarriesPromptBlock(t *testing.T) {
	resp := &llm.Response{Block: &llm.Block{Reason: "SAFETY"}, Stop: llm.StopContentFilter}
	got, err := Fold(Synthesise(resp), FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if got.Block == nil || got.Block.Reason != "SAFETY" || got.Stop != llm.StopContentFilter {
		t.Fatalf("response = %+v", got)
	}
	if len(got.Choices) != 0 {
		t.Fatal("a blocked prompt gained a candidate")
	}
}

func TestSynthesiseEmptyAnswerKeepsOneChoice(t *testing.T) {
	resp := &llm.Response{Choices: []llm.Choice{{Index: 0, Message: llm.Message{Role: llm.RoleAssistant}, Stop: llm.StopEnd}}}
	got, err := Fold(Synthesise(resp), FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Choices) != 1 || got.Choices[0].Stop != llm.StopEnd {
		t.Fatalf("response = %+v: an empty answer is still an answer", got)
	}
}
