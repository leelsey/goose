// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
)

type thinkingBlock struct {
	text       string
	signatures int
}

// thinkingBlocks reads back each thinking block a stream opened, with the
// text and the number of signatures written into it.
func thinkingBlocks(evs []event) []thinkingBlock {
	var out []thinkingBlock
	at := map[float64]int{}
	for _, e := range evs {
		switch e.name {
		case "content_block_start":
			if b, _ := e.body["content_block"].(map[string]any); b["type"] == "thinking" {
				at[e.body["index"].(float64)] = len(out)
				out = append(out, thinkingBlock{})
			}
		case "content_block_delta":
			i, ok := at[e.body["index"].(float64)]
			if !ok {
				continue
			}
			d := e.body["delta"].(map[string]any)
			switch d["type"] {
			case "thinking_delta":
				out[i].text += d["thinking"].(string)
			case "signature_delta":
				out[i].signatures++
			}
		}
	}
	return out
}

// A thinking block already signed, or one another named block follows, is
// over: writing the next block's deltas into it leaves one block carrying two
// signatures, which no backend will verify. A buffered tool call writes
// nothing, so it cannot be what closes the block. Regression test for
// r4-wire-1.
func TestASignedOrRenamedThinkingBlockIsNotContinued(t *testing.T) {
	i0 := 0
	for _, tc := range []struct {
		name   string
		frames []*llm.Frame
		want   []thinkingBlock
	}{
		{"signed blocks either side of a buffered call", []*llm.Frame{
			{Delta: []llm.Part{&llm.Thinking{ID: "rs_1", Text: "one"}}},
			{Delta: []llm.Part{&llm.Thinking{ID: "rs_1", Signature: "S1", Signer: llm.FamilyOpenAI}}},
			{Delta: []llm.Part{&llm.ToolUse{ID: "call_1", Name: "f", Args: json.RawMessage(`{}`), Index: &i0}}},
			{Delta: []llm.Part{&llm.Thinking{ID: "rs_2", Text: "two"}}},
			{Delta: []llm.Part{&llm.Thinking{ID: "rs_2", Signature: "S2", Signer: llm.FamilyOpenAI}}},
			{Stop: llm.StopToolUse},
		}, []thinkingBlock{{"one", 1}, {"two", 1}}},
		{"two signed parts in one frame", []*llm.Frame{
			{Delta: []llm.Part{
				&llm.Thinking{Text: "a", Signature: "S1", Signer: llm.FamilyAnthropic},
				&llm.Thinking{Text: "b", Signature: "S2", Signer: llm.FamilyAnthropic},
			}},
			{Stop: llm.StopEnd},
		}, []thinkingBlock{{"a", 1}, {"b", 1}}},
		{"a noted-only part between two named blocks", []*llm.Frame{
			{Delta: []llm.Part{&llm.Thinking{ID: "rs_1", Text: "one"}}},
			{Delta: []llm.Part{&llm.Raw{Vendor: "other", Type: "x", JSON: json.RawMessage(`{}`)}}},
			{Delta: []llm.Part{&llm.Thinking{ID: "rs_2", Text: "two"}}},
			{Stop: llm.StopEnd},
		}, []thinkingBlock{{"one", 0}, {"two", 0}}},
		{"one block streamed apart from its signature", []*llm.Frame{
			{Delta: []llm.Part{&llm.Thinking{Text: "I should "}}},
			{Delta: []llm.Part{&llm.Thinking{Text: "check."}}},
			{Delta: []llm.Part{&llm.Thinking{Signature: "S", Signer: llm.FamilyAnthropic}}},
			{Stop: llm.StopEnd},
		}, []thinkingBlock{{"I should check.", 1}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			evs, _ := stream(t, tc.frames...)
			got := thinkingBlocks(evs)
			if len(got) != len(tc.want) {
				t.Fatalf("thinking blocks = %+v, want %+v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("thinking blocks = %+v, want %+v", got, tc.want)
				}
			}
		})
	}
}
