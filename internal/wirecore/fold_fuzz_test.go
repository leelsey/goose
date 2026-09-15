// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
)

func framesFrom(script []byte) []*llm.Frame {
	var out []*llm.Frame
	zero, one := 0, 1
	for _, b := range script {
		switch b % 13 {
		case 0:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Text{Text: "t"}}})
		case 1:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "r"}}})
		case 2:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "s"}}})
		case 3:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "a", Name: "f", Args: json.RawMessage(`{"k":`), Index: &zero}}})
		case 4:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.ToolUse{Args: json.RawMessage(`1}`), Index: &zero}}})
		case 5:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "b", Name: "g", Args: json.RawMessage(`{}`), Index: &one}}})
		case 6:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Image{MIME: "image/png", Data: []byte("ab"), Partial: true}}})
		case 7:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Image{MIME: "image/png", Data: []byte("cd")}}})
		case 8:
			out = append(out, &llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: "no"}}})
		case 9:
			out = append(out, &llm.Frame{Choice: &one, Delta: []llm.Part{&llm.Text{Text: "second"}}})
		case 10:
			u := llm.Usage{Prompt: 2, Completion: 3, Reported: true}
			out = append(out, &llm.Frame{Usage: &u, TurnEnd: true})
		case 11:
			out = append(out, &llm.Frame{Stop: llm.StopEnd})
		case 12:
			out = append(out, &llm.Frame{Block: &llm.Block{Reason: "SAFETY"}, Stop: llm.StopContentFilter})
		}
	}
	return out
}

func FuzzFold(f *testing.F) {
	for _, seed := range [][]byte{
		{0, 11}, {3, 4, 11}, {6, 7, 11}, {0, 9, 11}, {1, 2, 0, 11}, {12}, {10, 11}, {},
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, script []byte) {
		if len(script) > 48 {
			script = script[:48]
		}
		frames := framesFrom(script)
		s := llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "m"}, frames)
		resp, _ := Fold(s, FoldOptions{})
		if resp == nil {
			t.Fatal("folding produced no response at all")
		}

		var wantText string
		for _, fr := range frames {
			for _, p := range fr.Delta {
				if txt, ok := p.(*llm.Text); ok && fr.ChoiceIndex() == 0 {
					wantText += txt.Text
				}
			}
		}
		var haveText string
		for _, c := range resp.Choices {
			if c.Index == 0 {
				haveText = llm.TextOf(c.Message.Content)
			}
		}
		if haveText != wantText {
			t.Fatalf("text lost folding %v\n want: %q\n have: %q", script, wantText, haveText)
		}

		for _, c := range resp.Choices {
			if c.Stop == llm.StopNone {
				t.Fatalf("choice %d folded without a stop reason, script %v", c.Index, script)
			}
			if c.Message.Role == "" {
				t.Fatalf("choice %d folded without a role, script %v", c.Index, script)
			}
		}
	})
}
