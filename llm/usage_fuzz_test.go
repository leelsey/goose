// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"testing"
)

func usageScript(script []byte) []*Frame {
	var out []*Frame
	for _, b := range script {
		switch b % 9 {
		case 0:
			out = append(out, &Frame{Delta: []Part{&Text{Text: "t"}}})
		case 1:
			out = append(out, &Frame{Usage: &Usage{Prompt: 10, Reported: true}})
		case 2:
			out = append(out, &Frame{Usage: &Usage{Completion: 5, Reported: true}})
		case 3:
			out = append(out, &Frame{Usage: &Usage{Prompt: 10, CacheRead: 4, Reported: true}})
		case 4:
			out = append(out, &Frame{Usage: &Usage{Prompt: 10, Completion: 5, Reported: true}, TurnEnd: true})
		case 5:
			out = append(out, &Frame{TurnEnd: true})
		case 6:
			out = append(out, &Frame{Stop: StopEnd})
		case 7:
			out = append(out, &Frame{Usage: &Usage{Completion: 7, Reasoning: 2, Reported: true}, TurnEnd: true})
		case 8:
			out = append(out, &Frame{Usage: &Usage{Prompt: 20, Completion: 10, Reported: true}, Total: true})
		}
	}
	return out
}

func FuzzStreamAccounting(f *testing.F) {
	for _, seed := range [][]byte{
		{1, 2, 4}, {1, 1, 2, 4}, {4, 4}, {3, 2, 4}, {5, 5}, {0, 1, 6, 4}, {7, 7},
		{4, 4, 8}, {8}, {8, 4}, {},
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, script []byte) {
		if len(script) > 32 {
			script = script[:32]
		}
		frames := usageScript(script)
		s := StaticStream(StreamMeta{Provider: "stub"}, frames)

		var last Usage
		for range s.Frames() {
			now := s.Usage()
			if now.Prompt < last.Prompt || now.Completion < last.Completion {
				t.Fatalf("usage ran backwards: %+v then %+v, script %v", last, now, script)
			}
			last = now
		}
		final := s.Usage()
		if final.Total != final.Prompt+final.Completion {
			t.Fatalf("total %d is not %d + %d, script %v", final.Total, final.Prompt, final.Completion, script)
		}
		if final.Prompt < 0 || final.Completion < 0 || final.CacheRead < 0 {
			t.Fatalf("negative figures: %+v, script %v", final, script)
		}

		var ceiling, summary, turn Usage
		for _, fr := range frames {
			switch {
			case fr.Usage != nil && fr.Total:
				summary = summary.Cover(*fr.Usage)
			case fr.Usage != nil:
				turn = turn.Merge(*fr.Usage)
			}
			if fr.TurnEnd {
				ceiling = ceiling.Add(turn)
				turn = Usage{}
			}
		}
		ceiling = ceiling.Add(turn)
		ceiling = ceiling.Cover(summary)
		if final.Prompt > ceiling.Prompt || final.Completion > ceiling.Completion {
			t.Fatalf("usage over the ceiling: %+v vs %+v, script %v", final, ceiling, script)
		}
	})
}
