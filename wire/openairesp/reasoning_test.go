// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type reasoned struct {
	ID        string
	Encrypted string
	Summary   string
}

func reasoningOf(t *testing.T, raw json.RawMessage) (reasoned, bool) {
	t.Helper()
	var it struct {
		Type      string `json:"type"`
		ID        string `json:"id"`
		Encrypted string `json:"encrypted_content"`
		Summary   []struct {
			Text string `json:"text"`
		} `json:"summary"`
	}
	if err := json.Unmarshal(raw, &it); err != nil {
		t.Fatal(err)
	}
	if it.Type != "reasoning" {
		return reasoned{}, false
	}
	r := reasoned{ID: it.ID, Encrypted: it.Encrypted}
	for _, s := range it.Summary {
		r.Summary += s.Text
	}
	return r, true
}

func reasonings(t *testing.T, output []json.RawMessage, want []reasoned) []reasoned {
	t.Helper()
	var out []reasoned
	for _, raw := range output {
		r, ok := reasoningOf(t, raw)
		if !ok {
			continue
		}
		// An id goose minted is its own to choose; one the backend issued is
		// the client's to keep.
		if n := len(out); n < len(want) && want[n].ID == "" {
			if r.ID == "" {
				t.Errorf("reasoning item %d has no id", n)
			}
			r.ID = ""
		}
		out = append(out, r)
	}
	return out
}

// A reasoning item's encrypted content is what a client that keeps nothing on
// the backend hands back to have the reasoning count, and a Claude signature
// rides the same field. Each block keeps its own, on both paths.
func TestEveryReasoningItemKeepsItsSignature(t *testing.T) {
	for _, tc := range []struct {
		name   string
		frames []llm.Part
		whole  []llm.Part
		want   []reasoned
	}{
		{"named by the backend",
			[]llm.Part{
				&llm.Thinking{ID: "rs_up", Text: "because"},
				&llm.Thinking{ID: "rs_up", Signature: "ENC-1"},
				&llm.Text{Text: "answer"},
			},
			[]llm.Part{&llm.Thinking{ID: "rs_up", Text: "because", Signature: "ENC-1"}, &llm.Text{Text: "answer"}},
			[]reasoned{{ID: "rs_up", Encrypted: "ENC-1", Summary: "because"}}},
		{"a signature with no summary",
			[]llm.Part{&llm.Thinking{Signature: "ENC-2"}, &llm.Text{Text: "answer"}},
			[]llm.Part{&llm.Thinking{Signature: "ENC-2"}, &llm.Text{Text: "answer"}},
			[]reasoned{{Encrypted: "ENC-2"}}},
		{"two signed blocks",
			[]llm.Part{
				&llm.Thinking{Text: "first"}, &llm.Thinking{Signature: "S1"},
				&llm.Thinking{Text: "second"}, &llm.Thinking{Signature: "S2"},
				&llm.Text{Text: "answer"},
			},
			[]llm.Part{
				&llm.Thinking{Text: "first", Signature: "S1"},
				&llm.Thinking{Text: "second", Signature: "S2"},
				&llm.Text{Text: "answer"},
			},
			[]reasoned{{Encrypted: "S1", Summary: "first"}, {Encrypted: "S2", Summary: "second"}}},
		{"two items the backend named apart",
			[]llm.Part{&llm.Thinking{ID: "rs_a", Text: "first"}, &llm.Thinking{ID: "rs_b", Text: "second"}},
			[]llm.Part{&llm.Thinking{ID: "rs_a", Text: "first"}, &llm.Thinking{ID: "rs_b", Text: "second"}},
			[]reasoned{{ID: "rs_a", Summary: "first"}, {ID: "rs_b", Summary: "second"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := &llm.Response{ID: "r", Model: "m", Choices: []llm.Choice{{
				Message: llm.Message{Role: llm.RoleAssistant, Content: tc.whole}, Stop: llm.StopEnd,
			}}}
			out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
			if err != nil {
				t.Fatal(err)
			}
			var body struct {
				Output []json.RawMessage `json:"output"`
			}
			if err := json.Unmarshal(out.Body, &body); err != nil {
				t.Fatal(err)
			}
			if got := reasonings(t, body.Output, tc.want); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("unstreamed reasoning = %+v, want %+v", got, tc.want)
			}

			frames := make([]*llm.Frame, 0, len(tc.frames)+1)
			for _, p := range tc.frames {
				frames = append(frames, &llm.Frame{Delta: []llm.Part{p}})
			}
			frames = append(frames, &llm.Frame{Stop: llm.StopEnd, TurnEnd: true})
			var done, final []json.RawMessage
			summaries := 0
			for _, e := range events(t, frames...) {
				switch e.name {
				case "response.output_item.done":
					b, _ := json.Marshal(e.body["item"])
					done = append(done, b)
				case "response.completed":
					b, _ := json.Marshal(e.body["response"])
					var r struct {
						Output []json.RawMessage `json:"output"`
					}
					if err := json.Unmarshal(b, &r); err != nil {
						t.Fatal(err)
					}
					final = r.Output
				case "response.reasoning_summary_part.added":
					summaries++
				}
			}
			if got := reasonings(t, done, tc.want); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("streamed items = %+v, want %+v", got, tc.want)
			}
			if got := reasonings(t, final, tc.want); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("closing object = %+v, want %+v", got, tc.want)
			}
			// A summary part announced is one the closing item must carry.
			wantSummaries := 0
			for _, w := range tc.want {
				if w.Summary != "" {
					wantSummaries++
				}
			}
			if summaries != wantSummaries {
				t.Errorf("announced %d summary parts for %d summaries", summaries, wantSummaries)
			}
		})
	}
}

// "none" is how this wire asks for no reasoning, and a summary setting says
// how reasoning is reported rather than asking for any.
func TestOnlyAnEffortTurnsThinkingOn(t *testing.T) {
	for _, tc := range []struct {
		body     string
		thinking bool
		effort   llm.Effort
	}{
		{`{"model":"m","input":"hi","reasoning":{"effort":"none"}}`, false, "none"},
		{`{"model":"m","input":"hi","reasoning":{"summary":"auto"}}`, false, ""},
		{`{"model":"m","input":"hi","reasoning":{"effort":"minimal"}}`, true, llm.EffortMinimal},
		{`{"model":"m","input":"hi","reasoning":{"effort":"high","summary":"detailed"}}`, true, llm.EffortHigh},
	} {
		r, _ := decode(t, tc.body)
		if got := r.Thinking != nil && r.Thinking.Enabled; got != tc.thinking {
			t.Errorf("%s: thinking = %v, want %v", tc.body, got, tc.thinking)
		}
		if r.Effort != tc.effort {
			t.Errorf("%s: effort = %q, want %q: a backend on this wire is owed the word it was sent", tc.body, r.Effort, tc.effort)
		}
	}
}
