// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// A reasoning item goes back under the id the backend gave it, which is how a
// stored one is found again, and what the backend signed is marked as OpenAI's.
func TestAReplayedReasoningItemGoesBackUnderItsID(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[
			{"type":"reasoning","id":"rs_up","summary":[],"encrypted_content":"enc"}]}`))
	})
	req := &llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{ID: "rs_stored"}}},
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
	}}
	a := attempt("m", req)
	a.Format = "openai-responses"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains((*bodies)[0], `"id":"rs_stored"`) {
		t.Fatalf("body = %s: a stored item is found by its id and nothing else", (*bodies)[0])
	}
	if th := resp.Choices[0].Message.Content[0].(*llm.Thinking); th.Signer != llm.FamilyOpenAI {
		t.Fatalf("thinking = %+v: what OpenAI signed is OpenAI's", th)
	}
}

// Thinking another backend produced, once negotiate has taken off the signature
// this one cannot verify, is neither stored here nor encrypted for here, and a
// reasoning item with nothing to find it by is refused.
func TestThinkingThisBackendCannotResolveIsNotReplayed(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
	})
	req := &llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{Text: "because"}, &llm.Thinking{ID: "rs_stored"}, &llm.Text{Text: "hello"},
		}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "alone"}}},
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
	}}
	a := attempt("m", req)
	a.Format = "openai-responses"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[0]
	if strings.Contains(body, "because") || strings.Contains(body, "alone") || strings.Count(body, `"type":"reasoning"`) != 1 {
		t.Fatalf("body = %s: only the item this backend stored goes back", body)
	}
	notes := resp.Diagnostics.Find(llm.DropThinking)
	if len(notes) != 1 || notes[0].Count != 2 || notes[0].Severity != llm.SeverityInfo {
		t.Fatalf("diagnostics = %+v: what was not replayed is said once", resp.Diagnostics)
	}
}

// A stored reasoning item with no summary and no signature reaches a streaming
// client as it reaches one that did not stream.
func TestAStoredReasoningItemIsStreamedToo(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_stored","summary":[]}}`,
			`{"type":"response.reasoning_summary_text.delta","item_id":"rs_said","delta":"because"}`,
			`{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_said","summary":[{"type":"summary_text","text":"because"}]}}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	a := attempt("m", hello())
	a.Format = "openai-responses"
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var ids []string
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if th, ok := part.(*llm.Thinking); ok {
				ids = append(ids, th.ID)
			}
		}
	}
	if strings.Join(ids, ",") != "rs_stored,rs_said" {
		t.Fatalf("thinking parts = %v: the stored item arrives once, and a summarised one is not repeated", ids)
	}
}
