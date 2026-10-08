// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

// A tool the client's wire carried whole goes beside the function tools, not
// in place of them, and not left out because the field was already taken. It
// goes as written, down to a number no float holds.
func TestACarriedToolTravelsBesideTheFunctions(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, responsesOK)
	req := hello()
	req.Tools = []llm.Tool{{Name: "shell", Schema: []byte(`{"type":"object"}`)}}
	req.Extra = map[string]json.RawMessage{"tools": json.RawMessage(
		`[{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}},
		  {"type":"custom","name":"count","limit":12345678901234567890123,"scale":1e999}]`)}
	req.ExtraFrom = "openai-responses"
	a := attempt("gpt-5", req)
	a.Format = "openai-responses"
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Tools []struct {
			Type   string         `json:"type"`
			Name   string         `json:"name"`
			Format map[string]any `json:"format"`
		} `json:"tools"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Tools) != 3 || sent.Tools[0].Name != "shell" {
		t.Fatalf("tools = %+v", sent.Tools)
	}
	if c := sent.Tools[1]; c.Type != "custom" || c.Name != "apply_patch" || c.Format["definition"] != "start: /.+/" {
		t.Fatalf("carried tool = %+v", c)
	}
	for _, want := range []string{`"limit":12345678901234567890123`, `"scale":1e999`} {
		if !strings.Contains((*bodies)[0], want) {
			t.Errorf("body = %s, want %s as it was written", (*bodies)[0], want)
		}
	}
}

// The backend names its reasoning items, and a client on its wire keeps them
// by that name.
func TestTheBackendsReasoningIDIsKept(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[
			{"type":"reasoning","id":"rs_up","summary":[{"type":"summary_text","text":"because"}],"encrypted_content":"enc"}]}`))
	})
	a := attempt("m", hello())
	a.Format = "openai-responses"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if th, ok := resp.Choices[0].Message.Content[0].(*llm.Thinking); !ok || th.ID != "rs_up" || th.Signature != "enc" {
		t.Fatalf("unstreamed thinking = %#v", resp.Choices[0].Message.Content[0])
	}

	p, _, _ = serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.reasoning_summary_text.delta","item_id":"rs_up","delta":"because"}`,
			`{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_up","encrypted_content":"enc"}}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	seen := 0
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if th, ok := part.(*llm.Thinking); ok {
				seen++
				if th.ID != "rs_up" {
					t.Errorf("streamed thinking = %#v", th)
				}
			}
		}
	}
	if seen != 2 {
		t.Fatalf("saw %d thinking parts, want the summary and the signature", seen)
	}
}
