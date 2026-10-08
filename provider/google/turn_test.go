// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func answer(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(answered)) }

func user(text string) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: text}}}
}

func call(id, name, sig string) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
		&llm.ToolUse{ID: id, Name: name, Args: json.RawMessage(`{}`), Signature: sig},
	}}
}

func result(id string) llm.Message {
	return llm.Message{Role: llm.RoleTool, Content: []llm.Part{
		&llm.ToolResult{ID: id, Content: []llm.Part{&llm.Text{Text: "20C"}}},
	}}
}

// sentParts is every part of every content the backend was sent, in order.
func sentParts(t *testing.T, body string) []map[string]json.RawMessage {
	t.Helper()
	var b struct {
		Contents []struct {
			Parts []map[string]json.RawMessage `json:"parts"`
		} `json:"contents"`
	}
	if err := json.Unmarshal([]byte(body), &b); err != nil {
		t.Fatal(err)
	}
	var out []map[string]json.RawMessage
	for _, c := range b.Contents {
		out = append(out, c.Parts...)
	}
	return out
}

func signatureOn(t *testing.T, parts []map[string]json.RawMessage, fn string) string {
	t.Helper()
	for _, p := range parts {
		var fc struct {
			Name string `json:"name"`
		}
		if raw, ok := p["functionCall"]; ok && json.Unmarshal(raw, &fc) == nil && fc.Name == fn {
			var sig string
			_ = json.Unmarshal(p["thoughtSignature"], &sig)
			return sig
		}
	}
	t.Fatalf("no call to %s in %v", fn, parts)
	return ""
}

// The Anthropic, Responses and most chat clients name only the call a result
// answers, and Gemini refuses a function response with no name.
func TestAToolResultIsNamedAfterTheCallItAnswers(t *testing.T) {
	p, bodies, _ := serve(t, answer)
	req := &llm.Request{Messages: []llm.Message{user("weather?"), call("call_1", "get_weather", ""), result("call_1")}}
	if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains((*bodies)[0], `"functionResponse":{"name":"get_weather"`) {
		t.Fatalf("body = %s: the result goes out named after the call it answers", (*bodies)[0])
	}
	if req.Messages[2].Content[0].(*llm.ToolResult).Name != "" {
		t.Fatal("the caller's own request was written to")
	}
}

func TestAToolResultThatAnswersNoCallIsRefusedUnsent(t *testing.T) {
	p, bodies, _ := serve(t, answer)
	req := &llm.Request{Messages: []llm.Message{user("weather?"), result("call_9")}}
	_, err := p.Complete(context.Background(), attempt(req))
	e := llm.AsError(err)
	if e == nil || e.Kind != llm.KindInvalidRequest || e.Sent || !strings.Contains(e.Message, `"call_9"`) {
		t.Fatalf("error = %+v: a result with nothing to name it by cannot be sent, and says which", e)
	}
	if len(*bodies) != 0 {
		t.Fatal("the request was sent and paid for, to be refused")
	}
}

// Gemini 3 refuses a turn whose steps do not each carry the signature of their
// first call. A call that lost it on a client wire with no field for one goes
// with Google's placeholder; an older turn and a parallel call are not checked.
func TestAnUnsignedCallInTheTurnGoesWithGooglesPlaceholder(t *testing.T) {
	p, bodies, _ := serve(t, answer)
	parallel := call("b", "fb", "")
	parallel.Content = append(parallel.Content, &llm.ToolUse{ID: "c", Name: "fc", Args: json.RawMessage(`{}`)})
	req := &llm.Request{Messages: []llm.Message{
		user("first"), call("a", "fa", ""), result("a"),
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "done"}}},
		user("second"), parallel, result("b"), result("c"),
		call("d", "fd", "SIG-d"), result("d"),
	}}
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	parts := sentParts(t, (*bodies)[0])
	for fn, want := range map[string]string{"fa": "", "fb": skipSignature, "fc": "", "fd": "SIG-d"} {
		if got := signatureOn(t, parts, fn); got != want {
			t.Errorf("%s signature = %q, want %q", fn, got, want)
		}
	}
	notes := resp.Diagnostics.Find(llm.DropSignature)
	if len(notes) != 1 || notes[0].Count != 1 || notes[0].Severity != llm.SeverityInfo {
		t.Fatalf("diagnostics = %+v: the reasoning a placeholder stood in for is named once", resp.Diagnostics)
	}
	if req.Messages[5].Content[0].(*llm.ToolUse).Signature != "" {
		t.Fatal("the caller's own request was written to")
	}
}
