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
	"github.com/leelsey/goose/wire/openaichat"
	"github.com/leelsey/goose/wire/openairesp"
)

func TestAChatPartTheWireCarriedWholeReachesTheBackend(t *testing.T) {
	const file = `{"type":"file","file":{"filename":"a.pdf","file_data":"data:application/pdf;base64,JVBERg=="}}`
	for name, c := range map[string]struct {
		dialect Dialect
		carried bool
	}{
		"openai":     {OpenAIDialect(), true},
		"compatible": {CompatibleDialect(), false},
	} {
		p, bodies, _ := serve(t, Options{Dialect: c.dialect}, func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		})
		resp, err := p.Complete(context.Background(), attempt("gpt-4o", &llm.Request{Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{
				&llm.Text{Text: "summarise"},
				&llm.Raw{Vendor: openaichat.Name, Type: "file", JSON: json.RawMessage(file)},
			}},
		}}))
		if err != nil {
			t.Fatal(err)
		}
		var sent struct {
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
			t.Fatal(err)
		}
		got := string(sent.Messages[0].Content)
		if carried := strings.Contains(got, `"a.pdf"`) && strings.Contains(got, `"summarise"`); carried != c.carried {
			t.Errorf("%s: content = %s, carried = %v", name, got, c.carried)
		}
		if dropped := resp.Diagnostics.Has(llm.DropUnknownBlock); dropped == c.carried {
			t.Errorf("%s: diagnostics = %v", name, resp.Diagnostics.Codes())
		}
	}
}

func TestAResponsesPartOrItemTheWireCarriedWholeReachesTheBackend(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[]}`))
	})
	a := attempt("m", &llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{
			&llm.Text{Text: "summarise"},
			&llm.Raw{Vendor: openairesp.Name, Type: "input_file", JSON: json.RawMessage(`{"type":"input_file","file_id":"file_1"}`)},
			&llm.Raw{Vendor: openairesp.Name, Type: "mcp_approval_response", JSON: json.RawMessage(`{"type":"mcp_approval_response","approval_request_id":"mcpr_1","approve":true}`)},
		}},
	}})
	a.Format = "openai-responses"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Input []struct {
			Type    string `json:"type"`
			Content []struct {
				Type string `json:"type"`
			} `json:"content"`
			Approve bool `json:"approve"`
		} `json:"input"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	in := sent.Input
	if len(in) != 2 || in[0].Type != "message" || len(in[0].Content) != 2 || in[0].Content[1].Type != "input_file" {
		t.Fatalf("input = %s: the file belongs inside the message", (*bodies)[0])
	}
	if in[1].Type != "mcp_approval_response" || !in[1].Approve {
		t.Fatalf("input = %s: the item belongs beside the message, whole", (*bodies)[0])
	}
	if resp.Diagnostics.Has(llm.DropUnknownBlock) {
		t.Fatalf("diagnostics = %v: nothing was dropped", resp.Diagnostics.Codes())
	}
}
