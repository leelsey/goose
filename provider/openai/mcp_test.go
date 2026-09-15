// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"net/http"

	"github.com/leelsey/goose/llm"
)

func responsesOK(w http.ResponseWriter, r *http.Request) {
	_, _ = w.Write([]byte(`{"id":"resp_1","model":"m","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":3,"output_tokens":1}}`))
}

func TestAToolServerIsSentAsAResponsesTool(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, responsesOK)
	req := hello()
	req.Tools = []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`)}}
	req.MCPServers = []llm.MCPServer{{
		Name: "docs", URL: "https://tools.test/mcp", Description: "the docs",
		Token: "tok", Headers: map[string]string{"X-Team": "core"},
		AllowedTools: []string{"read"}, ReadOnly: true,
		Approval: llm.MCPApprovalNever, ApprovalTools: []string{"read"},
	}}
	a := attempt("gpt-5", req)
	a.Format = "openai-responses"
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}

	var sent struct {
		Tools []map[string]any `json:"tools"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Tools) != 2 {
		t.Fatalf("tools = %+v: the function and the connector both belong there", sent.Tools)
	}
	connector := sent.Tools[1]
	switch {
	case connector["type"] != "mcp" || connector["server_label"] != "docs":
		t.Fatalf("connector = %+v", connector)
	case connector["server_url"] != "https://tools.test/mcp" || connector["authorization"] != "tok":
		t.Fatalf("connector = %+v", connector)
	case connector["server_description"] != "the docs":
		t.Fatalf("connector = %+v", connector)
	}
	if allowed, ok := connector["allowed_tools"].(map[string]any); !ok || allowed["read_only"] != true {
		t.Fatalf("allowed = %+v", connector["allowed_tools"])
	}
	if approval, ok := connector["require_approval"].(map[string]any); !ok || approval["never"] == nil {
		t.Fatalf("approval = %+v", connector["require_approval"])
	}
}

func TestAToolServerChoosesTheEndpointThatServesIt(t *testing.T) {
	p, bodies, paths := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/responses") {
			responsesOK(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	})
	req := hello()
	req.MCPServers = []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp"}}
	if _, err := p.Complete(context.Background(), attempt("gpt-4o", req)); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix((*paths)[0], "/responses") {
		t.Fatalf("path = %s: chat completions has no field for a connector", (*paths)[0])
	}
	if !strings.Contains((*bodies)[0], `"type":"mcp"`) {
		t.Fatalf("body = %s", (*bodies)[0])
	}

	if _, err := p.Complete(context.Background(), attempt("gpt-4o", hello())); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix((*paths)[1], "/chat/completions") {
		t.Fatalf("path = %s", (*paths)[1])
	}
}

func TestAToolServerTurnedOffIsLeftOut(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, responsesOK)
	off := false
	req := hello()
	req.MCPServers = []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp", Enabled: &off}}
	a := attempt("gpt-5", req)
	a.Format = "openai-responses"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains((*bodies)[0], `"type":"mcp"`) {
		t.Fatalf("body = %s", (*bodies)[0])
	}
	if !resp.Diagnostics.Has(llm.DropMCPServer) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestToolsRefusedOnAServerAreNotSilentlyRestored(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, responsesOK)
	req := hello()
	req.MCPServers = []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp", DeniedTools: []string{"delete_all"}}}
	a := attempt("gpt-5", req)
	a.Format = "openai-responses"
	if _, err := p.Complete(context.Background(), a); err == nil {
		t.Fatal("a denylist this backend cannot express must not be dropped")
	}
	if len(*bodies) != 0 {
		t.Fatalf("a refused request was sent anyway: %v", *bodies)
	}
}

func TestAHostWithoutTheConnectorDoesNotClaimIt(t *testing.T) {
	for _, d := range []Dialect{
		OpenAIDialect(),
		{StreamUsage: true, PerModelCeiling: true, Responses: true},
		CompatibleDialect(),
	} {
		p, _, _ := serve(t, Options{Dialect: d}, responsesOK)
		if got := p.Capabilities().MCPServers; got != (d.Responses && d.Connectors) {
			t.Fatalf("dialect %+v claims MCPServers = %v", d, got)
		}
	}
}
