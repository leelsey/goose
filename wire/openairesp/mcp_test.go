// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func TestAToolServerTheBackendDialsIsCarried(t *testing.T) {
	r, _ := decode(t, `{"model":"gpt-5","input":"hi","tools":[
		{"type":"function","name":"f","parameters":{"type":"object"}},
		{"type":"mcp","server_label":"docs","server_url":"https://tools.test/mcp",
		 "server_description":"the docs","authorization":"tok","headers":{"X-Team":"core"},
		 "allowed_tools":{"tool_names":["read"],"read_only":true},
		 "require_approval":{"never":{"tool_names":["read"]}}}]}`)

	if len(r.Tools) != 1 || r.Tools[0].Name != "f" {
		t.Fatalf("tools = %+v: the connector is not a function the client runs", r.Tools)
	}
	if len(r.MCPServers) != 1 {
		t.Fatalf("servers = %+v", r.MCPServers)
	}
	s := r.MCPServers[0]
	switch {
	case s.Name != "docs" || s.URL != "https://tools.test/mcp":
		t.Fatalf("server = %+v", s)
	case s.Description != "the docs" || s.Token != "tok" || s.Headers["X-Team"] != "core":
		t.Fatalf("server = %+v", s)
	case len(s.AllowedTools) != 1 || s.AllowedTools[0] != "read" || !s.ReadOnly:
		t.Fatalf("allowed = %+v", s)
	case s.Approval != llm.MCPApprovalNever || len(s.ApprovalTools) != 1:
		t.Fatalf("approval = %+v", s)
	}

	out, err := json.Marshal(EncodeMCPTool(s))
	if err != nil {
		t.Fatal(err)
	}
	again, err := DecodeMCPTool(out)
	if err != nil {
		t.Fatal(err)
	}
	if again.Name != s.Name || again.ReadOnly != s.ReadOnly || again.Approval != s.Approval {
		t.Fatalf("round trip = %+v, was %+v", again, s)
	}
}

func TestTheShortFormsOfATheConnectorAreRead(t *testing.T) {
	r, _ := decode(t, `{"model":"gpt-5","input":"hi","tools":[
		{"type":"mcp","server_label":"docs","server_url":"https://tools.test/mcp",
		 "allowed_tools":["read","write"],"require_approval":"always"}]}`)
	s := r.MCPServers[0]
	if len(s.AllowedTools) != 2 || s.Approval != llm.MCPApprovalAlways || len(s.ApprovalTools) != 0 {
		t.Fatalf("server = %+v", s)
	}
}

func TestAnotherVendorOnlyToolIsStillRefused(t *testing.T) {
	body := []byte(`{"model":"gpt-5","input":"hi","tools":[{"type":"web_search"}]}`)
	_, _, _, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Path: "/v1/responses", Body: body})
	if err == nil {
		t.Fatal("a tool that exists only on one backend must be refused, not dropped")
	}
}
