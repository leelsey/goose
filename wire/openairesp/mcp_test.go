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

// A restriction the caller asked for must never widen on the way through. The
// empty list is the whole point: "none of this server's tools" is a setting,
// not the absence of one.
func TestAToolFilterSurvivesTheRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name, in string
		want     any
	}{
		{"none allowed", `{"type":"mcp","server_label":"x","server_url":"https://h/","allowed_tools":[]}`, []string{}},
		{"two allowed", `{"type":"mcp","server_label":"x","server_url":"https://h/","allowed_tools":["a","b"]}`, []string{"a", "b"}},
		{"none allowed, read only", `{"type":"mcp","server_label":"x","server_url":"https://h/","allowed_tools":{"read_only":true,"tool_names":[]}}`, nil},
		{"unset", `{"type":"mcp","server_label":"x","server_url":"https://h/"}`, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server, err := DecodeMCPTool(json.RawMessage(tc.in))
			if err != nil {
				t.Fatal(err)
			}
			out := EncodeMCPTool(server)
			got, set := out["allowed_tools"]
			switch want := tc.want.(type) {
			case nil:
				if tc.name == "unset" && set {
					t.Fatalf("allowed_tools = %v, want nothing: the caller set no filter", got)
				}
				if tc.name != "unset" && !set {
					t.Fatal("the read-only filter was dropped")
				}
			case []string:
				if !set {
					t.Fatalf("allowed_tools was dropped, so %q became every tool this server serves", tc.in)
				}
				names, ok := got.([]string)
				if !ok || len(names) != len(want) {
					t.Fatalf("allowed_tools = %#v, want %#v", got, want)
				}
			}
		})
	}
}

// An approval mode goose does not recognise must not read as "no approval
// needed": a backend that cannot wait for one refuses on exactly that signal.
func TestAnUnknownApprovalModeIsRefused(t *testing.T) {
	for _, word := range []string{"Always", "ALWAYS", "alway", "banana", ""} {
		raw := `{"type":"mcp","server_label":"x","server_url":"https://h/","require_approval":"` + word + `"}`
		server, err := DecodeMCPTool(json.RawMessage(raw))
		if err == nil {
			t.Errorf("require_approval %q was accepted as %q, and needs approval = %v",
				word, server.Approval, server.NeedsApproval())
		}
	}
	for _, word := range []string{"always", "never"} {
		raw := `{"type":"mcp","server_label":"x","server_url":"https://h/","require_approval":"` + word + `"}`
		if _, err := DecodeMCPTool(json.RawMessage(raw)); err != nil {
			t.Errorf("require_approval %q is a mode this wire defines: %v", word, err)
		}
	}
}

// Taking the permissive half of a policy and dropping the other is the one
// answer that must not be given.
func TestAnApprovalPolicyIsNotHalfApplied(t *testing.T) {
	both := `{"type":"mcp","server_label":"x","server_url":"https://h/",
		"require_approval":{"always":{"tool_names":["delete_everything"]},"never":{"tool_names":["search"]}}}`
	if server, err := DecodeMCPTool(json.RawMessage(both)); err == nil {
		t.Fatalf("both halves were asked for and one was kept: %+v", server)
	}

	only := `{"type":"mcp","server_label":"x","server_url":"https://h/",
		"require_approval":{"always":{"tool_names":["delete_everything"]}}}`
	server, err := DecodeMCPTool(json.RawMessage(only))
	if err != nil {
		t.Fatal(err)
	}
	if !server.NeedsApproval() {
		t.Fatalf("%+v: a named always list is a request for approval", server)
	}

	// "never" naming nothing exempts nothing, so everything still waits.
	exempt := `{"type":"mcp","server_label":"x","server_url":"https://h/",
		"require_approval":{"never":{"tool_names":[]}}}`
	server, err = DecodeMCPTool(json.RawMessage(exempt))
	if err != nil {
		t.Fatal(err)
	}
	if !server.NeedsApproval() {
		t.Fatalf("%+v: an empty exemption list exempts nothing", server)
	}
}
