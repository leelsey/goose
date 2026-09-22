// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func inbound(body string) *wire.Inbound {
	return &wire.Inbound{Method: "POST", Body: []byte(body)}
}

func jsonOf(v any) (json.RawMessage, error) { return json.Marshal(v) }

func TestAToolServerAndItsToolsetAreReadTogether(t *testing.T) {
	r, _, _ := decode(t, `{"model":"claude","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
		"mcp_servers":[{"type":"url","url":"https://tools.test/mcp","name":"docs","authorization_token":"tok"}],
		"tools":[{"name":"f","input_schema":{"type":"object"}},
		 {"type":"mcp_toolset","mcp_server_name":"docs",
		  "default_config":{"enabled":false,"defer_loading":true},
		  "configs":{"read":{"enabled":true},"write":{"enabled":true}}}]}`)

	if len(r.Tools) != 1 || r.Tools[0].Name != "f" {
		t.Fatalf("tools = %+v: a toolset is not a function the client runs", r.Tools)
	}
	if len(r.MCPServers) != 1 {
		t.Fatalf("servers = %+v", r.MCPServers)
	}
	s := r.MCPServers[0]
	switch {
	case s.Name != "docs" || s.URL != "https://tools.test/mcp" || s.Token != "tok":
		t.Fatalf("server = %+v", s)
	case len(s.AllowedTools) != 2 || s.AllowedTools[0] != "read":
		t.Fatalf("allowed = %+v: the named tools are an allowlist", s)
	case s.Enabled != nil:
		t.Fatalf("server = %+v: a default of off with tools named is a narrowing, not a server switched off", s)
	case s.DeferLoading == nil || !*s.DeferLoading:
		t.Fatalf("server = %+v", s)
	}
}

func TestToolsRefusedOnAServerAreKept(t *testing.T) {
	r, _, _ := decode(t, `{"model":"claude","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
		"mcp_servers":[{"type":"url","url":"https://tools.test/mcp","name":"docs"}],
		"tools":[{"type":"mcp_toolset","mcp_server_name":"docs","configs":{"delete_all":{"enabled":false}}}]}`)
	s := r.MCPServers[0]
	if len(s.DeniedTools) != 1 || s.DeniedTools[0] != "delete_all" || len(s.AllowedTools) != 0 {
		t.Fatalf("server = %+v", s)
	}
}

func TestAServerSwitchedOffIsKept(t *testing.T) {
	r, _, _ := decode(t, `{"model":"claude","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
		"mcp_servers":[{"type":"url","url":"https://tools.test/mcp","name":"docs"}],
		"tools":[{"type":"mcp_toolset","mcp_server_name":"docs","default_config":{"enabled":false}}]}`)
	s := r.MCPServers[0]
	if s.Enabled == nil || *s.Enabled {
		t.Fatalf("server = %+v", s)
	}
}

func TestTheSupersededSpellingIsStillRead(t *testing.T) {
	r, _, _ := decode(t, `{"model":"claude","max_tokens":16,"messages":[{"role":"user","content":"hi"}],
		"mcp_servers":[{"type":"url","url":"https://tools.test/mcp","name":"docs",
		 "tool_configuration":{"enabled":true,"allowed_tools":["read"]}}]}`)
	s := r.MCPServers[0]
	if len(s.AllowedTools) != 1 || s.Enabled == nil || !*s.Enabled {
		t.Fatalf("server = %+v", s)
	}
}

func TestAToolsetMustNameAServer(t *testing.T) {
	_, _, _, err := New(Options{}).Decode(inbound(`{"model":"claude","max_tokens":16,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"mcp_toolset","mcp_server_name":"missing"}]}`))
	if err == nil {
		t.Fatal("a toolset for a server that was never defined must be refused")
	}
}

func TestWhatThisWireCannotSayAboutAServerIsNamed(t *testing.T) {
	servers, toolsets, unsent := EncodeMCPServers([]llm.MCPServer{
		{Name: "docs", URL: "https://tools.test/mcp", Description: "d",
			Headers: map[string]string{"X-Team": "core"}, ReadOnly: true, Approval: llm.MCPApprovalAlways},
		{Name: "drive", Connector: "connector_googledrive"},
	})
	if len(servers) != 1 || len(toolsets) != 1 {
		t.Fatalf("servers = %+v, toolsets = %+v: every server is named by exactly one toolset", servers, toolsets)
	}
	want := map[string]bool{
		"docs: server_description": true,
		"docs: headers":            true,
		"docs: read_only":          true,
		"docs: require_approval":   true,
		"drive: connector_id":      true,
	}
	if len(unsent) != len(want) {
		t.Fatalf("unsent = %v", unsent)
	}
	for _, u := range unsent {
		if !want[u] {
			t.Fatalf("unsent = %v", unsent)
		}
	}
}

func TestEveryServerThisWireCannotCarryIsNamed(t *testing.T) {
	_, _, unsent := EncodeMCPServers([]llm.MCPServer{
		{Name: "drive", Connector: "connector_googledrive"},
		{Name: "mail", Connector: "connector_gmail"},
	})
	if len(unsent) != 2 {
		t.Fatalf("unsent = %v: two whole servers went missing, not one", unsent)
	}
	if unsent[0] == unsent[1] {
		t.Fatalf("unsent = %v: the caller cannot tell which servers were dropped", unsent)
	}
}

func TestPerToolApprovalIsNamedAsUnsent(t *testing.T) {
	_, _, unsent := EncodeMCPServers([]llm.MCPServer{
		{Name: "docs", URL: "https://tools.test/mcp", ApprovalTools: []string{"delete_file"}},
	})
	found := false
	for _, u := range unsent {
		if strings.HasSuffix(u, "require_approval") {
			found = true
		}
	}
	if !found {
		t.Fatalf("unsent = %v: an approval this wire cannot ask for must not vanish quietly", unsent)
	}
}

func TestAToolsetSurvivesTheRoundTrip(t *testing.T) {
	off := false
	for _, s := range []llm.MCPServer{
		{Name: "docs", URL: "https://tools.test/mcp", AllowedTools: []string{"read"}},
		{Name: "docs", URL: "https://tools.test/mcp", DeniedTools: []string{"delete"}},
		{Name: "docs", URL: "https://tools.test/mcp", Enabled: &off},
	} {
		servers, toolsets, _ := EncodeMCPServers([]llm.MCPServer{s})
		raw, err := jsonOf(toolsets[0])
		if err != nil {
			t.Fatal(err)
		}
		back := []llm.MCPServer{{Name: servers[0]["name"].(string), URL: s.URL}}
		if _, err := MergeMCPToolset(back, raw); err != nil {
			t.Fatal(err)
		}
		got := back[0]
		switch {
		case len(got.AllowedTools) != len(s.AllowedTools):
			t.Fatalf("allowed = %+v, was %+v", got, s)
		case len(got.DeniedTools) != len(s.DeniedTools):
			t.Fatalf("denied = %+v, was %+v", got, s)
		case (got.Enabled == nil) != (s.Enabled == nil):
			t.Fatalf("enabled = %+v, was %+v", got, s)
		}
	}
}

// A list naming nothing allows nothing. Reading it as "no list" hands the model
// every tool the server serves.
func TestAnEmptyAllowListAllowsNothing(t *testing.T) {
	servers, toolsets, _ := EncodeMCPServers([]llm.MCPServer{
		{Name: "fs", URL: "https://h/", AllowedTools: []string{}},
	})
	if len(servers) != 1 || len(toolsets) != 1 {
		t.Fatalf("servers = %d, toolsets = %d", len(servers), len(toolsets))
	}
	raw, err := json.Marshal(toolsets[0])
	if err != nil {
		t.Fatal(err)
	}
	var out struct {
		Defaults struct {
			Enabled *bool `json:"enabled"`
		} `json:"default_config"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if out.Defaults.Enabled == nil || *out.Defaults.Enabled {
		t.Fatalf("toolset = %s: an empty allow list left every tool on", raw)
	}
}
