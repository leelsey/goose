// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func TestAToolServerTravelsWithTheBetaThatOpensIt(t *testing.T) {
	var seen http.Header
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		b, _ := io.ReadAll(r.Body)
		body = string(b)
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()
	p, err := New(Options{BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}

	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model:    "claude-test",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		MCPServers: []llm.MCPServer{{
			Name: "docs", URL: "https://tools.test/mcp", Token: "tok",
			AllowedTools: []string{"read"},
			Description:  "the docs", Headers: map[string]string{"X-Team": "core"},
		}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(seen.Values("Anthropic-Beta"), ","), mcpBeta) {
		t.Fatalf("beta = %v: the field is refused without it", seen.Values("Anthropic-Beta"))
	}

	var sent struct {
		Servers []struct {
			Type  string `json:"type"`
			URL   string `json:"url"`
			Name  string `json:"name"`
			Token string `json:"authorization_token"`
		} `json:"mcp_servers"`
		Tools []struct {
			Type       string `json:"type"`
			ServerName string `json:"mcp_server_name"`
			Default    struct {
				Enabled *bool `json:"enabled"`
			} `json:"default_config"`
			Configs map[string]struct {
				Enabled *bool `json:"enabled"`
			} `json:"configs"`
		} `json:"tools"`
	}
	if err := json.Unmarshal([]byte(body), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Servers) != 1 {
		t.Fatalf("body = %s", body)
	}
	s := sent.Servers[0]
	if s.Type != "url" || s.URL != "https://tools.test/mcp" || s.Name != "docs" || s.Token != "tok" {
		t.Fatalf("server = %+v", s)
	}
	if len(sent.Tools) != 1 || sent.Tools[0].Type != "mcp_toolset" || sent.Tools[0].ServerName != "docs" {
		t.Fatalf("tools = %+v", sent.Tools)
	}
	set := sent.Tools[0]
	if set.Default.Enabled == nil || *set.Default.Enabled {
		t.Fatalf("toolset = %+v: an allowlist switches the rest off", set)
	}
	if c, ok := set.Configs["read"]; !ok || c.Enabled == nil || !*c.Enabled {
		t.Fatalf("toolset = %+v", set)
	}
	if !resp.Diagnostics.Has(llm.DropMCPServer) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestAHostedConnectorIsNotSentAsAServer(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model:      "claude-test",
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		MCPServers: []llm.MCPServer{{Name: "drive", Connector: "connector_googledrive"}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains((*bodies)[0], "mcp_servers") {
		t.Fatalf("body = %s: a server with no url is refused by this backend", (*bodies)[0])
	}
	if !resp.Diagnostics.Has(llm.DropMCPServer) {
		t.Fatalf("diagnostics = %+v: it went missing without a word", resp.Diagnostics)
	}
}

func TestAServerWithNoOtherToolsStillCarriesItsToolset(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	if _, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model:      "claude-test",
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		MCPServers: []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp"}},
	})); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Tools []struct {
			Type       string `json:"type"`
			ServerName string `json:"mcp_server_name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Tools) != 1 || sent.Tools[0].ServerName != "docs" {
		t.Fatalf("body = %s", (*bodies)[0])
	}
}

func TestARestrictionThisBackendCannotHonourIsRefused(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"msg_1","model":"claude-test","content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	ask := func(s llm.MCPServer) error {
		_, err := p.Complete(context.Background(), attempt(&llm.Request{
			Model:      "claude-test",
			Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
			MCPServers: []llm.MCPServer{s},
		}))
		return err
	}
	base := llm.MCPServer{Name: "docs", URL: "https://tools.test/mcp"}
	for _, s := range []llm.MCPServer{
		{Name: base.Name, URL: base.URL, Approval: llm.MCPApprovalAlways},
		{Name: base.Name, URL: base.URL, Approval: llm.MCPApprovalNever, ApprovalTools: []string{"read"}},
		{Name: base.Name, URL: base.URL, ReadOnly: true},
	} {
		if err := ask(s); err == nil {
			t.Fatalf("server %+v was served by a backend that cannot honour it", s)
		}
	}
	if len(*bodies) != 0 {
		t.Fatalf("a refused request was sent anyway: %v", *bodies)
	}

	if err := ask(llm.MCPServer{Name: base.Name, URL: base.URL, Approval: llm.MCPApprovalNever}); err != nil {
		t.Fatalf("a server that asks for no approval gate is servable here: %v", err)
	}
}
