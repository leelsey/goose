// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"encoding/json"
	"sort"

	"github.com/leelsey/goose/llm"
)

func sortStrings(v []string) { sort.Strings(v) }

type wireMCPServer struct {
	Type              string `json:"type"`
	URL               string `json:"url"`
	Name              string `json:"name"`
	Token             string `json:"authorization_token"`
	ToolConfiguration *struct {
		Enabled      *bool    `json:"enabled"`
		AllowedTools []string `json:"allowed_tools"`
	} `json:"tool_configuration"`
}

type wireToolConfig struct {
	Enabled      *bool `json:"enabled"`
	DeferLoading *bool `json:"defer_loading"`
}

type wireToolset struct {
	Type         string                    `json:"type"`
	ServerName   string                    `json:"mcp_server_name"`
	Default      *wireToolConfig           `json:"default_config"`
	Configs      map[string]wireToolConfig `json:"configs"`
	CacheControl json.RawMessage           `json:"cache_control"`
}

const ToolsetType = "mcp_toolset"

func DecodeMCPServers(raw json.RawMessage) ([]llm.MCPServer, error) {
	var in []wireMCPServer
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "mcp_servers: %v", err)
	}
	out := make([]llm.MCPServer, 0, len(in))
	for _, s := range in {
		server := llm.MCPServer{Name: s.Name, URL: s.URL, Token: s.Token}
		if c := s.ToolConfiguration; c != nil {
			server.AllowedTools, server.Enabled = c.AllowedTools, c.Enabled
		}
		out = append(out, server)
	}
	return out, nil
}

func MergeMCPToolset(servers []llm.MCPServer, raw json.RawMessage) (unsent []string, err error) {
	var in wireToolset
	if err := json.Unmarshal(raw, &in); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "tools: %s: %v", ToolsetType, err)
	}
	at := -1
	for i := range servers {
		if servers[i].Name == in.ServerName {
			at = i
			break
		}
	}
	if at < 0 {
		return nil, llm.Errorf(llm.KindInvalidRequest,
			"tools: %s names %q, which is not a server in mcp_servers", ToolsetType, in.ServerName)
	}
	s := &servers[at]
	if in.Default != nil {
		s.DeferLoading = in.Default.DeferLoading
		if in.Default.Enabled != nil && !*in.Default.Enabled {
			off := false
			s.Enabled = &off
		}
	}
	for name, config := range in.Configs {
		if config.Enabled == nil {
			if config.DeferLoading != nil {
				unsent = append(unsent, "configs."+name+".defer_loading")
			}
			continue
		}
		if *config.Enabled {
			s.AllowedTools = append(s.AllowedTools, name)
		} else {
			s.DeniedTools = append(s.DeniedTools, name)
		}
		if config.DeferLoading != nil {
			unsent = append(unsent, "configs."+name+".defer_loading")
		}
	}
	if len(s.AllowedTools) > 0 {
		s.Enabled = nil
	}
	sortStrings(s.AllowedTools)
	sortStrings(s.DeniedTools)
	if len(in.CacheControl) > 0 && string(in.CacheControl) != "null" {
		unsent = append(unsent, "cache_control")
	}
	return unsent, nil
}

func EncodeMCPServers(list []llm.MCPServer) (servers, toolsets []map[string]any, unsent []string) {
	seen := map[string]bool{}
	note := func(field string) {
		if !seen[field] {
			seen[field] = true
			unsent = append(unsent, field)
		}
	}
	for _, s := range list {
		if s.URL == "" {
			note("connector_id")
			continue
		}
		entry := map[string]any{"type": "url", "url": s.URL, "name": s.Name}
		if s.Token != "" {
			entry["authorization_token"] = s.Token
		}
		servers = append(servers, entry)

		toolset := map[string]any{"type": ToolsetType, "mcp_server_name": s.Name}
		defaults := map[string]any{}
		configs := map[string]any{}
		switch {
		case s.Enabled != nil && !*s.Enabled:
			defaults["enabled"] = false
		case len(s.AllowedTools) > 0:
			defaults["enabled"] = false
			for _, name := range s.AllowedTools {
				configs[name] = map[string]any{"enabled": true}
			}
		}
		if s.Enabled == nil || *s.Enabled {
			for _, name := range s.DeniedTools {
				configs[name] = map[string]any{"enabled": false}
			}
		}
		if s.DeferLoading != nil {
			defaults["defer_loading"] = *s.DeferLoading
		}
		if len(defaults) > 0 {
			toolset["default_config"] = defaults
		}
		if len(configs) > 0 {
			toolset["configs"] = configs
		}
		toolsets = append(toolsets, toolset)

		if s.Description != "" {
			note("server_description")
		}
		if len(s.Headers) > 0 {
			note("headers")
		}
		if s.ReadOnly {
			note("read_only")
		}
		if s.Approval != llm.MCPApprovalDefault {
			note("require_approval")
		}
	}
	return servers, toolsets, unsent
}
