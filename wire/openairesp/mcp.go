// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"

	"github.com/leelsey/goose/llm"
)

type wireMCP struct {
	Label       string            `json:"server_label"`
	URL         string            `json:"server_url"`
	Connector   string            `json:"connector_id"`
	Description string            `json:"server_description"`
	Authorise   string            `json:"authorization"`
	Headers     map[string]string `json:"headers"`
	Allowed     json.RawMessage   `json:"allowed_tools"`
	Approval    json.RawMessage   `json:"require_approval"`
}

func DecodeMCPTool(raw json.RawMessage) (llm.MCPServer, error) {
	var in wireMCP
	if err := json.Unmarshal(raw, &in); err != nil {
		return llm.MCPServer{}, llm.Errorf(llm.KindInvalidRequest, "tools: mcp: %v", err)
	}
	out := llm.MCPServer{
		Name: in.Label, URL: in.URL, Connector: in.Connector,
		Description: in.Description, Token: in.Authorise, Headers: in.Headers,
	}
	if len(in.Allowed) > 0 && string(in.Allowed) != "null" {
		var names []string
		if err := json.Unmarshal(in.Allowed, &names); err == nil {
			out.AllowedTools = names
		} else {
			var filter struct {
				Names    []string `json:"tool_names"`
				ReadOnly bool     `json:"read_only"`
			}
			if err := json.Unmarshal(in.Allowed, &filter); err != nil {
				return llm.MCPServer{}, llm.Errorf(llm.KindInvalidRequest, "tools: mcp: allowed_tools: %v", err)
			}
			out.AllowedTools, out.ReadOnly = filter.Names, filter.ReadOnly
		}
	}
	if len(in.Approval) > 0 && string(in.Approval) != "null" {
		var word string
		if err := json.Unmarshal(in.Approval, &word); err == nil {
			out.Approval = llm.MCPApproval(word)
		} else {
			var per map[string]struct {
				Names []string `json:"tool_names"`
			}
			if err := json.Unmarshal(in.Approval, &per); err != nil {
				return llm.MCPServer{}, llm.Errorf(llm.KindInvalidRequest, "tools: mcp: require_approval: %v", err)
			}
			for _, mode := range []llm.MCPApproval{llm.MCPApprovalNever, llm.MCPApprovalAlways} {
				if entry, ok := per[string(mode)]; ok {
					out.Approval, out.ApprovalTools = mode, entry.Names
					break
				}
			}
		}
	}
	return out, nil
}

func EncodeMCPTool(s llm.MCPServer) map[string]any {
	out := map[string]any{"type": "mcp", "server_label": s.Name}
	if s.URL != "" {
		out["server_url"] = s.URL
	}
	if s.Connector != "" {
		out["connector_id"] = s.Connector
	}
	if s.Description != "" {
		out["server_description"] = s.Description
	}
	if s.Token != "" {
		out["authorization"] = s.Token
	}
	if len(s.Headers) > 0 {
		out["headers"] = s.Headers
	}
	switch {
	case s.ReadOnly:
		filter := map[string]any{"read_only": true}
		if len(s.AllowedTools) > 0 {
			filter["tool_names"] = s.AllowedTools
		}
		out["allowed_tools"] = filter
	case len(s.AllowedTools) > 0:
		out["allowed_tools"] = s.AllowedTools
	}
	if s.Approval != llm.MCPApprovalDefault {
		if len(s.ApprovalTools) > 0 {
			out["require_approval"] = map[string]any{
				string(s.Approval): map[string]any{"tool_names": s.ApprovalTools},
			}
		} else {
			out["require_approval"] = string(s.Approval)
		}
	}
	return out
}
