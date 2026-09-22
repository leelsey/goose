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
			// An unrecognised mode must not read as "no approval needed": a
			// backend that cannot wait for one refuses on exactly that signal.
			if word != string(llm.MCPApprovalAlways) && word != string(llm.MCPApprovalNever) {
				return llm.MCPServer{}, llm.Errorf(llm.KindInvalidRequest,
					"tools: mcp: require_approval is %q; this wire knows %q and %q",
					word, llm.MCPApprovalAlways, llm.MCPApprovalNever)
			}
			out.Approval = llm.MCPApproval(word)
		} else {
			var per map[string]struct {
				Names []string `json:"tool_names"`
			}
			if err := json.Unmarshal(in.Approval, &per); err != nil {
				return llm.MCPServer{}, llm.Errorf(llm.KindInvalidRequest, "tools: mcp: require_approval: %v", err)
			}
			always, hasAlways := per[string(llm.MCPApprovalAlways)]
			never, hasNever := per[string(llm.MCPApprovalNever)]
			switch {
			case hasAlways && hasNever:
				// One mode and one list is all this carries, so keeping half of
				// a policy would quietly let the other half through.
				return llm.MCPServer{}, llm.Errorf(llm.KindInvalidRequest,
					"tools: mcp: require_approval names both %q and %q, and this proxy carries one of them",
					llm.MCPApprovalAlways, llm.MCPApprovalNever)
			case hasAlways:
				out.Approval, out.ApprovalTools = llm.MCPApprovalAlways, nonNil(always.Names)
			case hasNever:
				out.Approval, out.ApprovalTools = llm.MCPApprovalNever, nonNil(never.Names)
			}
		}
	}
	return out, nil
}

// nonNil keeps an empty list distinct from an absent one: naming nothing is a
// setting, and reading it as "unset" widens what the caller asked for.
func nonNil(names []string) []string {
	if names == nil {
		return []string{}
	}
	return names
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
		if s.AllowedTools != nil {
			filter["tool_names"] = s.AllowedTools
		}
		out["allowed_tools"] = filter
	case s.AllowedTools != nil:
		// A list naming nothing allows nothing. Dropping it because it is empty
		// would hand the model every tool the server serves.
		out["allowed_tools"] = s.AllowedTools
	}
	if s.Approval != llm.MCPApprovalDefault {
		if s.ApprovalTools != nil {
			out["require_approval"] = map[string]any{
				string(s.Approval): map[string]any{"tool_names": s.ApprovalTools},
			}
		} else {
			out["require_approval"] = string(s.Approval)
		}
	}
	return out
}
