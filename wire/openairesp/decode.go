// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"
	"strings"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

var known = map[string]bool{
	"model": true, "input": true, "instructions": true, "stream": true,
	"max_output_tokens": true, "temperature": true, "top_p": true,
	"tools": true, "tool_choice": true, "reasoning": true, "text": true,
	"previous_response_id": true, "metadata": true,
	"conversation": true,
}

var (
	modelledReasoning = map[string]bool{"effort": true}
	modelledText      = map[string]bool{"format": true}
)

func carry(req *llm.Request, key string, rest json.RawMessage) {
	if req.Extra == nil {
		req.Extra = map[string]json.RawMessage{}
		req.ExtraFrom = Name
	}
	req.Extra[key] = rest
}

func (c *Codec) Decode(in *wire.Inbound) (*llm.Request, wire.Options, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	opts := wire.Options{Format: Name, Header: in.Header}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(in.Body, &top); err != nil {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "malformed JSON body: %v", err)
	}
	if raw, ok := top["conversation"]; ok && string(raw) != "null" {
		return nil, opts, diags, llm.Errorf(llm.KindUnsupported, "the conversation object is not supported; use previous_response_id")
	}

	req := &llm.Request{}
	if err := field(top, "model", &req.Model); err != nil {
		return nil, opts, diags, err
	}
	opts.RequestModel = req.Model

	var instructions string
	if err := field(top, "instructions", &instructions); err != nil {
		return nil, opts, diags, err
	}
	if instructions != "" {
		req.System = []llm.Part{&llm.Text{Text: instructions}}
	}

	msgs, d, err := decodeInput(top["input"])
	if err != nil {
		return nil, opts, diags, err
	}
	diags = append(diags, d...)
	req.Messages = msgs

	if err := field(top, "stream", &opts.Stream); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "max_output_tokens", &req.MaxOutputTokens); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "temperature", &req.Temperature); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "top_p", &req.TopP); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "previous_response_id", &opts.Session); err != nil {
		return nil, opts, diags, err
	}

	if raw, ok := top["reasoning"]; ok && string(raw) != "null" {
		for _, name := range wirecore.Shadowed(raw, modelledReasoning) {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityLoss, Format: Name, Field: "reasoning" + "." + name,
				Detail: "this setting arrived spelled more than one way, and only the last of them was read",
			})
		}
		if rest := wirecore.NestedExtra(raw, modelledReasoning); rest != nil {
			carry(req, "reasoning", rest)
		}
		var r struct {
			Effort  string `json:"effort"`
			Summary string `json:"summary"`
		}
		if err := json.Unmarshal(raw, &r); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "reasoning: %v", err)
		}
		req.Effort = llm.Effort(r.Effort)
		if r.Effort != "" || r.Summary != "" {
			req.Thinking = &llm.ThinkingConfig{Enabled: true}
		}
	}

	if raw, ok := top["text"]; ok && string(raw) != "null" {
		for _, name := range wirecore.Shadowed(raw, modelledText) {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityLoss, Format: Name, Field: "text." + name,
				Detail: "this setting arrived spelled more than one way, and only the last of them was read",
			})
		}
		if rest := wirecore.NestedExtra(raw, modelledText); rest != nil {
			carry(req, "text", rest)
		}
		var txt struct {
			Format struct {
				Type   string          `json:"type"`
				Name   string          `json:"name"`
				Schema json.RawMessage `json:"schema"`
				Strict bool            `json:"strict"`
			} `json:"format"`
		}
		if err := json.Unmarshal(raw, &txt); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "text: %v", err)
		}
		if txt.Format.Type != "" && txt.Format.Type != "text" {
			req.ResponseFormat = &llm.ResponseFormat{
				Type: txt.Format.Type, Name: txt.Format.Name,
				Schema: txt.Format.Schema, Strict: txt.Format.Strict,
			}
		}
	}

	if raw, ok := top["tools"]; ok && string(raw) != "null" {
		var entries []json.RawMessage
		if err := json.Unmarshal(raw, &entries); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
		}
		for _, entry := range entries {
			var head struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(entry, &head); err != nil {
				return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
			}
			if head.Type == "mcp" {
				server, err := DecodeMCPTool(entry)
				if err != nil {
					return nil, opts, diags, err
				}
				req.MCPServers = append(req.MCPServers, server)
				continue
			}
			var t struct {
				Type        string          `json:"type"`
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
				Strict      bool            `json:"strict"`
			}
			if err := json.Unmarshal(entry, &t); err != nil {
				return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
			}
			if builtIn[t.Type] {
				return nil, opts, diags, llm.Errorf(llm.KindUnsupported, "tool type %q exists only on one provider; use a function tool", t.Type)
			}
			req.Tools = append(req.Tools, llm.Tool{
				Name: t.Name, Description: t.Description, Schema: t.Parameters, Strict: t.Strict,
			})
		}
	}
	if raw, ok := top["tool_choice"]; ok && string(raw) != "null" {
		tc, err := decodeToolChoice(raw)
		if err != nil {
			return nil, opts, diags, err
		}
		req.ToolChoice = tc
	}

	if raw, ok := top["metadata"]; ok && string(raw) != "null" {
		var meta map[string]string
		if err := json.Unmarshal(raw, &meta); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "metadata: %v", err)
		}
		for k, v := range meta {
			if req.Metadata == nil {
				req.Metadata = map[string]string{}
			}
			req.Metadata[k] = v
		}
	}

	for k, v := range top {
		if known[k] {
			continue
		}
		if req.Extra == nil {
			req.Extra = map[string]json.RawMessage{}
			req.ExtraFrom = Name
		}
		req.Extra[k] = v
	}
	opts.Key = bearer(in.Get("Authorization"))
	return req, opts, diags, nil
}

func decodeInput(raw json.RawMessage) ([]llm.Message, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	if len(raw) == 0 || string(raw) == "null" {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "input is required")
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: text}}}}, diags, nil
	}

	var items []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "input: %v", err)
	}

	var msgs []llm.Message
	appendTo := func(role llm.Role, parts ...llm.Part) {
		if n := len(msgs); n > 0 && msgs[n-1].Role == role {
			msgs[n-1].Content = append(msgs[n-1].Content, parts...)
			return
		}
		msgs = append(msgs, llm.Message{Role: role, Content: parts})
	}

	for _, item := range items {
		var kind string
		_ = field(item, "type", &kind)
		switch kind {
		case "", "message":
			var role string
			_ = field(item, "role", &role)
			if role == "" {
				role = "user"
			}
			parts, d, err := decodeContent(item["content"])
			if err != nil {
				return nil, diags, err
			}
			diags = append(diags, d...)
			appendTo(llm.Role(role), parts...)
		case "function_call":
			var fc struct {
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}
			b, _ := json.Marshal(item)
			_ = json.Unmarshal(b, &fc)
			appendTo(llm.RoleAssistant, &llm.ToolUse{ID: fc.CallID, Name: fc.Name, Args: json.RawMessage(fc.Arguments)})
		case "function_call_output":
			var callID string
			_ = field(item, "call_id", &callID)
			parts, d, err := decodeToolOutput(item["output"])
			if err != nil {
				return nil, diags, err
			}
			diags = append(diags, d...)
			appendTo(llm.RoleTool, &llm.ToolResult{ID: callID, Content: parts})
		case "reasoning":
			var r struct {
				Summary []struct {
					Text string `json:"text"`
				} `json:"summary"`
				EncryptedContent string `json:"encrypted_content"`
			}
			b, _ := json.Marshal(item)
			_ = json.Unmarshal(b, &r)
			th := &llm.Thinking{Signature: r.EncryptedContent}
			var sb strings.Builder
			for _, s := range r.Summary {
				sb.WriteString(s.Text)
			}
			th.Text = sb.String()
			appendTo(llm.RoleAssistant, th)
		case "input_file":
			return nil, diags, llm.Errorf(llm.KindUnsupported, "input_file has no analogue on other providers")
		default:
			b, _ := json.Marshal(item)
			appendTo(llm.RoleUser, &llm.Raw{Vendor: Name, Type: kind, JSON: b})
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityInfo, Format: Name, Field: kind, Detail: "input item carried verbatim",
			})
		}
	}
	return msgs, diags, nil
}

func decodeToolOutput(raw json.RawMessage) ([]llm.Part, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	if len(raw) == 0 || string(raw) == "null" {
		return nil, diags, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nil, diags, nil
		}
		return []llm.Part{&llm.Text{Text: text}}, diags, nil
	}
	if len(raw) > 0 && raw[0] == '[' {
		return decodeContent(raw)
	}
	return []llm.Part{&llm.Text{Text: string(raw)}}, diags, nil
}

func decodeContent(raw json.RawMessage) ([]llm.Part, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	if len(raw) == 0 || string(raw) == "null" {
		return nil, diags, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nil, diags, nil
		}
		return []llm.Part{&llm.Text{Text: text}}, diags, nil
	}
	var items []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "content: %v", err)
	}
	var parts []llm.Part
	for _, item := range items {
		var kind string
		_ = field(item, "type", &kind)
		switch kind {
		case "input_text", "output_text", "text":
			var s string
			_ = field(item, "text", &s)
			if s != "" {
				parts = append(parts, &llm.Text{Text: s})
			}
		case "refusal":
			var s string
			_ = field(item, "refusal", &s)
			if s != "" {
				parts = append(parts, &llm.Refusal{Text: s})
			}
		case "input_image":
			var url, detail string
			_ = field(item, "image_url", &url)
			_ = field(item, "detail", &detail)
			img := &llm.Image{URL: url, Detail: detail}
			if mime, data, ok := dataURL(url); ok {
				img.MIME, img.Data, img.URL = mime, data, ""
			}
			parts = append(parts, img)
		default:
			b, _ := json.Marshal(item)
			parts = append(parts, &llm.Raw{Vendor: Name, Type: kind, JSON: b})
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityInfo, Format: Name, Field: kind, Detail: "content part carried verbatim",
			})
		}
	}
	return parts, diags, nil
}

func decodeToolChoice(raw json.RawMessage) (*llm.ToolChoice, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto":
			return &llm.ToolChoice{Mode: llm.ToolAuto}, nil
		case "none":
			return &llm.ToolChoice{Mode: llm.ToolNone}, nil
		case "required":
			return &llm.ToolChoice{Mode: llm.ToolRequired}, nil
		default:
			return nil, llm.Errorf(llm.KindInvalidRequest, "tool_choice: unknown value %q", s)
		}
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "tool_choice: %v", err)
	}
	if builtIn[obj.Type] {
		return nil, llm.Errorf(llm.KindUnsupported, "tool_choice names a provider-specific tool %q", obj.Type)
	}
	return &llm.ToolChoice{Mode: llm.ToolSpecific, Name: obj.Name}, nil
}

func dataURL(s string) (mime string, data []byte, ok bool) { return wirecore.DataURL(s) }

func bearer(h string) string { return wirecore.Bearer(h) }

func field[T any](m map[string]json.RawMessage, key string, dst *T) error {
	return wirecore.Field(m, key, dst)
}
