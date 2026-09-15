// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropicmsg

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

var known = map[string]bool{
	"model": true, "messages": true, "system": true, "max_tokens": true,
	"stream": true, "temperature": true, "top_p": true, "top_k": true,
	"stop_sequences": true, "tools": true, "tool_choice": true,
	"thinking": true, "metadata": true, "mcp_servers": true,
}

type cacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl"`
}

func (c *cacheControl) hint() *llm.CacheHint {
	if c == nil {
		return nil
	}
	h := &llm.CacheHint{}
	if d, err := time.ParseDuration(c.TTL); err == nil {
		h.TTL = d
	}
	return h
}

func (c *Codec) Decode(in *wire.Inbound) (*llm.Request, wire.Options, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	opts := wire.Options{Format: Name, Header: in.Header}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(in.Body, &top); err != nil {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "malformed JSON body: %v", err)
	}

	req := &llm.Request{}
	if err := field(top, "model", &req.Model); err != nil {
		return nil, opts, diags, err
	}
	opts.RequestModel = req.Model

	if raw, ok := top["system"]; ok && string(raw) != "null" {
		parts, hint, d, err := decodeBlocks(raw)
		if err != nil {
			return nil, opts, diags, fmt.Errorf("system: %w", err)
		}
		diags = append(diags, d...)
		req.System = parts
		req.SystemCache = hint
	}

	var msgs []struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
		Cache   *cacheControl   `json:"cache_control"`
	}
	if err := field(top, "messages", &msgs); err != nil {
		return nil, opts, diags, err
	}
	if len(msgs) == 0 {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "messages is required")
	}
	for i, m := range msgs {
		if m.Role != "user" && m.Role != "assistant" {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "messages[%d]: unknown role %q", i, m.Role)
		}
		parts, hint, d, err := decodeBlocks(m.Content)
		if err != nil {
			return nil, opts, diags, fmt.Errorf("messages[%d]: %w", i, err)
		}
		diags = append(diags, d...)
		if hint == nil {
			hint = m.Cache.hint()
		}
		req.Messages = append(req.Messages, llm.Message{Role: llm.Role(m.Role), Content: parts, Cache: hint})
	}

	var maxTokens int
	if err := field(top, "max_tokens", &maxTokens); err != nil {
		return nil, opts, diags, err
	}
	if maxTokens <= 0 {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "max_tokens is required")
	}
	req.MaxOutputTokens = &maxTokens

	if err := field(top, "stream", &opts.Stream); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "temperature", &req.Temperature); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "top_p", &req.TopP); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "top_k", &req.TopK); err != nil {
		return nil, opts, diags, err
	}
	if err := field(top, "stop_sequences", &req.Stop); err != nil {
		return nil, opts, diags, err
	}

	if raw, ok := top["mcp_servers"]; ok && string(raw) != "null" {
		servers, err := DecodeMCPServers(raw)
		if err != nil {
			return nil, opts, diags, err
		}
		req.MCPServers = servers
	}

	if raw, ok := top["tools"]; ok && string(raw) != "null" {
		var entries []json.RawMessage
		if err := json.Unmarshal(raw, &entries); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
		}
		for _, entry := range entries {
			var t struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Schema      json.RawMessage `json:"input_schema"`
				Type        string          `json:"type"`
				Cache       *cacheControl   `json:"cache_control"`
			}
			if err := json.Unmarshal(entry, &t); err != nil {
				return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
			}
			if t.Type == ToolsetType {
				unsent, err := MergeMCPToolset(req.MCPServers, entry)
				if err != nil {
					return nil, opts, diags, err
				}
				for _, field := range unsent {
					diags = append(diags, llm.Diagnostic{
						Code: llm.DropMCPServer, Class: llm.ClassCapability, Stage: llm.StageDecode,
						Severity: llm.SeverityInfo, Format: Name, Field: field,
						Detail: "this setting on a tool server is not carried between backends, so it reaches only one of them",
					})
				}
				continue
			}
			if t.Type != "" && !strings.HasPrefix(t.Type, "custom") {
				return nil, opts, diags, llm.Errorf(llm.KindUnsupported, "tool type %q is not portable across providers", t.Type)
			}
			req.Tools = append(req.Tools, llm.Tool{
				Name: t.Name, Description: t.Description, Schema: t.Schema, Cache: t.Cache.hint(),
			})
		}
	}
	if raw, ok := top["tool_choice"]; ok && string(raw) != "null" {
		var tc struct {
			Type            string `json:"type"`
			Name            string `json:"name"`
			DisableParallel *bool  `json:"disable_parallel_tool_use"`
		}
		if err := json.Unmarshal(raw, &tc); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tool_choice: %v", err)
		}
		if tc.DisableParallel != nil {
			parallel := !*tc.DisableParallel
			req.ParallelTools = &parallel
		}
		switch tc.Type {
		case "auto":
			req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolAuto}
		case "any":
			req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolRequired}
		case "none":
			req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolNone}
		case "tool":
			req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolSpecific, Name: tc.Name}
		default:
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tool_choice: unknown type %q", tc.Type)
		}
	}

	if raw, ok := top["thinking"]; ok && string(raw) != "null" {
		var th struct {
			Type   string `json:"type"`
			Budget int    `json:"budget_tokens"`
		}
		if err := json.Unmarshal(raw, &th); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "thinking: %v", err)
		}
		req.Thinking = &llm.ThinkingConfig{Enabled: th.Type == "enabled", Budget: th.Budget}
		if th.Type != "enabled" && th.Type != "disabled" {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityLoss, Format: Name, Field: "thinking.type",
				Detail: "thinking mode " + quoted(th.Type) + " is not one this format knows, so thinking was not enabled",
			})
		}
	}

	if raw, ok := top["metadata"]; ok && string(raw) != "null" {
		var md map[string]string
		if err := json.Unmarshal(raw, &md); err == nil {
			for k, v := range md {
				req.Metadata = setMeta(req.Metadata, k, v)
			}
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

	opts.Key = in.Get("x-api-key")
	if opts.Key == "" {
		opts.Key = bearer(in.Get("Authorization"))
	}
	return req, opts, diags, nil
}

func DecodeBlocks(raw json.RawMessage) ([]llm.Part, *llm.CacheHint, llm.Diagnostics, error) {
	return decodeBlocks(raw)
}

func decodeBlocks(raw json.RawMessage) ([]llm.Part, *llm.CacheHint, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil, diags, nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nil, nil, diags, nil
		}
		return []llm.Part{&llm.Text{Text: text}}, nil, diags, nil
	}

	var items []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, nil, diags, llm.Errorf(llm.KindInvalidRequest, "content: %v", err)
	}
	var parts []llm.Part
	var hint *llm.CacheHint
	for i, item := range items {
		var kind string
		_ = field(item, "type", &kind)
		if raw, ok := item["cache_control"]; ok && string(raw) != "null" {
			var cc cacheControl
			if err := json.Unmarshal(raw, &cc); err == nil {
				if i == len(items)-1 {
					hint = cc.hint()
				} else {
					diags = append(diags, llm.Diagnostic{
						Code: llm.DropCacheHint, Class: llm.ClassCapability, Stage: llm.StageDecode,
						Severity: llm.SeverityInfo, Format: Name, Count: i,
						Detail: "a breakpoint on a block other than the last one cannot be expressed, and was not extended to cover more than was asked",
					})
				}
			}
		}
		switch kind {
		case "text":
			var s string
			_ = field(item, "text", &s)
			if s != "" {
				parts = append(parts, &llm.Text{Text: s})
			}
		case "thinking":
			var t struct {
				Thinking  string `json:"thinking"`
				Signature string `json:"signature"`
			}
			b, _ := json.Marshal(item)
			_ = json.Unmarshal(b, &t)
			parts = append(parts, &llm.Thinking{Text: t.Thinking, Signature: t.Signature})
		case "redacted_thinking":
			var d string
			_ = field(item, "data", &d)
			parts = append(parts, &llm.Thinking{Redacted: []byte(d)})
		case "tool_use":
			var t struct {
				ID    string          `json:"id"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			}
			b, _ := json.Marshal(item)
			_ = json.Unmarshal(b, &t)
			parts = append(parts, &llm.ToolUse{ID: t.ID, Name: t.Name, Args: t.Input})
		case "tool_result":
			var t struct {
				ID      string          `json:"tool_use_id"`
				IsError bool            `json:"is_error"`
				Content json.RawMessage `json:"content"`
			}
			b, _ := json.Marshal(item)
			_ = json.Unmarshal(b, &t)
			inner, _, d, err := decodeBlocks(t.Content)
			if err != nil {
				return nil, nil, diags, err
			}
			diags = append(diags, d...)
			parts = append(parts, &llm.ToolResult{ID: t.ID, Content: inner, IsError: t.IsError})
		case "image":
			img, err := decodeImage(item)
			if err != nil {
				return nil, nil, diags, err
			}
			parts = append(parts, img)
		default:
			b, _ := json.Marshal(item)
			parts = append(parts, &llm.Raw{Vendor: Name, Type: kind, JSON: b})
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityInfo, Format: Name, Field: kind,
				Detail: "content block carried verbatim",
			})
		}
	}
	return parts, hint, diags, nil
}

func decodeImage(item map[string]json.RawMessage) (*llm.Image, error) {
	var src struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	}
	if raw, ok := item["source"]; ok {
		if err := json.Unmarshal(raw, &src); err != nil {
			return nil, llm.Errorf(llm.KindInvalidRequest, "image source: %v", err)
		}
	}
	img := &llm.Image{MIME: src.MediaType, URL: src.URL}
	if src.Type == "base64" && src.Data != "" {
		b, err := base64.StdEncoding.DecodeString(src.Data)
		if err != nil {
			return nil, llm.Errorf(llm.KindInvalidRequest, "image data: %v", err)
		}
		img.Data = b
	}
	return img, nil
}

func setMeta(m map[string]string, k, v string) map[string]string {
	if m == nil {
		m = map[string]string{}
	}
	m[k] = v
	return m
}

func bearer(h string) string {
	const p = "Bearer "
	if len(h) > len(p) && strings.EqualFold(h[:len(p)], p) {
		return h[len(p):]
	}
	return ""
}

func field[T any](m map[string]json.RawMessage, key string, dst *T) error {
	raw, ok := m[key]
	if !ok || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return llm.Errorf(llm.KindInvalidRequest, "%s: %v", key, err)
	}
	return nil
}

func quoted(s string) string {
	if s == "" {
		return `""`
	}
	return `"` + s + `"`
}
