// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openaichat

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

var known = map[string]bool{
	"model": true, "messages": true, "stream": true, "stream_options": true,
	"max_tokens": true, "max_completion_tokens": true, "temperature": true,
	"top_p": true, "n": true, "stop": true, "seed": true, "tools": true,
	"tool_choice": true, "functions": true, "function_call": true,
	"response_format": true, "reasoning_effort": true, "frequency_penalty": true,
	"presence_penalty": true, "user": true, "metadata": true,
	"parallel_tool_calls": true,
}

var validRole = map[string]bool{
	"system": true, "developer": true, "user": true,
	"assistant": true, "tool": true, "function": true,
}

type wireTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
		Strict      bool            `json:"strict"`
	} `json:"function"`
}

type wireToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Index    *int   `json:"index"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type wireMessage struct {
	Role         string          `json:"role"`
	Name         string          `json:"name"`
	Content      json.RawMessage `json:"content"`
	Refusal      *string         `json:"refusal"`
	ToolCalls    []wireToolCall  `json:"tool_calls"`
	ToolCallID   string          `json:"tool_call_id"`
	FunctionCall *struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function_call"`
}

type legacyCalls struct {
	n       int
	pending map[string]string
}

func (l *legacyCalls) mint(name string) string {
	l.n++
	id := "call_legacy_" + strconv.Itoa(l.n)
	if l.pending == nil {
		l.pending = map[string]string{}
	}
	l.pending[name] = id
	return id
}

func (l *legacyCalls) take(name string) string {
	id, ok := l.pending[name]
	if !ok {
		return ""
	}
	delete(l.pending, name)
	return id
}

func (c *Codec) Decode(in *wire.Inbound) (*llm.Request, wire.Options, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	opts := wire.Options{Format: Name, Header: in.Header}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(in.Body, &top); err != nil {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "malformed JSON body: %v", err)
	}

	req := &llm.Request{}
	if err := jsonField(top, "model", &req.Model); err != nil {
		return nil, opts, diags, err
	}
	opts.RequestModel = req.Model

	var msgs []wireMessage
	if err := jsonField(top, "messages", &msgs); err != nil {
		return nil, opts, diags, err
	}
	if len(msgs) == 0 {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "messages is required")
	}
	var legacy legacyCalls
	for i, m := range msgs {
		msg, d, err := decodeMessage(m, &legacy)
		if err != nil {
			return nil, opts, diags, fmt.Errorf("messages[%d]: %w", i, err)
		}
		diags = append(diags, d...)
		req.Messages = append(req.Messages, msg)
	}

	if err := jsonField(top, "stream", &opts.Stream); err != nil {
		return nil, opts, diags, err
	}
	if raw, ok := top["stream_options"]; ok {
		var so struct {
			IncludeUsage bool `json:"include_usage"`
		}
		if err := json.Unmarshal(raw, &so); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "stream_options: %v", err)
		}
		opts.StreamUsage = so.IncludeUsage
	}

	var ceiling *int
	if err := jsonField(top, "max_completion_tokens", &ceiling); err != nil {
		return nil, opts, diags, err
	}
	var older *int
	if err := jsonField(top, "max_tokens", &older); err != nil {
		return nil, opts, diags, err
	}
	if older != nil {
		ceiling = older
	}
	req.MaxOutputTokens = ceiling

	if err := jsonField(top, "temperature", &req.Temperature); err != nil {
		return nil, opts, diags, err
	}
	if err := jsonField(top, "top_p", &req.TopP); err != nil {
		return nil, opts, diags, err
	}
	if err := jsonField(top, "seed", &req.Seed); err != nil {
		return nil, opts, diags, err
	}
	if err := jsonField(top, "frequency_penalty", &req.FrequencyPenalty); err != nil {
		return nil, opts, diags, err
	}
	if err := jsonField(top, "parallel_tool_calls", &req.ParallelTools); err != nil {
		return nil, opts, diags, err
	}
	if err := jsonField(top, "presence_penalty", &req.PresencePenalty); err != nil {
		return nil, opts, diags, err
	}

	if raw, ok := top["n"]; ok && string(raw) != "null" {
		var n float64
		if err := json.Unmarshal(raw, &n); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "n: %v", err)
		}
		req.Candidates = int(n)
		if float64(req.Candidates) != n || req.Candidates < 1 {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "n must be a positive whole number")
		}
	}

	if raw, ok := top["stop"]; ok && string(raw) != "null" {
		stop, err := decodeStop(raw)
		if err != nil {
			return nil, opts, diags, err
		}
		req.Stop = stop
	}

	if raw, ok := top["tools"]; ok && string(raw) != "null" {
		var tools []wireTool
		if err := json.Unmarshal(raw, &tools); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
		}
		for _, t := range tools {
			if t.Type != "" && t.Type != "function" {
				return nil, opts, diags, llm.Errorf(llm.KindUnsupported, "tool type %q is not portable across providers", t.Type)
			}
			req.Tools = append(req.Tools, llm.Tool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Schema:      t.Function.Parameters,
				Strict:      t.Function.Strict,
			})
		}
	}
	if raw, ok := top["functions"]; ok && len(req.Tools) > 0 && string(raw) != "null" {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StageDecode,
			Severity: llm.SeverityInfo, Format: Name, Field: "functions",
			Detail: "the request carried both spellings of the tool list, so the deprecated one was not read",
		})
	}
	if raw, ok := top["functions"]; ok && len(req.Tools) == 0 && string(raw) != "null" {
		var fns []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		}
		if err := json.Unmarshal(raw, &fns); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "functions: %v", err)
		}
		for _, f := range fns {
			req.Tools = append(req.Tools, llm.Tool{Name: f.Name, Description: f.Description, Schema: f.Parameters})
		}
	}
	if raw, ok := top["tool_choice"]; ok && string(raw) != "null" {
		tc, err := decodeToolChoice(raw)
		if err != nil {
			return nil, opts, diags, err
		}
		req.ToolChoice = tc
	}
	if raw, ok := top["function_call"]; ok && req.ToolChoice != nil && string(raw) != "null" {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StageDecode,
			Severity: llm.SeverityInfo, Format: Name, Field: "function_call",
			Detail: "the request carried both spellings of the tool choice, so the deprecated one was not read",
		})
	}
	if raw, ok := top["function_call"]; ok && req.ToolChoice == nil && string(raw) != "null" {
		tc, err := decodeFunctionCall(raw)
		if err != nil {
			return nil, opts, diags, err
		}
		req.ToolChoice = tc
	}

	if raw, ok := top["response_format"]; ok && string(raw) != "null" {
		var rf struct {
			Type       string `json:"type"`
			JSONSchema struct {
				Name   string          `json:"name"`
				Strict bool            `json:"strict"`
				Schema json.RawMessage `json:"schema"`
			} `json:"json_schema"`
		}
		if err := json.Unmarshal(raw, &rf); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "response_format: %v", err)
		}
		req.ResponseFormat = &llm.ResponseFormat{
			Type:   rf.Type,
			Name:   rf.JSONSchema.Name,
			Schema: rf.JSONSchema.Schema,
			Strict: rf.JSONSchema.Strict,
		}
	}

	var effort string
	if err := jsonField(top, "reasoning_effort", &effort); err != nil {
		return nil, opts, diags, err
	}
	req.Effort = llm.Effort(effort)

	var user string
	if err := jsonField(top, "user", &user); err != nil {
		return nil, opts, diags, err
	}
	if user != "" {
		req.Metadata = map[string]string{"user": user}
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
			if _, taken := req.Metadata[k]; !taken {
				req.Metadata[k] = v
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

	opts.Key = bearer(in.Get("Authorization"))
	return req, opts, diags, nil
}

func decodeMessage(m wireMessage, legacy *legacyCalls) (llm.Message, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	out := llm.Message{Role: llm.Role(m.Role), Name: m.Name}
	if !validRole[m.Role] {
		return out, diags, llm.Errorf(llm.KindInvalidRequest, "unknown role %q", m.Role)
	}
	if m.Role == "function" {
		out.Role = llm.RoleTool
	}

	parts, d, err := decodeContent(m.Content)
	if err != nil {
		return out, diags, err
	}
	diags = append(diags, d...)

	if out.Role == llm.RoleTool {
		id := m.ToolCallID
		if id == "" && m.Name != "" {
			id = legacy.take(m.Name)
		}
		out.Content = []llm.Part{&llm.ToolResult{ID: id, Name: m.Name, Content: parts}}
		return out, diags, nil
	}

	out.Content = parts
	if m.Refusal != nil && *m.Refusal != "" {
		out.Content = append(out.Content, &llm.Refusal{Text: *m.Refusal})
	}
	for i, tc := range m.ToolCalls {
		idx := i
		if tc.Index != nil {
			idx = *tc.Index
		}
		out.Content = append(out.Content, &llm.ToolUse{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Args:  json.RawMessage(tc.Function.Arguments),
			Index: &idx,
		})
	}
	if fc := m.FunctionCall; fc != nil && fc.Name != "" && len(m.ToolCalls) == 0 {
		idx := 0
		out.Content = append(out.Content, &llm.ToolUse{
			ID:    legacy.mint(fc.Name),
			Name:  fc.Name,
			Args:  json.RawMessage(fc.Arguments),
			Index: &idx,
		})
	}
	return out, diags, nil
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
		_ = jsonField(item, "type", &kind)
		switch kind {
		case "text", "input_text", "output_text":
			var s string
			_ = jsonField(item, "text", &s)
			if s != "" {
				parts = append(parts, &llm.Text{Text: s})
			}
		case "refusal":
			var s string
			_ = jsonField(item, "refusal", &s)
			if s != "" {
				parts = append(parts, &llm.Refusal{Text: s})
			}
		case "image_url":
			var iu struct {
				URL    string `json:"url"`
				Detail string `json:"detail"`
			}
			if raw, ok := item["image_url"]; ok {
				if err := json.Unmarshal(raw, &iu); err != nil {
					return nil, diags, llm.Errorf(llm.KindInvalidRequest, "image_url: %v", err)
				}
			}
			img := &llm.Image{URL: iu.URL, Detail: iu.Detail}
			if mime, data, ok := dataURL(iu.URL); ok {
				img.MIME, img.Data, img.URL = mime, data, ""
			}
			parts = append(parts, img)
		default:
			b, _ := json.Marshal(item)
			parts = append(parts, &llm.Raw{Vendor: Name, Type: kind, JSON: b})
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityInfo, Format: Name, Field: kind,
				Detail: "content block carried verbatim; a provider that cannot take it will say so",
			})
		}
	}
	return parts, diags, nil
}

func decodeStop(raw json.RawMessage) ([]string, error) {
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		if one == "" {
			return nil, nil
		}
		return []string{one}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "stop: %v", err)
	}
	return many, nil
}

func decodeToolChoice(raw json.RawMessage) (*llm.ToolChoice, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto":
			return &llm.ToolChoice{Mode: llm.ToolAuto}, nil
		case "none":
			return &llm.ToolChoice{Mode: llm.ToolNone}, nil
		case "required", "any":
			return &llm.ToolChoice{Mode: llm.ToolRequired}, nil
		default:
			return nil, llm.Errorf(llm.KindInvalidRequest, "tool_choice: unknown value %q", s)
		}
	}
	var obj struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "tool_choice: %v", err)
	}
	return &llm.ToolChoice{Mode: llm.ToolSpecific, Name: obj.Function.Name}, nil
}

func decodeFunctionCall(raw json.RawMessage) (*llm.ToolChoice, error) {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		switch s {
		case "auto":
			return &llm.ToolChoice{Mode: llm.ToolAuto}, nil
		case "none":
			return &llm.ToolChoice{Mode: llm.ToolNone}, nil
		default:
			return nil, llm.Errorf(llm.KindInvalidRequest, "function_call: unknown value %q", s)
		}
	}
	var obj struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "function_call: %v", err)
	}
	if obj.Name == "" {
		return nil, llm.Errorf(llm.KindInvalidRequest, "function_call names no function")
	}
	return &llm.ToolChoice{Mode: llm.ToolSpecific, Name: obj.Name}, nil
}

func dataURL(s string) (mime string, data []byte, ok bool) {
	if !strings.HasPrefix(s, "data:") {
		return "", nil, false
	}
	rest := s[len("data:"):]
	i := strings.Index(rest, ",")
	if i < 0 {
		return "", nil, false
	}
	meta, payload := rest[:i], rest[i+1:]
	if !strings.HasSuffix(meta, ";base64") {
		return "", nil, false
	}
	b, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", nil, false
	}
	return strings.TrimSuffix(meta, ";base64"), b, true
}

func bearer(h string) string {
	const p = "Bearer "
	if len(h) > len(p) && strings.EqualFold(h[:len(p)], p) {
		return h[len(p):]
	}
	return ""
}

func jsonField[T any](m map[string]json.RawMessage, key string, dst *T) error {
	raw, ok := m[key]
	if !ok || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return llm.Errorf(llm.KindInvalidRequest, "%s: %v", key, err)
	}
	return nil
}
