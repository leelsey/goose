// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"encoding/json"
	"sort"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

const defaultMaxTokens = 4096

func (p *Provider) buildRequest(a *llm.Attempt, stream bool) ([]byte, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	sentParallel := false
	req := a.Request
	model := a.Model
	if model == "" {
		model = p.opts.Model
	}
	if model == "" {
		model = req.Model
	}

	body := map[string]any{"model": model, "stream": stream}

	maxTokens := defaultMaxTokens
	switch {
	case req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0:
		maxTokens = *req.MaxOutputTokens
	case req.MaxOutputTokens != nil:
		diags = append(diags, llm.Diagnostic{
			Code: llm.CapCeilingSupplied, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityLoss, Provider: p.ID(), Field: "max_tokens",
			Detail: "this backend needs an output ceiling of at least one, so the default was sent instead",
		})
	default:
		diags = append(diags, llm.Diagnostic{
			Code: llm.CapCeilingSupplied, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "max_tokens",
			Detail: "this backend refuses a request with no output ceiling, so the answer stops at the default rather than where the model would have",
		})
	}
	body["max_tokens"] = maxTokens

	if u, key := callerID(req.Metadata); u != "" {
		body["metadata"] = map[string]any{"user_id": u}
		if rest := unsentMeta(req.Metadata, key); len(rest) > 0 {
			diags = append(diags, metaDropped(p.ID(), rest))
		}
	} else if rest := unsentMeta(req.Metadata, ""); len(rest) > 0 {
		diags = append(diags, metaDropped(p.ID(), rest))
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.TopK != nil {
		body["top_k"] = *req.TopK
	}
	if len(req.Stop) > 0 {
		body["stop_sequences"] = req.Stop
	}
	if req.Thinking != nil && req.Thinking.Enabled {
		th := map[string]any{"type": "enabled"}
		if req.Thinking.Budget > 0 {
			th["budget_tokens"] = req.Thinking.Budget
		} else {
			if req.Thinking.Budget < 0 {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
					Severity: llm.SeverityInfo, Provider: p.ID(), Field: "thinking.budget",
					Detail: "this backend takes a number of tokens, not a request to decide, so half the output ceiling was sent",
				})
			}
			th["budget_tokens"] = maxTokens / 2
		}
		body["thinking"] = th
	}

	budget := maxBreakpoints

	system, msgs := req.SystemAndMessages()
	if len(system) > 0 {
		blocks, d := anthropicmsg.EncodeBlocks(system)
		diags = append(diags, d...)
		if req.SystemCache != nil && budget > 0 {
			blocks = markCache(blocks, req.SystemCache)
			budget--
		}
		body["system"] = blocks
	}

	var toolsets []map[string]any
	if len(req.MCPServers) > 0 {
		for _, s := range req.MCPServers {
			if s.NeedsApproval() {
				return nil, diags, llm.Errorf(llm.KindUnsupported,
					"tool server %q asks this backend to stop for approval before a tool call, which it cannot do", s.Name)
			}
			if s.ReadOnly {
				return nil, diags, llm.Errorf(llm.KindUnsupported,
					"tool server %q asks this backend for its read-only tools alone, which it cannot select", s.Name)
			}
		}
		servers, sets, unsent := anthropicmsg.EncodeMCPServers(req.MCPServers)
		if len(servers) > 0 {
			body["mcp_servers"] = servers
			toolsets = sets
		}
		for _, field := range unsent {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropMCPServer, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityLoss, Provider: p.ID(), Field: field,
				Detail: "this backend has no field for it on a tool server, so that part of the connector was not sent",
			})
		}
	}

	tools := a.Tools
	if len(tools) == 0 {
		tools = req.Tools
	}
	if len(tools) == 0 && len(toolsets) > 0 {
		body["tools"] = toolsets
	}
	if len(tools) > 0 {
		out := make([]map[string]any, 0, len(tools))
		for _, t := range tools {
			schema := t.Schema
			if len(schema) == 0 {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			entry := map[string]any{"name": t.Name, "input_schema": schema}
			if t.Description != "" {
				entry["description"] = t.Description
			}
			if t.Cache != nil && budget > 0 {
				entry["cache_control"] = cacheControl(t.Cache)
				budget--
			}
			out = append(out, entry)
		}
		body["tools"] = append(out, toolsets...)
		tc := map[string]any{}
		if req.ToolChoice != nil {
			if named := toolChoice(req.ToolChoice); named != nil {
				tc = named
			}
		}
		if req.ParallelTools != nil {
			if _, named := tc["type"]; !named {
				tc["type"] = "auto"
			}
			tc["disable_parallel_tool_use"] = !*req.ParallelTools
			sentParallel = true
		}
		if len(tc) > 0 {
			body["tool_choice"] = tc
		}
	}

	if llm.StrictTools(req.Tools) {
		diags = append(diags, llm.ToolStrictness(p.ID()))
	}
	turns, d, err := p.turns(msgs, &budget)
	if err != nil {
		return nil, diags, err
	}
	diags = append(diags, d...)
	if len(turns) == 0 {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "no messages survived translation")
	}
	body["messages"] = turns

	if len(req.Extra) > 0 {
		for k := range req.Extra {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: k,
				Detail: "this backend does not accept unknown request fields",
			})
		}
	}

	sent := []string{
		llm.FieldTemperature, llm.FieldTopP, llm.FieldTopK, llm.FieldStop,
		llm.FieldMaxOutputTokens,
	}
	if sentParallel {
		sent = append(sent, llm.FieldParallelTools)
	}
	diags = append(diags, llm.UnsentSampling(req, p.ID(), sent...)...)
	if p.opts.RewriteBody != nil {
		p.opts.RewriteBody(body, stream)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode request: %v", err)
	}
	for _, x := range diags {
		a.Note(x)
	}
	return raw, diags, nil
}

type turn struct {
	Role    string            `json:"role"`
	Content []json.RawMessage `json:"content"`
}

func spentBreakpoint(provider string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.CapCacheBreakpoint, Class: llm.ClassCapability, Stage: llm.StageRequest,
		Severity: llm.SeverityInfo, Provider: provider,
		Detail: "this backend takes four cache breakpoints; the earliest were dropped",
	}
}

func (p *Provider) turns(msgs []llm.Message, budget *int) ([]turn, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	var out []turn

	for _, m := range msgs {
		role := "user"
		if m.Role == llm.RoleAssistant {
			role = "assistant"
		}
		if m.Name != "" {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "name",
				Detail: "this backend has no field to name a participant, so the name was not sent",
			})
		}
		parts := m.Content
		if role == "assistant" {
			kept, dropped := signedOnly(parts)
			if dropped > 0 {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropSignature, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityInfo, Provider: p.ID(), Count: dropped,
					Detail: "thinking without its signature cannot be replayed to this backend",
				})
			}
			parts = kept
			if len(parts) == 0 {
				diags = append(diags, llm.DroppedMessage(m.Role, p.ID(),
					"this turn held only thinking this backend cannot verify, so the conversation arrives a turn shorter"))
				continue
			}
		}
		blocks, d := anthropicmsg.EncodeBlocks(parts)
		diags = append(diags, d...)
		if len(blocks) == 0 {
			diags = append(diags, llm.DroppedMessage(m.Role, p.ID(),
				"this turn reached no block on this endpoint, so the conversation arrives a turn shorter"))
			continue
		}
		if n := len(out); n > 0 && out[n-1].Role == role {
			out[n-1].Content = append(out[n-1].Content, blocks...)
			if m.Cache != nil {
				if *budget > 0 {
					out[n-1].Content = markCache(out[n-1].Content, m.Cache)
					*budget--
				} else {
					diags = append(diags, spentBreakpoint(p.ID()))
				}
			}
			continue
		}
		out = append(out, turn{Role: role, Content: blocks})
		if m.Cache != nil {
			if *budget > 0 {
				out[len(out)-1].Content = markCache(out[len(out)-1].Content, m.Cache)
				*budget--
			} else {
				diags = append(diags, spentBreakpoint(p.ID()))
			}
		}
	}
	return out, diags, nil
}

func signedOnly(parts []llm.Part) ([]llm.Part, int) {
	dropped := 0
	out := make([]llm.Part, 0, len(parts))
	for _, p := range parts {
		if th, ok := p.(*llm.Thinking); ok && th.Signature == "" && len(th.Redacted) == 0 {
			dropped++
			continue
		}
		out = append(out, p)
	}
	return out, dropped
}

func markCache(blocks []json.RawMessage, hint *llm.CacheHint) []json.RawMessage {
	for i := len(blocks) - 1; i >= 0; i-- {
		var obj map[string]json.RawMessage
		if json.Unmarshal(blocks[i], &obj) != nil {
			continue
		}
		var kind string
		if raw, ok := obj["type"]; ok {
			_ = json.Unmarshal(raw, &kind)
		}
		switch kind {
		case "thinking", "redacted_thinking":
			continue
		}
		cc, err := json.Marshal(cacheControl(hint))
		if err != nil {
			return blocks
		}
		obj["cache_control"] = cc
		if merged, err := json.Marshal(obj); err == nil {
			blocks[i] = merged
		}
		return blocks
	}
	return blocks
}

func cacheControl(h *llm.CacheHint) map[string]any {
	cc := map[string]any{"type": "ephemeral"}
	switch {
	case h == nil || h.TTL == 0:
	case h.TTL >= time.Hour:
		cc["ttl"] = "1h"
	default:
		cc["ttl"] = "5m"
	}
	return cc
}

func toolChoice(tc *llm.ToolChoice) map[string]any {
	switch tc.Mode {
	case llm.ToolAuto:
		return map[string]any{"type": "auto"}
	case llm.ToolNone:
		return map[string]any{"type": "none"}
	case llm.ToolRequired:
		return map[string]any{"type": "any"}
	case llm.ToolSpecific:
		return map[string]any{"type": "tool", "name": tc.Name}
	default:
		return nil
	}
}

func callerID(meta map[string]string) (string, string) {
	for _, key := range []string{"user_id", "user"} {
		if v := meta[key]; v != "" {
			return v, key
		}
	}
	return "", ""
}

func metaDropped(provider string, fields []string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
		Severity: llm.SeverityInfo, Provider: provider, Count: len(fields),
		Field:  strings.Join(fields, ","),
		Detail: "this backend names the end user and carries no other metadata, so the rest was not sent",
	}
}

func unsentMeta(meta map[string]string, carried string) []string {
	var rest []string
	for k := range meta {
		if k != carried {
			rest = append(rest, k)
		}
	}
	sort.Strings(rest)
	return rest
}
