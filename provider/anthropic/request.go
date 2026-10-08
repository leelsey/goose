// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"encoding/json"
	"math"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

const (
	defaultMaxTokens  = 4096
	minThinkingBudget = 1024
	minThinkingTopP   = 0.95
	// thinkingCeiling is the least any model that thinks will write in one
	// answer, so a ceiling raised to fit a budget is never one it refuses.
	thinkingCeiling = 32000
)

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

	maxTokens, capped := defaultMaxTokens, false
	switch {
	case req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0:
		maxTokens, capped = *req.MaxOutputTokens, true
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
			Detail: "this backend refuses a request with no output ceiling, so goose supplied one and the answer stops there rather than where the model would have",
		})
	}

	if u, key := callerID(req.Metadata); u != "" {
		body["metadata"] = map[string]any{"user_id": u}
		if rest := unsentMeta(req.Metadata, key); len(rest) > 0 {
			diags = append(diags, metaDropped(p.ID(), rest))
		}
	} else if rest := unsentMeta(req.Metadata, ""); len(rest) > 0 {
		diags = append(diags, metaDropped(p.ID(), rest))
	}
	tools := a.Tools
	if len(tools) == 0 {
		tools = req.Tools
	}
	system, msgs := req.SystemAndMessages()
	if len(req.Stop) > 0 {
		body["stop_sequences"] = req.Stop
	}

	budget := maxBreakpoints

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
			detail := "this backend has no field for it on a tool server, so that part of the connector was not sent"
			if strings.HasSuffix(field, ": connector_id") {
				detail = "this backend reaches a tool server by URL alone, so this connector was not sent at all"
			}
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropMCPServer, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityLoss, Provider: p.ID(), Field: field,
				Detail: detail,
			})
		}
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
	}
	// A tool server's tools are tools like any other, and a choice over them
	// alone governs them the same.
	if len(tools) > 0 || len(toolsets) > 0 {
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

	// Decided on the turns as encoded, because whether the turn thinking would
	// resume into leads with the block this backend requires is known only
	// once tool results have been folded into user turns and unsigned thinking
	// has been dropped.
	interleaved := slices.ContainsFunc(p.betas(a), func(b string) bool { return strings.HasPrefix(b, interleavedBeta) })
	// A tool server reaches this backend as a toolset the same as a function
	// tool, so a choice forced over one alone must turn thinking off too.
	hasTools := len(tools) > 0 || slices.ContainsFunc(req.MCPServers, func(s llm.MCPServer) bool { return s.URL != "" })
	thinking, maxTokens, d := p.thinking(req.Thinking, maxTokens, capped,
		hasTools && forced(req.ToolChoice), unsafeToResume(turns), interleaved)
	diags = append(diags, d...)
	body["max_tokens"] = maxTokens
	if thinking != nil {
		body["thinking"] = thinking
	}
	temperature, topP, topK := req.Temperature, req.TopP, req.TopK
	if thinking != nil {
		temperature, topP, topK, d = p.besideThinking(temperature, topP, topK)
		diags = append(diags, d...)
	}
	if temperature != nil {
		body["temperature"] = *temperature
	}
	if topP != nil {
		body["top_p"] = *topP
	}
	if topK != nil {
		body["top_k"] = *topK
	}

	// This wire's own fields travel with the caller's beta flags, which open
	// most of them. Where those flags are not sent, a field sent without the
	// flag it needs makes the backend refuse the whole request.
	own := p.Capabilities().Takes(req.ExtraFrom)
	detail := "this field belongs to another format, so this backend was not sent it"
	if own {
		detail = "the caller's beta flags are not sent to this backend, and a field sent without the flag it needs makes it refuse the whole request, so this one was not sent"
	}
	for k, v := range req.Extra {
		if own && !p.opts.NoClientBetas {
			if _, taken := body[k]; !taken {
				body[k] = v
			}
			continue
		}
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageRequest,
			Severity: llm.SeverityLoss, Provider: p.ID(), Field: k,
			Detail: detail,
		})
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
	if err := a.Check(); err != nil {
		return nil, diags, err
	}
	return raw, diags, nil
}

// thinking is the thinking block this backend takes for what was asked, and
// the output ceiling to send beside it. It thinks for at least 1024 tokens and
// for fewer than the output ceiling, unless the interleaved flag lets a budget
// run past it, and not at all before a tool call it is made to make, nor when
// the turn it would resume into does not lead with the thinking this backend
// requires of it; a request outside that is refused whole. capped says the
// ceiling is the caller's own.
func (p *Provider) thinking(th *llm.ThinkingConfig, maxTokens int, capped, forced, unsafeResume, interleaved bool) (map[string]any, int, llm.Diagnostics) {
	if th == nil || !th.Enabled {
		return nil, maxTokens, nil
	}
	off := func(detail string) (map[string]any, int, llm.Diagnostics) {
		return nil, maxTokens, llm.Diagnostics{{
			Code: llm.DropThinking, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityLoss, Provider: p.ID(), Field: "thinking",
			Detail: detail,
		}}
	}
	if forced {
		return off("this backend cannot think before a tool call it is made to make, so thinking was not enabled")
	}
	if unsafeResume {
		return off("the turn whose tool call this request answers does not lead with thinking this backend can verify, which it requires of that turn once thinking is on, so thinking was not enabled")
	}
	// The models a caller names the adaptive mode for refuse a budget of any
	// size, so the mode is sent as it was named.
	if th.Adaptive {
		return map[string]any{"type": "adaptive"}, maxTokens, nil
	}
	most := maxTokens - 1
	// A ceiling goose supplied is its own to raise, so the caller's budget is
	// kept as far as any model that thinks can write, and the answer after it
	// has the default to itself.
	if !capped && th.Budget > 0 {
		most = min(max(th.Budget, minThinkingBudget), thinkingCeiling-defaultMaxTokens)
		maxTokens = most + defaultMaxTokens
	}
	if interleaved {
		most = math.MaxInt
	}
	if most < minThinkingBudget {
		return off("this backend thinks for at least 1024 tokens and for fewer than the output ceiling, which this ceiling leaves no room between, so thinking was not enabled")
	}
	var diags llm.Diagnostics
	budget := th.Budget
	switch {
	case budget <= 0:
		budget = max(maxTokens/2, minThinkingBudget)
		if th.Budget < 0 {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "thinking.budget",
				Detail: "this backend takes a number of tokens, not a request to decide, so one was chosen from the output ceiling",
			})
		}
	case budget < minThinkingBudget || budget > most:
		budget = min(max(budget, minThinkingBudget), most)
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "thinking.budget",
			Detail: "this backend thinks for at least 1024 tokens and for fewer than the output ceiling, so the budget was moved to the nearest it takes",
		})
	}
	return map[string]any{"type": "enabled", "budget_tokens": budget}, maxTokens, diags
}

// besideThinking is the sampling this backend takes while it thinks: no
// temperature but 1, no top_k and no top_p under 0.95. Anything else beside
// thinking is refused whole.
func (p *Provider) besideThinking(temperature, topP *float64, topK *int) (*float64, *float64, *int, llm.Diagnostics) {
	var diags llm.Diagnostics
	unsent := func(field, detail string) {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: field,
			Detail: detail,
		})
	}
	if temperature != nil && *temperature != 1 {
		temperature = nil
		unsent(llm.FieldTemperature, "this backend samples at its own temperature while it thinks, so the one asked for was not sent")
	}
	if topK != nil {
		topK = nil
		unsent(llm.FieldTopK, "this backend takes no top_k while it thinks, so it was not sent")
	}
	if topP != nil && *topP < minThinkingTopP {
		topP = llm.Ptr(minThinkingTopP)
		unsent(llm.FieldTopP, "this backend takes a top_p of at least 0.95 while it thinks, so 0.95 was sent")
	}
	return temperature, topP, topK, diags
}

// forced says a tool choice leaves the model no answer but a tool call.
func forced(tc *llm.ToolChoice) bool {
	return tc != nil && (tc.Mode == llm.ToolRequired || tc.Mode == llm.ToolSpecific)
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

// unsafeToResume reports whether turns end with a tool result answering an
// assistant turn that does not lead with thinking. Claude requires the
// assistant turn before a trailing tool result to start with thinking or
// redacted_thinking whenever thinking is enabled, however that turn came to
// lose it: dropped here unsigned, stripped before it arrived, or never there.
func unsafeToResume(turns []turn) bool {
	n := len(turns)
	if n < 2 || turns[n-1].Role != "user" || turns[n-2].Role != "assistant" {
		return false
	}
	if !slices.ContainsFunc(turns[n-1].Content, func(b json.RawMessage) bool { return blockType(b) == "tool_result" }) {
		return false
	}
	lead := turns[n-2].Content
	return len(lead) == 0 || (blockType(lead[0]) != "thinking" && blockType(lead[0]) != "redacted_thinking")
}

func blockType(b json.RawMessage) string {
	var v struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(b, &v)
	return v.Type
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
