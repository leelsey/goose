// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"fmt"
	"sort"
	"strings"

	"github.com/leelsey/goose/llm"
)

func negotiate(a *llm.Attempt, caps llm.Capabilities) error {
	req := a.Request
	var fatal llm.Diagnostics
	fail := func(d llm.Diagnostic) {
		fatal = append(fatal, d)
		a.Note(d)
	}

	if req.Candidates > 1 && req.Candidates > caps.MaxCandidates() {
		fail(llm.Diagnostic{
			Code: llm.CapCandidatesOne, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityFatal, Provider: a.Provider, Model: a.Model,
			Count:  req.Candidates,
			Detail: "this backend returns one completion; answering with one would deliver a fraction of what was asked for and bill for it",
		})
	}

	if !caps.ImageIn && hasImage(req) {
		fail(llm.Diagnostic{
			Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StagePlan,
			Severity: llm.SeverityFatal, Provider: a.Provider, Model: a.Model,
			Detail: "this backend cannot read an image, and answering without one answers a different question",
		})
	}

	if !caps.ToolResults && hasToolResult(req) {
		fail(llm.Diagnostic{
			Code: llm.DropToolResult, Class: llm.ClassContent, Stage: llm.StagePlan,
			Severity: llm.SeverityFatal, Provider: a.Provider, Model: a.Model,
			Detail: "this backend takes no tool results, and a history whose tool calls went unanswered is a different conversation",
		})
	}

	if caps.Tools && len(a.Tools) > 0 && a.Renders.Declared() && !a.Renders.ToolCalls {
		fail(llm.Diagnostic{
			Code: llm.DropToolCall, Class: llm.ClassContent, Stage: llm.StagePlan,
			Severity: llm.SeverityFatal, Provider: a.Provider, Model: a.Model, Format: a.Format,
			Detail: "this format has no field for a tool call, so a tool the model reached for could not be reported back",
		})
	}

	if max := caps.MaxInputBytes; max > 0 {
		if n := approxBytes(req); n > max {
			fail(llm.Diagnostic{
				Code: llm.PolicyBodyTooLarge, Class: llm.ClassPolicy, Stage: llm.StagePlan,
				Severity: llm.SeverityFatal, Provider: a.Provider, Model: a.Model, Count: int(n),
				Detail: "the request is larger than this backend accepts, and sending it would be paid for and refused",
			})
		}
	}

	if len(fatal) > 0 {
		e := llm.AsError(llm.LossPolicy{}.Check(fatal))
		if e == nil {
			e = &llm.Error{Kind: llm.KindUnsupported, Message: fatal[0].Detail}
		}
		e.Kind = llm.KindUnsupported
		e.Provider = a.Provider
		e.Model = a.Model
		e.Diagnostics = fatal
		e.Sent = false
		e.Retryable = true
		return e
	}

	if !a.Credential.Empty() && !credentialTravels(a, caps) {
		a.Note(llm.Diagnostic{
			Code: llm.PolicyKeyWithheld, Class: llm.ClassPolicy, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider,
			Detail: "the caller's own credential was minted for another backend, so this attempt uses the operator's",
		})
		a.Credential = llm.Credential{}
	}

	if len(req.Tools) > 0 && !caps.Tools {
		a.Note(llm.Diagnostic{
			Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Count: len(req.Tools),
			Detail: "this backend takes no tool definitions, so the model answered without the tools it was offered",
		})
		req.Tools = nil
		a.Tools = nil
	}
	if len(req.MCPServers) > 0 && !caps.MCPServers {
		a.Note(llm.Diagnostic{
			Code: llm.DropMCPServer, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Field: "mcp_servers",
			Count:  len(req.MCPServers),
			Detail: "this backend does not connect to a tool server of its own, so the model answered without the tools it was offered",
		})
		req.MCPServers = nil
	}
	if req.ToolChoice != nil && (!caps.ToolChoice || len(req.Tools) == 0) {
		detail := "this backend does not honour a tool choice"
		if len(req.Tools) == 0 {
			detail = "no tool definition reached this backend, so the tool choice had nothing to name"
		}
		a.Note(llm.Diagnostic{
			Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider, Field: "tool_choice",
			Detail: detail,
		})
		req.ToolChoice = nil
	}
	if tc := req.ToolChoice; tc != nil && tc.Mode == llm.ToolSpecific && tc.Name == "" {
		a.Note(llm.Diagnostic{
			Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Field: "tool_choice",
			Detail: "the tool choice named no tool, so the backend was left to choose for itself",
		})
		req.ToolChoice = nil
	}
	if f := req.ResponseFormat; f != nil && f.Type == "" && len(f.Schema) == 0 {
		a.Note(llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider, Field: "response_format",
			Detail: "the response format named no type and carried no schema, so there was nothing to ask for",
		})
		req.ResponseFormat = nil
	}
	if req.ResponseFormat != nil && !caps.ResponseFormat {
		a.Note(llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Field: "response_format",
			Detail: "this backend cannot be asked for a structured answer, so the reply may not parse",
		})
		req.ResponseFormat = nil
	}
	if !caps.SystemPrompt && hasSystem(req) {
		foldSystem(req)
		a.Note(llm.Diagnostic{
			Code: llm.DropSystemPrompt, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider, Field: "system",
			Detail: "this backend has no system channel, so the prompt was folded into the conversation and is billed as fresh input every call",
		})
	}
	if len(req.Extra) > 0 && !caps.Takes(req.ExtraFrom) {
		fields := make([]string, 0, len(req.Extra))
		for k := range req.Extra {
			fields = append(fields, k)
		}
		sort.Strings(fields)
		a.Note(llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Count: len(fields),
			Field:  named(fields),
			Detail: extraDetail(req.ExtraFrom, caps),
		})
		req.Extra = nil
	}
	if req.Thinking != nil && !caps.Thinking {
		a.Note(llm.Diagnostic{
			Code: llm.DropThinking, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider, Field: "thinking",
			Detail: "this backend has no extended thinking to enable",
		})
		req.Thinking = nil
	}
	if req.Effort != "" && !caps.Effort {
		a.Note(llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider, Field: "reasoning_effort",
			Detail: "this backend has no effort setting",
		})
		req.Effort = ""
	}
	if !caps.ThinkingSignatures && stripSignatures(req) {
		a.Note(llm.Diagnostic{
			Code: llm.DropSignature, Class: llm.ClassContent, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider,
			Detail: "this backend cannot verify a thinking signature, so the block was replayed without one",
		})
	}
	if !caps.RedactedThinking && stripRedacted(req) {
		a.Note(llm.Diagnostic{
			Code: llm.DropThinking, Class: llm.ClassContent, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider,
			Detail: "this backend has no redacted thinking block, so the reasoning it stood for was dropped",
		})
	}
	if !caps.ToolErrors && hasFailedToolResult(req) {
		a.Note(llm.Diagnostic{
			Code: llm.DropToolError, Class: llm.ClassContent, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Field: "is_error",
			Detail: "this backend has no field marking a tool result as a failure, so only the text of the failure travelled",
		})
	}
	if !caps.ParallelTools && hasParallelTools(req) {
		a.Note(llm.Diagnostic{
			Code: llm.CapParallelTools, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider,
			Detail: "this backend issues one tool call at a time; a history that ran two together may not replay",
		})
	}
	if a.Session != "" && !caps.Sessions {
		a.Note(llm.Diagnostic{
			Code: llm.CapNoSessions, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider,
			Detail: "this backend has no session to resume, so the handle was not used",
		})
		a.Session = ""
	}
	if caps.CacheBreakpoints == 0 && hasCacheHint(req) {
		a.Note(llm.Diagnostic{
			Code: llm.CapCacheBreakpoint, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider,
			Detail: "this backend cannot express a cache breakpoint, so the whole prompt is billed fresh",
		})
		clearCacheHints(req)
	} else if dropped := trimCacheHints(req, caps.CacheBreakpoints); dropped > 0 {
		a.Note(llm.Diagnostic{
			Code: llm.DropCacheHint, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: a.Provider, Count: dropped,
			Detail: "this backend takes a limited number of cache breakpoints, so the later ones were not sent",
		})
	}
	if caps.Usage.MultiTurn {
		a.Note(llm.Diagnostic{
			Code: llm.AcctMultiTurn, Class: llm.ClassAccounting, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider,
			Detail: "this backend runs several billed turns inside one call, so what it reports is their total rather than one turn's",
		})
	}
	if !caps.Usage.StreamUsage && a.Stream {
		a.Note(llm.Diagnostic{
			Code: llm.CapNoStreamUsage, Class: llm.ClassAccounting, Stage: llm.StagePlan,
			Severity: llm.SeverityInfo, Provider: a.Provider,
			Detail: "this backend reports no usage while streaming, so the call books as unmeasured",
		})
	}
	return nil
}

func hasImage(req *llm.Request) bool {
	for _, m := range req.Messages {
		if llm.HasKind(m.Content, llm.KindImage) {
			return true
		}
		for _, p := range m.Content {
			if tr, ok := p.(*llm.ToolResult); ok && llm.HasKind(tr.Content, llm.KindImage) {
				return true
			}
		}
	}
	return llm.HasKind(req.System, llm.KindImage)
}

func hasFailedToolResult(req *llm.Request) bool {
	for _, m := range req.Messages {
		for _, p := range m.Content {
			if tr, ok := p.(*llm.ToolResult); ok && tr.IsError {
				return true
			}
		}
	}
	return false
}

func hasCacheHint(req *llm.Request) bool {
	if req.SystemCache != nil {
		return true
	}
	for _, m := range req.Messages {
		if m.Cache != nil {
			return true
		}
	}
	for _, t := range req.Tools {
		if t.Cache != nil {
			return true
		}
	}
	return false
}

func credentialTravels(a *llm.Attempt, caps llm.Capabilities) bool {
	if a.Credential.Scheme != "" && caps.KeyScheme != "" {
		return a.Credential.Scheme == caps.KeyScheme
	}
	return a.Index == 0
}

func hasToolResult(req *llm.Request) bool {
	for _, m := range req.Messages {
		if llm.HasKind(m.Content, llm.KindToolResult) {
			return true
		}
	}
	return false
}

func hasSystem(req *llm.Request) bool {
	if len(req.System) > 0 {
		return true
	}
	for _, m := range req.Messages {
		if m.Role == llm.RoleSystem || m.Role == llm.RoleDeveloper {
			return true
		}
	}
	return false
}

// foldSystem turns every system turn into an ordinary one. A backend with no
// system channel has nowhere to put one wherever it stands, and leaving a turn
// behind while reporting it folded fails a caller who refuses the loss only
// after the call has been paid for. What a turn carried — its cache
// breakpoint — travels with it.
func foldSystem(req *llm.Request) {
	folded := make([]llm.Message, 0, len(req.Messages)+1)
	if len(req.System) > 0 {
		folded = append(folded, llm.Message{
			Role: llm.RoleUser, Content: req.System, Cache: req.SystemCache,
		})
	}
	req.System, req.SystemCache = nil, nil
	for _, m := range req.Messages {
		if m.Role == llm.RoleSystem || m.Role == llm.RoleDeveloper {
			m.Role = llm.RoleUser
		}
		folded = append(folded, m)
	}
	req.Messages = folded
}

// The keys are the caller's to name and the list is written to the ledger once
// for every link of a chain, so what it can cost is not the caller's to choose.
const maxNamedFields = 512

func named(fields []string) string {
	var b strings.Builder
	for i, f := range fields {
		sep := 0
		if i > 0 {
			sep = 1
		}
		if b.Len()+sep+len(f) > maxNamedFields {
			if i > 0 {
				b.WriteByte(',')
			}
			fmt.Fprintf(&b, "… and %d more", len(fields)-i)
			break
		}
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(f)
	}
	return b.String()
}

func hasParallelTools(req *llm.Request) bool {
	for _, m := range req.Messages {
		calls := 0
		for _, p := range m.Content {
			if p.Kind() == llm.KindToolUse {
				calls++
			}
		}
		if calls > 1 {
			return true
		}
	}
	return false
}

func stripSignatures(req *llm.Request) bool {
	found := false
	for _, m := range req.Messages {
		for _, p := range m.Content {
			if t, ok := p.(*llm.Thinking); ok && t.Signature != "" {
				t.Signature = ""
				found = true
			}
		}
	}
	return found
}

func stripRedacted(req *llm.Request) bool {
	found := false
	for i := range req.Messages {
		kept := req.Messages[i].Content[:0]
		for _, p := range req.Messages[i].Content {
			t, ok := p.(*llm.Thinking)
			if !ok || len(t.Redacted) == 0 {
				kept = append(kept, p)
				continue
			}
			found = true
			t.Redacted = nil
			if t.Text != "" {
				kept = append(kept, t)
			}
		}
		req.Messages[i].Content = kept
	}
	return found
}

func approxBytes(req *llm.Request) int64 {
	var n int64
	n += partBytes(req.System)
	for _, m := range req.Messages {
		n += int64(len(m.Name)) + partBytes(m.Content)
	}
	for _, t := range req.Tools {
		n += int64(len(t.Name) + len(t.Description) + len(t.Schema) + len(t.OutputSchema))
	}
	// Extras and tool servers are the caller's to size too, and a ceiling that
	// does not weigh them is one a caller can walk past.
	for k, v := range req.Extra {
		n += int64(len(k) + len(v))
	}
	for _, s := range req.MCPServers {
		n += int64(len(s.Name) + len(s.URL) + len(s.Connector) + len(s.Description) + len(s.Token))
		for k, v := range s.Headers {
			n += int64(len(k) + len(v))
		}
		for _, list := range [][]string{s.AllowedTools, s.DeniedTools, s.ApprovalTools} {
			for _, name := range list {
				n += int64(len(name))
			}
		}
	}
	for k, v := range req.Metadata {
		n += int64(len(k) + len(v))
	}
	return n
}

func partBytes(parts []llm.Part) int64 {
	var n int64
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text:
			n += int64(len(v.Text))
		case *llm.Thinking:
			n += int64(len(v.Text) + len(v.Signature) + len(v.Redacted))
		case *llm.ToolUse:
			n += int64(len(v.ID) + len(v.Name) + len(v.Args))
		case *llm.ToolResult:
			n += int64(len(v.ID)+len(v.Name)) + partBytes(v.Content)
		case *llm.Image:
			n += int64(len(v.MIME)) + int64(len(v.Data)+len(v.URL))*4/3
		case *llm.Refusal:
			n += int64(len(v.Text))
		case *llm.Raw:
			n += int64(len(v.JSON))
		}
	}
	return n
}

func clearCacheHints(req *llm.Request) {
	req.SystemCache = nil
	for i := range req.Messages {
		req.Messages[i].Cache = nil
	}
	for i := range req.Tools {
		req.Tools[i].Cache = nil
	}
}

func trimCacheHints(req *llm.Request, budget int) int {
	if budget <= 0 {
		return 0
	}
	left, dropped := budget, 0
	take := func(h **llm.CacheHint) {
		if *h == nil {
			return
		}
		if left > 0 {
			left--
			return
		}
		*h = nil
		dropped++
	}
	for i := range req.Tools {
		take(&req.Tools[i].Cache)
	}
	take(&req.SystemCache)
	for i := range req.Messages {
		take(&req.Messages[i].Cache)
	}
	return dropped
}

func extraDetail(from string, caps llm.Capabilities) string {
	if len(caps.Extensions) == 0 {
		return "this backend refuses request fields it does not know, so they were not sent"
	}
	if from == "" {
		return "these fields name no wire, so there is no backend they are known to be good for"
	}
	return "these fields belong to the " + from + " wire, which is not the one this backend speaks"
}
