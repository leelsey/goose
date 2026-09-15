// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import "slices"

type Severity int

const (
	SeverityInfo Severity = iota
	SeverityLoss
	SeverityFatal
)

func (s Severity) String() string {
	switch s {
	case SeverityInfo:
		return "info"
	case SeverityLoss:
		return "loss"
	case SeverityFatal:
		return "fatal"
	default:
		return "unknown"
	}
}

type Class string

const (
	ClassContent    Class = "content"
	ClassCapability Class = "capability"
	ClassAccounting Class = "accounting"
	ClassTransport  Class = "transport"
	ClassPolicy     Class = "policy"
)

type Stage string

const (
	StageDecode   Stage = "decode"
	StagePlan     Stage = "plan"
	StageRequest  Stage = "request"
	StageResponse Stage = "response"
	StageEncode   Stage = "encode"
)

type Diagnostic struct {
	Code     string   `json:"code"`
	Class    Class    `json:"class"`
	Stage    Stage    `json:"stage"`
	Severity Severity `json:"severity"`

	Provider string `json:"provider,omitempty"`
	Model    string `json:"model,omitempty"`
	Format   string `json:"format,omitempty"`
	Field    string `json:"field,omitempty"`
	Detail   string `json:"detail,omitempty"`
	Count    int    `json:"count,omitempty"`
}

const (
	DropThinking       = "drop.thinking"
	DropSignature      = "drop.signature"
	DropToolCall       = "drop.tool_call"
	DropToolResult     = "drop.tool_result"
	DropToolError      = "drop.tool_error"
	DropImage          = "drop.image"
	DropCandidate      = "drop.candidate"
	DropUnknownBlock   = "drop.unknown_block"
	DropSystemPrompt   = "drop.system_prompt"
	DropExtraField     = "drop.extra_field"
	DropCacheHint      = "drop.cache_hint"
	DropToolDefinition = "drop.tool_definition"
	DropMCPServer      = "drop.mcp_server"
	DropMessage        = "drop.message"

	CapNoStreamUsage    = "cap.no_stream_usage"
	CapNoSessions       = "cap.no_sessions"
	CapSessionElsewhere = "cap.session_elsewhere"
	CapCeilingRespelt   = "cap.ceiling_respelt"
	CapCeilingSupplied  = "cap.ceiling_supplied"
	CapCacheBreakpoint  = "cap.cache_breakpoint_budget"
	CapCandidatesOne    = "cap.candidates_one"
	CapParallelTools    = "cap.parallel_tools"
	CapNoModelList      = "cap.no_model_list"

	AcctRowDropped      = "acct.row_dropped"
	AcctTurnRestate     = "acct.turn_restated"
	AcctMultiTurn       = "acct.multi_turn"
	AcctFigureUncarried = "acct.figure_uncarried"

	WireFrameMalformed = "wire.frame_malformed"
	WireErrorInBody    = "wire.error_in_body"
	WireNoFinish       = "wire.no_finish_reason"
	WireBufferFull     = "wire.buffer_full"

	PolicyBodyTooLarge      = "policy.body_too_large"
	PolicySaturated         = "policy.saturated"
	PolicyBreakerOpen       = "policy.breaker_open"
	PolicyKeyWithheld       = "policy.client_key_withheld"
	PolicyPromptOnArgv      = "policy.prompt_on_argv"
	PolicyAgentUnrestricted = "policy.agent_unrestricted"
)

func Codes() []string {
	return []string{
		DropThinking, DropSignature, DropToolCall, DropToolResult, DropImage,
		DropCandidate, DropUnknownBlock, DropSystemPrompt, DropExtraField,
		DropCacheHint, DropToolDefinition, DropMCPServer, DropMessage, DropToolError,
		CapNoStreamUsage, CapNoSessions, CapSessionElsewhere, CapCeilingRespelt,
		CapCacheBreakpoint, CapCandidatesOne, CapParallelTools, CapNoModelList,
		CapCeilingSupplied,
		AcctRowDropped, AcctTurnRestate, AcctMultiTurn, AcctFigureUncarried,
		WireFrameMalformed, WireErrorInBody, WireNoFinish, WireBufferFull,
		PolicyBodyTooLarge, PolicySaturated, PolicyBreakerOpen, PolicyKeyWithheld,
		PolicyPromptOnArgv, PolicyAgentUnrestricted,
	}
}

func KnownCode(code string) bool {
	return slices.Contains(Codes(), code)
}

type Recorder interface {
	Note(Diagnostic)
}

type Diagnostics []Diagnostic

func (d *Diagnostics) Note(x Diagnostic) { *d = append(*d, x) }

func (d Diagnostics) Worst() Severity {
	worst := SeverityInfo
	for _, x := range d {
		if x.Severity > worst {
			worst = x.Severity
		}
	}
	return worst
}

func (d Diagnostics) Codes() []string {
	out := make([]string, 0, len(d))
	for _, x := range d {
		if !slices.Contains(out, x.Code) {
			out = append(out, x.Code)
		}
	}
	return out
}

func DroppedMessage(role Role, provider, detail string) Diagnostic {
	return Diagnostic{
		Code: DropMessage, Class: ClassContent, Stage: StageRequest,
		Severity: SeverityLoss, Provider: provider, Field: string(role),
		Detail: detail,
	}
}

func (d Diagnostics) Has(code string) bool {
	for _, x := range d {
		if x.Code == code {
			return true
		}
	}
	return false
}

func (d Diagnostics) Find(code string) Diagnostics {
	var out Diagnostics
	for _, x := range d {
		if x.Code == code {
			out = append(out, x)
		}
	}
	return out
}

type NopRecorder struct{}

func (NopRecorder) Note(Diagnostic) {}

type LossPolicy struct {
	Strict bool
	Allow  []string
	Deny   []string
}

func (p LossPolicy) Apply(d Diagnostic) Diagnostic {
	switch {
	case slices.Contains(p.Deny, d.Code):
		d.Severity = SeverityFatal
	case slices.Contains(p.Allow, d.Code):
		d.Severity = SeverityInfo
	case p.Strict && d.Class == ClassContent && d.Severity < SeverityFatal:
		d.Severity = SeverityFatal
	}
	return d
}

func (p LossPolicy) Check(ds []Diagnostic) error {
	var fatal []Diagnostic
	for _, d := range ds {
		if p.Apply(d).Severity == SeverityFatal {
			fatal = append(fatal, d)
		}
	}
	if len(fatal) == 0 {
		return nil
	}
	e := &Error{
		Kind:        KindLoss,
		Code:        fatal[0].Code,
		Message:     "translation would lose content: " + fatal[0].Code,
		Diagnostics: fatal,
	}
	if fatal[0].Detail != "" {
		e.Message += " (" + fatal[0].Detail + ")"
	}
	return e
}
