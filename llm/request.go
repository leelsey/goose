// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package llm is goose's neutral model of a call, its answer and its cost.
package llm

import (
	"encoding/json"
	"time"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Message struct {
	Role    Role
	Name    string
	Content []Part
	Cache   *CacheHint
}

type CacheHint struct {
	TTL time.Duration
}

type Tool struct {
	Name         string
	Description  string
	Schema       json.RawMessage
	OutputSchema json.RawMessage
	Strict       bool
	Cache        *CacheHint
}

func StrictTools(tools []Tool) bool {
	for _, t := range tools {
		if t.Strict {
			return true
		}
	}
	return false
}

func ToolStrictness(provider string) Diagnostic {
	return Diagnostic{
		Code: DropToolDefinition, Class: ClassCapability, Stage: StageRequest,
		Severity: SeverityLoss, Provider: provider, Field: "strict",
		Detail: "this backend cannot be told to bind the model to a tool's schema, so the schema is a suggestion here",
	}
}

type MCPServer struct {
	Name          string
	URL           string
	Connector     string
	Description   string
	Token         string
	Headers       map[string]string
	AllowedTools  []string
	DeniedTools   []string
	ReadOnly      bool
	Approval      MCPApproval
	ApprovalTools []string
	Enabled       *bool
	DeferLoading  *bool
}

func (s MCPServer) NeedsApproval() bool {
	switch s.Approval {
	case MCPApprovalAlways:
		return true
	case MCPApprovalNever:
		return len(s.ApprovalTools) > 0
	default:
		return false
	}
}

type MCPApproval string

const (
	MCPApprovalDefault MCPApproval = ""
	MCPApprovalNever   MCPApproval = "never"
	MCPApprovalAlways  MCPApproval = "always"
)

type ToolMode string

const (
	ToolAuto     ToolMode = "auto"
	ToolNone     ToolMode = "none"
	ToolRequired ToolMode = "required"
	ToolSpecific ToolMode = "specific"
)

type ToolChoice struct {
	Mode ToolMode
	Name string
}

type ThinkingConfig struct {
	Enabled bool
	Budget  int
}

type Effort string

const (
	EffortNone    Effort = ""
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
)

type ResponseFormat struct {
	Type   string
	Name   string
	Schema json.RawMessage
	Strict bool
}

type Request struct {
	Model       string
	System      []Part
	Messages    []Message
	SystemCache *CacheHint

	Tools         []Tool
	ToolChoice    *ToolChoice
	MCPServers    []MCPServer
	ParallelTools *bool

	MaxOutputTokens  *int
	Temperature      *float64
	TopP             *float64
	TopK             *int
	Seed             *int64
	Stop             []string
	Candidates       int
	FrequencyPenalty *float64
	PresencePenalty  *float64

	Thinking       *ThinkingConfig
	Effort         Effort
	ResponseFormat *ResponseFormat

	Metadata map[string]string

	Extra     map[string]json.RawMessage
	ExtraFrom string
}

func (r *Request) Clone() *Request {
	if r == nil {
		return nil
	}
	c := *r
	c.System = CloneParts(r.System)
	if r.Messages != nil {
		c.Messages = make([]Message, len(r.Messages))
		for i, m := range r.Messages {
			m.Content = CloneParts(m.Content)
			if m.Cache != nil {
				h := *m.Cache
				m.Cache = &h
			}
			c.Messages[i] = m
		}
	}
	if r.Tools != nil {
		c.Tools = make([]Tool, len(r.Tools))
		for i, t := range r.Tools {
			t.Schema = append(json.RawMessage(nil), t.Schema...)
			t.OutputSchema = append(json.RawMessage(nil), t.OutputSchema...)
			if t.Cache != nil {
				h := *t.Cache
				t.Cache = &h
			}
			c.Tools[i] = t
		}
	}
	if r.MCPServers != nil {
		c.MCPServers = make([]MCPServer, len(r.MCPServers))
		for i, m := range r.MCPServers {
			m.AllowedTools = append([]string(nil), m.AllowedTools...)
			m.DeniedTools = append([]string(nil), m.DeniedTools...)
			m.ApprovalTools = append([]string(nil), m.ApprovalTools...)
			m.Enabled = clonePtr(m.Enabled)
			m.DeferLoading = clonePtr(m.DeferLoading)
			if m.Headers != nil {
				h := make(map[string]string, len(m.Headers))
				for k, v := range m.Headers {
					h[k] = v
				}
				m.Headers = h
			}
			c.MCPServers[i] = m
		}
	}
	c.SystemCache = clonePtr(r.SystemCache)
	c.ToolChoice = clonePtr(r.ToolChoice)
	c.MaxOutputTokens = clonePtr(r.MaxOutputTokens)
	c.Temperature = clonePtr(r.Temperature)
	c.TopP = clonePtr(r.TopP)
	c.TopK = clonePtr(r.TopK)
	c.Seed = clonePtr(r.Seed)
	c.FrequencyPenalty = clonePtr(r.FrequencyPenalty)
	c.PresencePenalty = clonePtr(r.PresencePenalty)
	c.Thinking = clonePtr(r.Thinking)
	c.Stop = append([]string(nil), r.Stop...)
	if r.ResponseFormat != nil {
		f := *r.ResponseFormat
		f.Schema = append(json.RawMessage(nil), r.ResponseFormat.Schema...)
		c.ResponseFormat = &f
	}
	if r.Metadata != nil {
		c.Metadata = make(map[string]string, len(r.Metadata))
		for k, v := range r.Metadata {
			c.Metadata[k] = v
		}
	}
	if r.Extra != nil {
		c.Extra = make(map[string]json.RawMessage, len(r.Extra))
		for k, v := range r.Extra {
			c.Extra[k] = append(json.RawMessage(nil), v...)
		}
	}
	return &c
}

func clonePtr[T any](p *T) *T {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}

func Ptr[T any](v T) *T { return &v }

func (r *Request) SystemAndMessages() ([]Part, []Message) {
	if r == nil {
		return nil, nil
	}
	msgs := r.Messages
	lead := 0
	for lead < len(msgs) && (msgs[lead].Role == RoleSystem || msgs[lead].Role == RoleDeveloper) {
		lead++
	}
	if lead == 0 {
		return r.System, msgs
	}
	system := make([]Part, 0, len(r.System)+lead)
	system = append(system, r.System...)
	for _, m := range msgs[:lead] {
		system = append(system, m.Content...)
	}
	return system, msgs[lead:]
}

const (
	FieldTemperature      = "temperature"
	FieldTopP             = "top_p"
	FieldTopK             = "top_k"
	FieldSeed             = "seed"
	FieldStop             = "stop"
	FieldFrequencyPenalty = "frequency_penalty"
	FieldPresencePenalty  = "presence_penalty"
	FieldMaxOutputTokens  = "max_output_tokens"
	FieldParallelTools    = "parallel_tool_calls"
)

func UnsentSampling(req *Request, provider string, sent ...string) Diagnostics {
	if req == nil {
		return nil
	}
	carried := map[string]bool{}
	for _, name := range sent {
		carried[name] = true
	}
	set := []struct {
		name string
		on   bool
	}{
		{FieldTemperature, req.Temperature != nil},
		{FieldTopP, req.TopP != nil},
		{FieldTopK, req.TopK != nil},
		{FieldSeed, req.Seed != nil},
		{FieldStop, len(req.Stop) > 0},
		{FieldFrequencyPenalty, req.FrequencyPenalty != nil},
		{FieldPresencePenalty, req.PresencePenalty != nil},
		{FieldMaxOutputTokens, req.MaxOutputTokens != nil},
		{FieldParallelTools, req.ParallelTools != nil},
	}
	var out Diagnostics
	for _, f := range set {
		if !f.on || carried[f.name] {
			continue
		}
		out = append(out, Diagnostic{
			Code: DropExtraField, Class: ClassCapability, Stage: StageRequest,
			Severity: SeverityInfo, Provider: provider, Field: f.name,
			Detail: "this backend has no field for this setting, so it was not sent",
		})
	}
	return out
}
