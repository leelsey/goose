// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package anthropicmsg serves the Anthropic Messages shape.
package anthropicmsg

import (
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const Name = "anthropic"

type Options struct {
	IDPrefix string
	IDSource func() string
	Now      func() time.Time
}

type Codec struct{ opts Options }

func New(o Options) *Codec {
	if o.IDPrefix == "" {
		o.IDPrefix = "msg_"
	}
	return &Codec{opts: o}
}

func (c *Codec) Name() string { return Name }

func (c *Codec) Routes() []wire.Route {
	return []wire.Route{
		{Method: "POST", Pattern: "/v1/messages"},
		{Method: "POST", Pattern: "/messages"},
	}
}

func (c *Codec) Renders() wire.Renders {
	return wire.Renders{
		Thinking:   true,
		ToolCalls:  true,
		Candidates: 1,
	}
}

func (c *Codec) id(id string) string {
	if id != "" {
		return id
	}
	if c.opts.IDSource != nil {
		return c.opts.IDPrefix + c.opts.IDSource()
	}
	return c.opts.IDPrefix + "goose"
}

func stopReason(s string) llm.StopReason {
	switch s {
	case "end_turn":
		return llm.StopEnd
	case "max_tokens":
		return llm.StopLength
	case "stop_sequence":
		return llm.StopSequence
	case "tool_use":
		return llm.StopToolUse
	case "pause_turn":
		return llm.StopPause
	case "refusal":
		return llm.StopRefusal
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

func stopName(s llm.StopReason) (string, bool) {
	switch s {
	case llm.StopEnd:
		return "end_turn", true
	case llm.StopLength, llm.StopContextOverflow:
		return "max_tokens", true
	case llm.StopSequence:
		return "stop_sequence", true
	case llm.StopToolUse:
		return "tool_use", true
	case llm.StopPause:
		return "pause_turn", true
	case llm.StopRefusal, llm.StopContentFilter:
		return "refusal", true
	case llm.StopNone:
		return "", true
	default:
		return "end_turn", false
	}
}

type wireUsage struct {
	Input              int `json:"input_tokens"`
	Output             int `json:"output_tokens"`
	CacheRead          int `json:"cache_read_input_tokens,omitempty"`
	CacheCreationInput int `json:"cache_creation_input_tokens,omitempty"`
	CacheCreation      *struct {
		Ephemeral5m int `json:"ephemeral_5m_input_tokens,omitempty"`
		Ephemeral1h int `json:"ephemeral_1h_input_tokens,omitempty"`
	} `json:"cache_creation,omitempty"`
	ServiceTier string `json:"service_tier,omitempty"`
}

type Usage = wireUsage

func DecodeUsage(u wireUsage) llm.Usage {
	raw := llm.RawUsage{
		Prompt:             u.Input,
		Completion:         u.Output,
		CacheRead:          u.CacheRead,
		CacheWrite:         u.CacheCreationInput,
		CacheOutsidePrompt: true,
		Tier:               u.ServiceTier,
		Reported:           true,
	}
	if u.CacheCreation != nil {
		raw.CacheWrite5m = u.CacheCreation.Ephemeral5m
		if raw.CacheWrite == 0 {
			raw.CacheWrite = u.CacheCreation.Ephemeral5m + u.CacheCreation.Ephemeral1h
		}
	}
	return raw.Normalise()
}

func encodeUsage(u llm.Usage) wireUsage {
	out := wireUsage{
		Output:             u.Completion,
		CacheRead:          u.CacheRead,
		CacheCreationInput: u.CacheWrite,
		ServiceTier:        u.Tier,
	}
	if n, ok := u.Uncached(); ok {
		out.Input = n
	}
	if u.CacheWrite5m > 0 || u.CacheWrite > 0 {
		out.CacheCreation = &struct {
			Ephemeral5m int `json:"ephemeral_5m_input_tokens,omitempty"`
			Ephemeral1h int `json:"ephemeral_1h_input_tokens,omitempty"`
		}{Ephemeral5m: u.CacheWrite5m, Ephemeral1h: u.CacheWrite - u.CacheWrite5m}
	}
	return out
}
