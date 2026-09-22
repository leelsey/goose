// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package openaichat serves the OpenAI chat completions shape.
package openaichat

import (
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const Name = "openai"

type Options struct {
	ReasoningField string
	IDPrefix       string
	IDSource       func() string
}

type Codec struct{ opts Options }

func New(o Options) *Codec {
	if o.IDPrefix == "" {
		o.IDPrefix = "chatcmpl-"
	}
	return &Codec{opts: o}
}

func (c *Codec) Name() string { return Name }

func (c *Codec) Routes() []wire.Route {
	return []wire.Route{
		{Method: "POST", Pattern: "/v1/chat/completions"},
		{Method: "POST", Pattern: "/chat/completions"},
	}
}

func (c *Codec) Renders() wire.Renders {
	return wire.Renders{
		Thinking:   c.opts.ReasoningField != "",
		ToolCalls:  true,
		Candidates: 128,
	}
}

func finishReason(s llm.StopReason) (string, bool) {
	switch s {
	case llm.StopEnd, llm.StopSequence, llm.StopRefusal, llm.StopPause:
		return "stop", true
	case llm.StopLength, llm.StopContextOverflow:
		return "length", true
	case llm.StopToolUse:
		return "tool_calls", true
	case llm.StopContentFilter:
		return "content_filter", true
	case llm.StopNone:
		return "", true
	default:
		return "stop", false
	}
}
