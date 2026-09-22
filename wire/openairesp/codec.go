// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package openairesp serves the OpenAI Responses shape.
package openairesp

import (
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const Name = "openai-responses"

type Options struct {
	IDPrefix  string
	IDSource  func() string
	Now       func() time.Time
	MaxBuffer int
}

const DefaultMaxBuffer = 32 << 20

type Codec struct{ opts Options }

func New(o Options) *Codec {
	if o.IDPrefix == "" {
		o.IDPrefix = "resp_"
	}
	return &Codec{opts: o}
}

func (c *Codec) Name() string { return Name }

func (c *Codec) Routes() []wire.Route {
	return []wire.Route{
		{Method: "POST", Pattern: "/v1/responses"},
		{Method: "POST", Pattern: "/responses"},
	}
}

func (c *Codec) Renders() wire.Renders {
	return wire.Renders{Thinking: true, ToolCalls: true, Images: true, Candidates: 1}
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
func status(stop llm.StopReason, failed bool) (state, incomplete string) {
	if failed {
		return "failed", ""
	}
	switch stop {
	case llm.StopError:
		return "failed", ""
	case llm.StopLength, llm.StopContextOverflow:
		return "incomplete", "max_output_tokens"
	case llm.StopContentFilter:
		return "incomplete", "content_filter"
	case llm.StopPause:
		return "incomplete", "pause"
	default:
		return "completed", ""
	}
}

type wireUsage struct {
	InputTokens        int `json:"input_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
	OutputTokens        int `json:"output_tokens"`
	OutputTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens,omitempty"`
	} `json:"output_tokens_details,omitempty"`
	TotalTokens int `json:"total_tokens"`
}

func encodeUsage(u llm.Usage) wireUsage {
	out := wireUsage{InputTokens: u.Prompt, OutputTokens: u.Completion, TotalTokens: u.Prompt + u.Completion}
	if u.CacheRead != 0 {
		out.InputTokensDetails = &struct {
			CachedTokens int `json:"cached_tokens,omitempty"`
		}{CachedTokens: u.CacheRead}
	}
	if u.Reasoning != 0 {
		out.OutputTokensDetails = &struct {
			ReasoningTokens int `json:"reasoning_tokens,omitempty"`
		}{ReasoningTokens: u.Reasoning}
	}
	return out
}

var builtIn = map[string]bool{
	"web_search": true, "web_search_preview": true, "file_search": true,
	"code_interpreter": true, "computer_use_preview": true, "computer_use": true,
	"image_generation": true, "retrieval": true, "mcp": true,
}
