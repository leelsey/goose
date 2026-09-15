// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package geminiapi serves the Google generateContent shape.
package geminiapi

import (
	"strings"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const Name = "google"

type Options struct {
	Now func() time.Time
}

type Codec struct{ opts Options }

func New(o Options) *Codec { return &Codec{opts: o} }

func (c *Codec) Name() string { return Name }

func (c *Codec) Routes() []wire.Route {
	return []wire.Route{
		{Method: "POST", Pattern: "/v1beta/models/"},
		{Method: "POST", Pattern: "/v1/models/"},
		{Method: "POST", Pattern: "/models/"},
	}
}

func (c *Codec) Renders() wire.Renders {
	return wire.Renders{Thinking: true,
		ToolCalls: true, Images: true, Candidates: 8,
	}
}

func parsePath(path string) (model, action string, err error) {
	i := strings.LastIndex(path, "/models/")
	if i < 0 {
		return "", "", llm.Errorf(llm.KindInvalidRequest, "path %q names no model", path)
	}
	rest := path[i+len("/models/"):]
	j := strings.LastIndex(rest, ":")
	if j < 0 {
		return rest, "generateContent", nil
	}
	return rest[:j], rest[j+1:], nil
}

func finishName(s llm.StopReason) (string, bool) {
	switch s {
	case llm.StopEnd, llm.StopSequence, llm.StopToolUse, llm.StopPause:
		return "STOP", true
	case llm.StopLength, llm.StopContextOverflow:
		return "MAX_TOKENS", true
	case llm.StopContentFilter, llm.StopRefusal:
		return "SAFETY", true
	case llm.StopNone:
		return "", true
	default:
		return "OTHER", false
	}
}

func finishReason(s string) llm.StopReason {
	switch s {
	case "STOP":
		return llm.StopEnd
	case "MAX_TOKENS":
		return llm.StopLength
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY", "LANGUAGE":
		return llm.StopContentFilter
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

type usageMetadata struct {
	PromptTokenCount        int `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount    int `json:"candidatesTokenCount,omitempty"`
	ThoughtsTokenCount      int `json:"thoughtsTokenCount,omitempty"`
	CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	ToolUsePromptTokenCount int `json:"toolUsePromptTokenCount,omitempty"`
	TotalTokenCount         int `json:"totalTokenCount,omitempty"`
}

func encodeUsage(u llm.Usage) *usageMetadata {
	if u.Empty() {
		return nil
	}
	return &usageMetadata{
		PromptTokenCount:        u.Prompt,
		CandidatesTokenCount:    u.Completion - u.Reasoning,
		ThoughtsTokenCount:      u.Reasoning,
		CachedContentTokenCount: u.CacheRead,
		TotalTokenCount:         u.Prompt + u.Completion,
	}
}

func DecodeUsage(m usageMetadata) llm.Usage {
	return llm.RawUsage{
		Prompt:                     m.PromptTokenCount,
		Completion:                 m.CandidatesTokenCount,
		Reasoning:                  m.ThoughtsTokenCount,
		CacheRead:                  m.CachedContentTokenCount,
		ToolPrompt:                 m.ToolUsePromptTokenCount,
		ReasoningOutsideCompletion: true,
		ToolPromptOutsidePrompt:    true,
		Reported:                   true,
	}.Normalise()
}

type idMinter = wirecore.IDMinter

func newIDMinter() *idMinter { return wirecore.NewIDMinter() }
