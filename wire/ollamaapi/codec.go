// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package ollamaapi serves the Ollama /api/chat and /api/generate shapes.
package ollamaapi

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const (
	NameChat     = "ollama"
	NameGenerate = "ollama-generate"
)

type Options struct {
	Now func() time.Time
}

func (o Options) now() time.Time {
	if o.Now != nil {
		return o.Now()
	}
	return time.Now()
}

type ollamaOptions struct {
	Temperature      *float64 `json:"temperature"`
	TopP             *float64 `json:"top_p"`
	TopK             *int     `json:"top_k"`
	NumPredict       *int     `json:"num_predict"`
	Seed             *int64   `json:"seed"`
	Stop             []string `json:"stop"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
	PresencePenalty  *float64 `json:"presence_penalty"`
}

var modelledOptions = map[string]bool{
	"temperature": true, "top_p": true, "top_k": true,
	"num_predict": true, "seed": true, "stop": true,
	"frequency_penalty": true, "presence_penalty": true,
}

func (o *ollamaOptions) apply(r *llm.Request, format string) llm.Diagnostics {
	if o == nil {
		return nil
	}
	r.Temperature = o.Temperature
	r.TopP = o.TopP
	r.TopK = o.TopK
	r.Seed = o.Seed
	r.Stop = o.Stop
	r.FrequencyPenalty = o.FrequencyPenalty
	r.PresencePenalty = o.PresencePenalty
	if o.NumPredict == nil || *o.NumPredict >= 0 {
		r.MaxOutputTokens = o.NumPredict
		return nil
	}
	return llm.Diagnostics{{
		Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageDecode,
		Severity: llm.SeverityInfo, Format: format, Field: "options.num_predict",
		Detail: "a negative ceiling is this wire's way of saying there is none, which the neutral request says by leaving it unset",
	}}
}

func doneReason(s llm.StopReason) (string, bool) {
	switch s {
	case llm.StopEnd, llm.StopSequence, llm.StopRefusal, llm.StopPause:
		return "stop", true
	case llm.StopLength, llm.StopContextOverflow:
		return "length", true
	case llm.StopToolUse:
		return "stop", true
	case llm.StopContentFilter:
		return "stop", false
	case llm.StopNone:
		return "", true
	default:
		return "stop", false
	}
}

func stopReason(s string) llm.StopReason {
	switch s {
	case "stop":
		return llm.StopEnd
	case "length":
		return llm.StopLength
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

func decodeFormat(raw json.RawMessage) *llm.ResponseFormat {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil
		}
		return &llm.ResponseFormat{Type: "json_object"}
	}
	return &llm.ResponseFormat{Type: "json_schema", Schema: raw}
}

func encodeFormat(f *llm.ResponseFormat) json.RawMessage {
	switch {
	case f == nil:
		return nil
	case len(f.Schema) > 0:
		return f.Schema
	default:
		return json.RawMessage(`"json"`)
	}
}

func decodeImages(images []string) []llm.Part {
	var out []llm.Part
	for _, s := range images {
		if b, err := base64.StdEncoding.DecodeString(s); err == nil {
			out = append(out, &llm.Image{Data: b})
			continue
		}
		out = append(out, &llm.Image{URL: s})
	}
	return out
}

func toolsFrom(raw json.RawMessage) ([]llm.Tool, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
	}
	out := make([]llm.Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, llm.Tool{Name: t.Function.Name, Description: t.Function.Description, Schema: t.Function.Parameters})
	}
	return out, nil
}

type wireUsage struct {
	TotalDuration      int64 `json:"total_duration,omitempty"`
	PromptEvalCount    int   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64 `json:"prompt_eval_duration,omitempty"`
	EvalCount          int   `json:"eval_count,omitempty"`
	EvalDuration       int64 `json:"eval_duration,omitempty"`
}

func encodeUsage(u llm.Usage) wireUsage {
	return wireUsage{
		TotalDuration:   u.DurationMS * int64(time.Millisecond),
		PromptEvalCount: u.Prompt,
		EvalCount:       u.Completion,
	}
}

func stamp(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339Nano)
}

func bearer(h string) string {
	const p = "Bearer "
	if len(h) > len(p) && strings.EqualFold(h[:len(p)], p) {
		return h[len(p):]
	}
	return ""
}

func field[T any](m map[string]json.RawMessage, key string, dst *T) error {
	raw, ok := m[key]
	if !ok || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return llm.Errorf(llm.KindInvalidRequest, "%s: %v", key, err)
	}
	return nil
}

func drop(format, code, detail string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: code, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: format, Detail: detail,
	}
}

var _ = wire.Route{}

func decodeThink(raw json.RawMessage) (*llm.ThinkingConfig, llm.Effort, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, "", nil
	}
	var on bool
	if err := json.Unmarshal(raw, &on); err == nil {
		if !on {
			return &llm.ThinkingConfig{}, "", nil
		}
		return &llm.ThinkingConfig{Enabled: true}, "", nil
	}
	var level string
	if err := json.Unmarshal(raw, &level); err != nil {
		return nil, "", llm.Errorf(llm.KindInvalidRequest, "think: %v", err)
	}
	switch llm.Effort(level) {
	case llm.EffortLow, llm.EffortMedium, llm.EffortHigh:
		return &llm.ThinkingConfig{Enabled: true}, llm.Effort(level), nil
	default:
		return nil, "", llm.Errorf(llm.KindInvalidRequest, "think: unknown level %q", level)
	}
}

func unwrapArgs(raw json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '"' {
		return raw
	}
	var inner string
	if json.Unmarshal(trimmed, &inner) != nil {
		return raw
	}
	unquoted := json.RawMessage(strings.TrimSpace(inner))
	if len(unquoted) > 0 && (unquoted[0] == '{' || unquoted[0] == '[') && json.Valid(unquoted) {
		return unquoted
	}
	return raw
}
