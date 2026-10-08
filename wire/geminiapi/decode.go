// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"encoding/base64"
	"encoding/json"
	"strconv"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

var known = map[string]bool{
	"contents": true, "systemInstruction": true, "system_instruction": true,
	"tools": true, "toolConfig": true, "tool_config": true,
	"generationConfig": true, "generation_config": true}

type genConfig struct {
	Temperature      *float64        `json:"temperature"`
	TopP             *float64        `json:"topP"`
	TopK             *int            `json:"topK"`
	MaxOutputTokens  *int            `json:"maxOutputTokens"`
	StopSequences    []string        `json:"stopSequences"`
	CandidateCount   int             `json:"candidateCount"`
	ResponseMimeType string          `json:"responseMimeType"`
	ResponseSchema   json.RawMessage `json:"responseSchema"`
	Seed             *int64          `json:"seed"`
	FrequencyPenalty *float64        `json:"frequencyPenalty"`
	PresencePenalty  *float64        `json:"presencePenalty"`
	ThinkingConfig   json.RawMessage `json:"thinkingConfig"`
}

var modelledConfig = map[string]bool{
	"temperature": true, "topp": true, "topk": true, "maxoutputtokens": true,
	"stopsequences": true, "candidatecount": true, "responsemimetype": true,
	"responseschema": true, "seed": true, "thinkingconfig": true,
	"frequencypenalty": true, "presencepenalty": true,
}

var modelledThinking = map[string]bool{"thinkingbudget": true, "includethoughts": true}

func (c *Codec) Decode(in *wire.Inbound) (*llm.Request, wire.Options, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	opts := wire.Options{Format: Name, Header: in.Header}

	model, action, err := parsePath(in.Path)
	if err != nil {
		return nil, opts, diags, err
	}
	switch action {
	case "generateContent":
	case "streamGenerateContent":
		opts.Stream = true
	default:
		return nil, opts, diags, &llm.Error{
			Kind:    llm.KindNotFound,
			Message: "this proxy serves generateContent and streamGenerateContent; " + action + " is not one of them",
		}
	}
	if in.Query != nil && in.Query.Get("alt") == "sse" {
		opts.Stream = true
	}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(in.Body, &top); err != nil {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "malformed JSON body: %v", err)
	}

	req := &llm.Request{Model: model}
	opts.RequestModel = model

	var contents []struct {
		Role  string            `json:"role"`
		Parts []json.RawMessage `json:"parts"`
	}
	if err := field(top, "contents", &contents); err != nil {
		return nil, opts, diags, err
	}
	if len(contents) == 0 {
		return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "contents is required")
	}
	mint := newIDMinters()
	for _, ct := range contents {
		role := llm.RoleUser
		switch ct.Role {
		case "model":
			role = llm.RoleAssistant
		case "", "user":
		default:
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityLoss, Format: Name, Field: "role",
				Detail: "role " + strconv.Quote(ct.Role) + " is not one this format has, so the turn was read as the caller's",
			})
		}
		parts, d, err := decodeParts(ct.Parts, mint, true)
		if err != nil {
			return nil, opts, diags, err
		}
		diags = append(diags, d...)
		req.Messages = append(req.Messages, llm.Message{Role: role, Content: parts})
	}

	sys := top["systemInstruction"]
	if len(sys) == 0 {
		sys = top["system_instruction"]
	}
	if len(sys) > 0 && string(sys) != "null" {
		var block struct {
			Parts []json.RawMessage `json:"parts"`
		}
		if err := json.Unmarshal(sys, &block); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "systemInstruction: %v", err)
		}
		parts, d, err := decodeParts(block.Parts, mint, true)
		if err != nil {
			return nil, opts, diags, err
		}
		diags = append(diags, d...)
		req.System = parts
	}

	cfgRaw := top["generationConfig"]
	if len(cfgRaw) == 0 {
		cfgRaw = top["generation_config"]
	}
	if len(cfgRaw) > 0 && string(cfgRaw) != "null" {
		var cfg genConfig
		if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "generationConfig: %v", err)
		}
		req.Temperature, req.TopP, req.TopK = cfg.Temperature, cfg.TopP, cfg.TopK
		req.MaxOutputTokens, req.Seed, req.Stop = cfg.MaxOutputTokens, cfg.Seed, cfg.StopSequences
		req.Candidates = cfg.CandidateCount
		req.FrequencyPenalty, req.PresencePenalty = cfg.FrequencyPenalty, cfg.PresencePenalty
		for _, name := range wirecore.Shadowed(cfgRaw, modelledConfig) {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityLoss, Format: Name, Field: "generationConfig" + "." + name,
				Detail: "this setting arrived spelled more than one way, and only the last of them was read",
			})
		}
		rest := map[string]json.RawMessage{}
		if kept := wirecore.NestedExtra(cfgRaw, modelledConfig); kept != nil {
			if err := json.Unmarshal(kept, &rest); err != nil {
				return nil, opts, diags, llm.Errorf(llm.KindInternal, "generationConfig: %v", err)
			}
		}
		if cfg.ResponseMimeType == "application/json" || len(cfg.ResponseSchema) > 0 {
			req.ResponseFormat = &llm.ResponseFormat{Type: "json_schema", Schema: cfg.ResponseSchema}
			if len(cfg.ResponseSchema) == 0 {
				req.ResponseFormat.Type = "json_object"
			}
		}
		if len(cfg.ThinkingConfig) > 0 && string(cfg.ThinkingConfig) != "null" {
			thinking, carried, err := decodeThinking(cfg.ThinkingConfig)
			if err != nil {
				return nil, opts, diags, err
			}
			req.Thinking = thinking
			if carried != nil {
				rest["thinkingConfig"] = carried
			}
		}
		if len(rest) > 0 {
			raw, err := json.Marshal(rest)
			if err != nil {
				return nil, opts, diags, llm.Errorf(llm.KindInternal, "generationConfig: %v", err)
			}
			if req.Extra == nil {
				req.Extra = map[string]json.RawMessage{}
				req.ExtraFrom = Name
			}
			req.Extra["generationConfig"] = raw
		}
	}

	if raw, ok := top["tools"]; ok && string(raw) != "null" {
		var tools []struct {
			FunctionDeclarations []struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"functionDeclarations"`
		}
		if err := json.Unmarshal(raw, &tools); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "tools: %v", err)
		}
		for _, t := range tools {
			for _, fd := range t.FunctionDeclarations {
				req.Tools = append(req.Tools, llm.Tool{Name: fd.Name, Description: fd.Description, Schema: fd.Parameters})
			}
		}
	}

	tc := top["toolConfig"]
	if len(tc) == 0 {
		tc = top["tool_config"]
	}
	if len(tc) > 0 && string(tc) != "null" {
		var cfg struct {
			FunctionCallingConfig struct {
				Mode                 string   `json:"mode"`
				AllowedFunctionNames []string `json:"allowedFunctionNames"`
			} `json:"functionCallingConfig"`
		}
		if err := json.Unmarshal(tc, &cfg); err != nil {
			return nil, opts, diags, llm.Errorf(llm.KindInvalidRequest, "toolConfig: %v", err)
		}
		switch cfg.FunctionCallingConfig.Mode {
		case "AUTO", "":
			req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolAuto}
		case "NONE":
			req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolNone}
		case "ANY":
			ch := &llm.ToolChoice{Mode: llm.ToolRequired}
			if n := cfg.FunctionCallingConfig.AllowedFunctionNames; len(n) == 1 {
				ch.Mode, ch.Name = llm.ToolSpecific, n[0]
			}
			req.ToolChoice = ch
		}
	}

	for k, v := range top {
		if known[k] {
			continue
		}
		if req.Extra == nil {
			req.Extra = map[string]json.RawMessage{}
			req.ExtraFrom = Name
		}
		req.Extra[k] = v
	}

	opts.Key = in.Get("x-goog-api-key")
	if opts.Key == "" && in.Query != nil {
		opts.Key = in.Query.Get("key")
	}
	return req, opts, diags, nil
}

// decodeThinking reads a thinkingConfig into the neutral request and returns
// what the neutral request has no field for, such as a thinking level, to
// travel to a backend on this wire. This wire hands the thoughts back only
// when asked to.
func decodeThinking(raw json.RawMessage) (*llm.ThinkingConfig, json.RawMessage, error) {
	var th struct {
		ThinkingBudget  *int `json:"thinkingBudget"`
		IncludeThoughts bool `json:"includeThoughts"`
	}
	if err := json.Unmarshal(raw, &th); err != nil {
		return nil, nil, llm.Errorf(llm.KindInvalidRequest, "generationConfig.thinkingConfig: %v", err)
	}
	out := &llm.ThinkingConfig{Enabled: true}
	if th.ThinkingBudget != nil {
		out.Budget = *th.ThinkingBudget
		out.Enabled = *th.ThinkingBudget != 0
	}
	// With thinking off there is no level to keep and no thought to show.
	if !out.Enabled {
		return out, nil, nil
	}
	out.Hidden = !th.IncludeThoughts
	return out, wirecore.NestedExtra(raw, modelledThinking), nil
}

// DecodeParts reads parts a Gemini backend sent, so every signature on them is
// Google's as it stands.
func DecodeParts(raw []json.RawMessage) ([]llm.Part, llm.Diagnostics, error) {
	return decodeParts(raw, newIDMinters(), false)
}

// textKeys are what a text part is made of. A signed part with anything else
// is a kind this wire does not model, and read as text it would lose that.
var textKeys = map[string]bool{"text": true, "thought": true, "thoughtSignature": true}

func textLike(item json.RawMessage) bool {
	var keys map[string]json.RawMessage
	if json.Unmarshal(item, &keys) != nil {
		return false
	}
	for k := range keys {
		if !textKeys[k] {
			return false
		}
	}
	return true
}

// decodeParts reads a turn's parts. A client's signature may be one goose
// handed it for another family, and says so; a backend's is its own.
func decodeParts(raw []json.RawMessage, mint *idMinters, client bool) ([]llm.Part, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	var out []llm.Part
	for _, item := range raw {
		var p struct {
			Text             string `json:"text"`
			Thought          bool   `json:"thought"`
			ThoughtSignature string `json:"thoughtSignature"`
			InlineData       *struct {
				MimeType string `json:"mimeType"`
				Data     string `json:"data"`
			} `json:"inlineData"`
			FileData *struct {
				MimeType string `json:"mimeType"`
				FileURI  string `json:"fileUri"`
			} `json:"fileData"`
			FunctionCall *struct {
				ID   string          `json:"id"`
				Name string          `json:"name"`
				Args json.RawMessage `json:"args"`
			} `json:"functionCall"`
			FunctionResponse *struct {
				ID       string          `json:"id"`
				Name     string          `json:"name"`
				Response json.RawMessage `json:"response"`
			} `json:"functionResponse"`
		}
		if err := json.Unmarshal(item, &p); err != nil {
			return nil, diags, llm.Errorf(llm.KindInvalidRequest, "parts: %v", err)
		}
		signer, sig := llm.Family(""), p.ThoughtSignature
		switch {
		case client:
			signer, sig = wirecore.OpenBytesSignature(llm.FamilyGoogle, sig)
		case sig != "":
			signer = llm.FamilyGoogle
		}
		switch {
		case p.FunctionCall != nil:
			id := p.FunctionCall.ID
			if id == "" {
				id = mint.calls.Next(p.FunctionCall.Name)
			}
			out = append(out, &llm.ToolUse{ID: id, Name: p.FunctionCall.Name, Args: p.FunctionCall.Args, Signature: sig, Signer: signer})
		case p.FunctionResponse != nil:
			id := p.FunctionResponse.ID
			if id == "" {
				id = mint.results.Next(p.FunctionResponse.Name)
			}
			out = append(out, &llm.ToolResult{ID: id, Name: p.FunctionResponse.Name,
				Content: []llm.Part{&llm.Text{Text: string(p.FunctionResponse.Response)}}})
		case p.InlineData != nil:
			b, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
			if err != nil {
				return nil, diags, llm.Errorf(llm.KindInvalidRequest, "inlineData: %v", err)
			}
			out = append(out, &llm.Image{MIME: p.InlineData.MimeType, Data: b, Signature: sig, Signer: signer})
		case p.FileData != nil:
			out = append(out, &llm.Image{MIME: p.FileData.MimeType, URL: p.FileData.FileURI, Signature: sig, Signer: signer})
		case p.Thought:
			out = append(out, &llm.Thinking{Text: p.Text, Signature: sig, Signer: signer})
		case p.Text != "" || sig != "" && textLike(item):
			out = append(out, &llm.Text{Text: p.Text, Signature: sig, Signer: signer})
		default:
			out = append(out, &llm.Raw{Vendor: Name, Type: "part", JSON: item})
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageDecode,
				Severity: llm.SeverityInfo, Format: Name, Detail: "part carried verbatim",
			})
		}
	}
	return out, diags, nil
}

func field[T any](m map[string]json.RawMessage, key string, dst *T) error {
	return wirecore.Field(m, key, dst)
}
