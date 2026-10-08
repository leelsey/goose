// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"encoding/json"

	"github.com/leelsey/goose/internal/awsstream"
	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	body, _, err := p.body(a)
	if err != nil {
		return nil, err
	}
	h := p.header()
	h.Set("Accept", "application/vnd.amazon.eventstream")
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.url(a, "converse-stream"), Body: body, Header: h, Sign: p.sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	st := &parser{
		ctx:      ctx,
		provider: p.ID(),
		reader:   awsstream.NewReader(resp.Body, a.Limits.MaxFrameBytes),
		blocks:   map[int]string{},
	}
	return llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	), nil
}

type parser struct {
	ctx      context.Context
	provider string
	reader   *awsstream.Reader
	blocks   map[int]string
	done     bool
	ended    bool
}

func (b *parser) next() (*llm.Frame, error) {
	for {
		if b.done {
			return nil, nil
		}
		if !b.reader.Next() {
			if err := b.reader.Err(); err != nil {
				return nil, httpcore.ReadError(b.ctx, b.provider, err)
			}
			b.done = true
			if !b.ended {
				b.ended = true
				return &llm.Frame{TurnEnd: true}, nil
			}
			return nil, nil
		}
		msg := b.reader.Message()
		if msg.Kind() == "exception" || msg.Kind() == "error" ||
			msg.Headers[":exception-type"] != "" || msg.Headers[":error-code"] != "" {
			b.done = true
			return &llm.Frame{Err: b.exception(msg)}, nil
		}
		frame, err := b.event(msg.Type(), msg.Payload)
		if err != nil {
			return nil, err
		}
		if frame != nil {
			return frame, nil
		}
	}
}

func (b *parser) exception(msg awsstream.Message) *llm.Error {
	var body struct {
		Message string `json:"message"`
	}
	_ = json.Unmarshal(msg.Payload, &body)
	code := msg.Type()
	if code == "" {
		code = msg.Headers[":error-code"]
	}
	if body.Message == "" {
		body.Message = msg.Headers[":error-message"]
	}
	said := body.Message
	if body.Message == "" {
		body.Message = "the stream failed part-way"
	}
	kind := llm.KindUpstream
	retryable := false
	switch code {
	case "throttlingException":
		kind, retryable = llm.KindRateLimit, true
	case "modelStreamErrorException", "internalServerException":
		kind, retryable = llm.KindUpstream, true
	case "validationException":
		kind = llm.KindInvalidRequest
	case "modelTimeoutException":
		kind, retryable = llm.KindTimeout, true
	case "serviceUnavailableException":
		kind, retryable = llm.KindOverloaded, true
	}
	return &llm.Error{
		Kind: kind, Provider: b.provider, Sent: true, Retryable: retryable,
		Code: code, Message: body.Message, Body: httpcore.ErrorBody([]byte(said)),
	}
}

func (b *parser) unmodelled(payload []byte, field string) *llm.Frame {
	var probe map[string]json.RawMessage
	if json.Unmarshal(payload, &probe) != nil {
		return nil
	}
	raw := probe[field]
	if len(raw) == 0 || string(raw) == "null" || string(raw) == "{}" {
		return nil
	}
	return &llm.Frame{
		Delta: []llm.Part{&llm.Raw{Vendor: "bedrock", Type: "contentBlock", JSON: raw}},
		Diagnostics: llm.Diagnostics{{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
			Severity: llm.SeverityInfo, Provider: b.provider,
			Detail: "content block carried verbatim",
		}},
	}
}

func (b *parser) event(name string, payload []byte) (*llm.Frame, error) {
	switch name {
	case "messageStart":
		return &llm.Frame{Role: llm.RoleAssistant}, nil

	case "contentBlockStart":
		var m struct {
			Index int `json:"contentBlockIndex"`
			Start struct {
				ToolUse *struct {
					ID   string `json:"toolUseId"`
					Name string `json:"name"`
				} `json:"toolUse"`
			} `json:"start"`
		}
		if err := json.Unmarshal(payload, &m); err != nil {
			return nil, b.malformed(name, err)
		}
		if m.Start.ToolUse != nil {
			b.blocks[m.Index] = "tool"
			idx := m.Index
			return &llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: m.Start.ToolUse.ID, Name: m.Start.ToolUse.Name, Index: &idx}}}, nil
		}
		return b.unmodelled(payload, "start"), nil

	case "contentBlockDelta":
		var m struct {
			Index int `json:"contentBlockIndex"`
			Delta struct {
				Text    string `json:"text"`
				ToolUse *struct {
					Input string `json:"input"`
				} `json:"toolUse"`
				Reasoning *struct {
					Text            string `json:"text"`
					Signature       string `json:"signature"`
					RedactedContent string `json:"redactedContent"`
				} `json:"reasoningContent"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(payload, &m); err != nil {
			return nil, b.malformed(name, err)
		}
		switch {
		case m.Delta.ToolUse != nil:
			idx := m.Index
			return &llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &idx, Args: json.RawMessage(m.Delta.ToolUse.Input)}}}, nil
		case m.Delta.Reasoning != nil:
			r := m.Delta.Reasoning
			if r.RedactedContent != "" {
				// The shared shape keeps a backend's own spelling, and Anthropic's is
				// the base64 this wire already carries.
				data := []byte(r.RedactedContent)
				return &llm.Frame{Delta: []llm.Part{&llm.Thinking{Redacted: data}}}, nil
			}
			return &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: r.Text, Signature: r.Signature, Signer: llm.FamilyAnthropic}}}, nil
		case m.Delta.Text != "":
			return &llm.Frame{Delta: []llm.Part{&llm.Text{Text: m.Delta.Text}}}, nil
		}
		return b.unmodelled(payload, "delta"), nil

	case "contentBlockStop":
		return nil, nil

	case "messageStop":
		var m struct {
			StopReason string `json:"stopReason"`
		}
		if err := json.Unmarshal(payload, &m); err != nil {
			return nil, b.malformed(name, err)
		}
		return &llm.Frame{Stop: decodeStop(m.StopReason)}, nil

	case "metadata":
		var m struct {
			Usage struct {
				InputTokens           int `json:"inputTokens"`
				OutputTokens          int `json:"outputTokens"`
				CacheReadInputTokens  int `json:"cacheReadInputTokens"`
				CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
			} `json:"usage"`
			Metrics struct {
				LatencyMS int64 `json:"latencyMs"`
			} `json:"metrics"`
		}
		if err := json.Unmarshal(payload, &m); err != nil {
			return nil, b.malformed(name, err)
		}
		u := decodeUsage(m.Usage.InputTokens, m.Usage.OutputTokens,
			m.Usage.CacheReadInputTokens, m.Usage.CacheWriteInputTokens, m.Metrics.LatencyMS)
		b.ended = true
		return &llm.Frame{Usage: &u, TurnEnd: true}, nil

	default:
		return &llm.Frame{Diagnostics: llm.Diagnostics{{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
			Severity: llm.SeverityLoss, Provider: b.provider, Field: name,
			Detail: "stream event not modelled by this adapter",
		}}}, nil
	}
}

func (b *parser) malformed(event string, err error) error {
	return &llm.Error{
		Kind: llm.KindUpstream, Provider: b.provider, Sent: true, Code: llm.WireFrameMalformed,
		Message: "malformed " + event + " event: " + err.Error(), Err: err,
	}
}
