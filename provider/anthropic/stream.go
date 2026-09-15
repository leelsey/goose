// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"encoding/json"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	body, _, err := p.buildRequest(a, true)
	if err != nil {
		return nil, err
	}
	h := p.header(a)
	h.Set("Accept", "text/event-stream")
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.endpoint(a, true), Body: body, Header: h, Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}

	sc := wirecore.NewScanner(resp.Body, a.Limits.MaxFrameBytes, a.Limits.MaxBufferBytes)
	st := &parser{provider: p.ID(), sc: sc, blocks: map[int]*blockState{}, attempt: a}
	stream := llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Requested: a.Request.Model, Model: a.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	)
	return stream, nil
}

type blockState struct {
	kind string
	id   string
	name string
	idx  int
}

type parser struct {
	provider string
	sc       *wirecore.Scanner
	blocks   map[int]*blockState
	done     bool
	attempt  *llm.Attempt
}

func (p *parser) next() (*llm.Frame, error) {
	for {
		if p.done {
			return nil, nil
		}
		if !p.sc.Scan() {
			if err := p.sc.Err(); err != nil {
				return nil, &llm.Error{Kind: llm.KindTransport, Provider: p.provider, Message: err.Error(), Sent: true, Err: err}
			}
			p.done = true
			return nil, nil
		}
		ev := p.sc.Event()
		if len(ev.Data) == 0 {
			continue
		}
		name := ev.Name
		if name == "" {
			var typed struct {
				Type string `json:"type"`
			}
			if json.Unmarshal(ev.Data, &typed) == nil {
				name = typed.Type
			}
		}
		frame, err := p.event(name, ev.Data)
		if err != nil {
			return nil, err
		}
		if frame != nil {
			return frame, nil
		}
	}
}

func (p *parser) event(name string, data []byte) (*llm.Frame, error) {
	switch name {
	case "ping", "content_block_stop":
		return nil, nil

	case "message_start":
		var m struct {
			Message json.RawMessage `json:"message"`
		}
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, p.malformed(name, err)
		}
		var msg struct {
			ID    string              `json:"id"`
			Model string              `json:"model"`
			Usage *anthropicmsg.Usage `json:"usage"`
		}
		if err := json.Unmarshal(m.Message, &msg); err != nil {
			return nil, p.malformed(name, err)
		}
		f := &llm.Frame{
			ID: msg.ID, Model: msg.Model, Role: llm.RoleAssistant,
			Extra: wirecore.Extras(m.Message, responseKnown, p.attempt, anthropicmsg.Name),
		}
		if msg.Usage != nil {
			u := anthropicmsg.DecodeUsage(*msg.Usage)
			f.Usage = &u
		}
		return f, nil

	case "content_block_start":
		var m struct {
			Index int             `json:"index"`
			Block json.RawMessage `json:"content_block"`
		}
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, p.malformed(name, err)
		}
		var b struct {
			Type string `json:"type"`
			ID   string `json:"id"`
			Name string `json:"name"`
			Text string `json:"text"`
			Data string `json:"data"`
		}
		if err := json.Unmarshal(m.Block, &b); err != nil {
			return nil, p.malformed(name, err)
		}
		st := &blockState{kind: b.Type, id: b.ID, name: b.Name, idx: len(p.blocks)}
		p.blocks[m.Index] = st
		switch b.Type {
		case "tool_use":
			idx := st.idx
			return &llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: b.ID, Name: b.Name, Index: &idx}}}, nil
		case "redacted_thinking":
			return &llm.Frame{Delta: []llm.Part{&llm.Thinking{Redacted: []byte(b.Data)}}}, nil
		case "text":
			if b.Text != "" {
				return &llm.Frame{Delta: []llm.Part{&llm.Text{Text: b.Text}}}, nil
			}
		case "thinking":
		default:
			return &llm.Frame{
				Delta: []llm.Part{&llm.Raw{Vendor: anthropicmsg.Name, Type: b.Type, JSON: m.Block}},
				Diagnostics: llm.Diagnostics{{
					Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
					Severity: llm.SeverityInfo, Provider: p.provider, Field: b.Type,
					Detail: "content block carried verbatim",
				}},
			}, nil
		}
		return nil, nil

	case "content_block_delta":
		var m struct {
			Index int `json:"index"`
			Delta struct {
				Type      string `json:"type"`
				Text      string `json:"text"`
				Thinking  string `json:"thinking"`
				Signature string `json:"signature"`
				Partial   string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, p.malformed(name, err)
		}
		switch m.Delta.Type {
		case "text_delta":
			return &llm.Frame{Delta: []llm.Part{&llm.Text{Text: m.Delta.Text}}}, nil
		case "thinking_delta":
			return &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: m.Delta.Thinking}}}, nil
		case "signature_delta":
			return &llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: m.Delta.Signature}}}, nil
		case "input_json_delta":
			st := p.blocks[m.Index]
			idx := 0
			if st != nil {
				idx = st.idx
			}
			return &llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &idx, Args: json.RawMessage(m.Delta.Partial)}}}, nil
		}
		return nil, nil

	case "message_delta":
		var m struct {
			Delta struct {
				StopReason   string `json:"stop_reason"`
				StopSequence string `json:"stop_sequence"`
			} `json:"delta"`
			Usage *anthropicmsg.Usage `json:"usage"`
		}
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, p.malformed(name, err)
		}
		f := &llm.Frame{Stop: decodeStop(m.Delta.StopReason), StopSequence: m.Delta.StopSequence}
		if m.Usage != nil {
			u := anthropicmsg.DecodeUsage(*m.Usage)
			f.Usage = &u
		}
		return f, nil

	case "message_stop":
		p.done = true
		return &llm.Frame{TurnEnd: true}, nil

	case "error":
		var m struct {
			Error struct {
				Type    string `json:"type"`
				Message string `json:"message"`
			} `json:"error"`
		}
		_ = json.Unmarshal(data, &m)
		p.done = true
		return &llm.Frame{Err: &llm.Error{
			Kind: kindForType(m.Error.Type), Provider: p.provider,
			Code: m.Error.Type, Message: m.Error.Message, Sent: true,
			Retryable: m.Error.Type == "overloaded_error" || m.Error.Type == "api_error",
		}}, nil

	default:
		return nil, nil
	}
}

func (p *parser) malformed(event string, err error) error {
	return &llm.Error{
		Kind: llm.KindUpstream, Provider: p.provider, Sent: true,
		Code: llm.WireFrameMalformed, Message: "malformed " + event + " frame: " + err.Error(), Err: err,
	}
}

func kindForType(t string) llm.ErrorKind {
	switch t {
	case "invalid_request_error":
		return llm.KindInvalidRequest
	case "authentication_error":
		return llm.KindAuth
	case "permission_error":
		return llm.KindPermission
	case "not_found_error":
		return llm.KindNotFound
	case "rate_limit_error":
		return llm.KindRateLimit
	case "overloaded_error":
		return llm.KindOverloaded
	default:
		return llm.KindUpstream
	}
}
