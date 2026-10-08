// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"encoding/json"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

type claudeEvent struct {
	Type      string `json:"type"`
	SessionID string `json:"session_id"`

	Event struct {
		Type    string `json:"type"`
		Index   int    `json:"index"`
		Message struct {
			ID    string              `json:"id"`
			Model string              `json:"model"`
			Usage *anthropicmsg.Usage `json:"usage"`
		} `json:"message"`
		ContentBlock struct {
			Type string `json:"type"`
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"content_block"`
		Delta struct {
			Type        string `json:"type"`
			Text        string `json:"text"`
			Thinking    string `json:"thinking"`
			Signature   string `json:"signature"`
			PartialJSON string `json:"partial_json"`
			StopReason  string `json:"stop_reason"`
		} `json:"delta"`
		Usage *anthropicmsg.Usage `json:"usage"`
	} `json:"event"`

	Message struct {
		ID      string              `json:"id"`
		Model   string              `json:"model"`
		Content []json.RawMessage   `json:"content"`
		Usage   *anthropicmsg.Usage `json:"usage"`
	} `json:"message"`

	Subtype        string              `json:"subtype"`
	IsError        bool                `json:"is_error"`
	APIErrorStatus *int                `json:"api_error_status"`
	Result         string              `json:"result"`
	Usage          *anthropicmsg.Usage `json:"usage"`
	TotalCostUSD   *float64            `json:"total_cost_usd"`
	NumTurns       int                 `json:"num_turns"`
	DurationMS     int64               `json:"duration_ms"`
}

func claudeParse(p *Provider, line []byte, st *state) ([]*llm.Frame, error) {
	var evt claudeEvent
	if err := unmarshal(line, &evt, p.ID()); err != nil {
		return nil, err
	}
	if evt.SessionID != "" {
		st.began(p, evt.SessionID)
	}

	switch evt.Type {
	case "stream_event":
		return claudeStreamEvent(p, &evt, st)

	case "assistant":
		return claudeWholeTurn(p, &evt, st)

	case "result":
		var frames []*llm.Frame
		u := llm.Usage{}
		if evt.Usage != nil {
			u = anthropicmsg.DecodeUsage(*evt.Usage)
		}
		if evt.TotalCostUSD != nil {
			u.Cost = &llm.Cost{Amount: *evt.TotalCostUSD, Currency: "USD"}
		}
		u.DurationMS = evt.DurationMS
		u.Turns = evt.NumTurns
		if !st.sawContent && evt.Result != "" && !evt.IsError {
			st.sawContent = true
			frames = append(frames, &llm.Frame{
				Role: llm.RoleAssistant, Session: st.session,
				Delta: []llm.Part{&llm.Text{Text: evt.Result}},
				Diagnostics: llm.Diagnostics{{
					Code: llm.WireFrameMalformed, Class: llm.ClassTransport, Stage: llm.StageResponse,
					Severity: llm.SeverityInfo, Provider: p.ID(),
					Detail: "this binary streamed no frame goose could read, so the answer was taken from the line that closes the invocation",
				}},
			})
		}
		frames = append(frames, &llm.Frame{
			Session: st.session, Usage: &u, Total: true, TurnEnd: true,
			Stop: st.ending(),
		})
		if evt.IsError || evt.APIErrorStatus != nil {
			status := 0
			if evt.APIErrorStatus != nil {
				status = *evt.APIErrorStatus
			}
			frames = append(frames, &llm.Frame{Err: &llm.Error{
				Kind: llm.KindForStatus(status), Provider: p.ID(), Sent: true, Status: status,
				Message: claudeError(&evt), Body: httpcore.ErrorBody([]byte(evt.Result)),
				Retryable: status == 0 || llm.RetryableStatus(status),
			}})
		}
		return frames, nil
	}
	return nil, nil
}

func claudeWholeTurn(p *Provider, evt *claudeEvent, st *state) ([]*llm.Frame, error) {
	blocks, err := json.Marshal(evt.Message.Content)
	if err != nil {
		return nil, nil
	}
	parts, _, diags, err := anthropicmsg.DecodeBlocks(blocks)
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, nil
	}
	st.sawContent = true
	f := &llm.Frame{
		Role: llm.RoleAssistant, Session: st.session, Delta: parts,
		ID: evt.Message.ID, Model: evt.Message.Model, Diagnostics: diags,
	}
	if evt.Message.Usage != nil {
		u := anthropicmsg.DecodeUsage(*evt.Message.Usage)
		f.Usage = &u
	}
	// The binary prints one of these lines as each block of a message closes,
	// under the message's id and with the usage message_start announced, and
	// before the message is finished. With --include-partial-messages the
	// deltas carry that message, so each of its lines adds only what they had
	// no event for, and its figures are message_start's once more. Without
	// them the lines are the message itself, one turn however many it takes.
	id := evt.Message.ID
	switch {
	case st.streamed && id == st.msg:
		f.Delta, f.Usage = unstreamed(parts), nil
		if len(f.Delta) == 0 && len(f.Diagnostics) == 0 {
			return nil, nil
		}
		return []*llm.Frame{f}, nil
	case st.turnOpen && id != "" && id == st.msg:
		return []*llm.Frame{f}, nil
	}
	var frames []*llm.Frame
	if st.turnOpen {
		frames = append(frames, &llm.Frame{TurnEnd: true})
	}
	st.turnOpen, st.streamed, st.msg = true, false, id
	return append(frames, f), nil
}

// unstreamed keeps the parts of a restated turn that no stream event could
// have carried. claudeStreamEvent spells out text, thinking and tool calls;
// everything else — redacted thinking, a server-side tool, any block this
// binary learns to emit later — reaches the caller on the line restating it.
func unstreamed(parts []llm.Part) []llm.Part {
	var out []llm.Part
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text, *llm.ToolUse:
		case *llm.Thinking:
			if len(v.Redacted) > 0 {
				out = append(out, p)
			}
		default:
			out = append(out, p)
		}
	}
	return out
}

func claudeError(evt *claudeEvent) string {
	switch {
	case evt.Result != "":
		return evt.Result
	case evt.Subtype != "" && evt.Subtype != "success":
		return evt.Subtype
	default:
		return "the invocation reported an error"
	}
}

func claudeStreamEvent(p *Provider, evt *claudeEvent, st *state) ([]*llm.Frame, error) {
	inner := evt.Event
	switch inner.Type {
	case "message_start":
		var frames []*llm.Frame
		if st.turnOpen {
			frames = append(frames, &llm.Frame{TurnEnd: true})
		}
		st.turnOpen, st.streamed, st.msg = true, true, inner.Message.ID
		f := &llm.Frame{Role: llm.RoleAssistant, Session: st.session}
		if inner.Message.ID != "" {
			f.ID = inner.Message.ID
		}
		if inner.Message.Model != "" {
			f.Model = inner.Message.Model
		}
		if inner.Message.Usage != nil {
			u := anthropicmsg.DecodeUsage(*inner.Message.Usage)
			f.Usage = &u
		}
		return append(frames, f), nil

	case "content_block_start":
		st.openTool = inner.ContentBlock.Type == "tool_use"
		if st.openTool {
			idx := st.toolIdx
			st.toolIdx++
			return []*llm.Frame{{Delta: []llm.Part{&llm.ToolUse{
				ID: inner.ContentBlock.ID, Name: inner.ContentBlock.Name, Index: &idx,
			}}}}, nil
		}
		return nil, nil

	case "content_block_delta":
		switch inner.Delta.Type {
		case "text_delta":
			if inner.Delta.Text == "" {
				return nil, nil
			}
			return []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: inner.Delta.Text}}}}, nil
		case "thinking_delta":
			if inner.Delta.Thinking == "" {
				return nil, nil
			}
			return []*llm.Frame{{Delta: []llm.Part{&llm.Thinking{Text: inner.Delta.Thinking}}}}, nil
		case "signature_delta":
			if inner.Delta.Signature == "" {
				return nil, nil
			}
			return []*llm.Frame{{Delta: []llm.Part{&llm.Thinking{Signature: inner.Delta.Signature, Signer: llm.FamilyAnthropic}}}}, nil
		case "input_json_delta":
			// A server tool's input streams too, but belongs to its own block,
			// which the closing line restates whole.
			if inner.Delta.PartialJSON == "" || !st.openTool {
				return nil, nil
			}
			idx := st.toolIdx - 1
			if idx < 0 {
				idx = 0
			}
			return []*llm.Frame{{Delta: []llm.Part{&llm.ToolUse{
				Index: &idx, Args: json.RawMessage(inner.Delta.PartialJSON),
			}}}}, nil
		}
		return nil, nil

	case "message_delta":
		f := &llm.Frame{}
		if inner.Delta.StopReason != "" {
			f.Stop = claudeStop(inner.Delta.StopReason)
			st.finished = true
		}
		if inner.Usage != nil {
			u := anthropicmsg.DecodeUsage(*inner.Usage)
			f.Usage = &u
		}
		if f.Empty() {
			return nil, nil
		}
		return []*llm.Frame{f}, nil
	}
	return nil, nil
}

func claudeStop(s string) llm.StopReason {
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
	default:
		return llm.StopUnknown
	}
}
