// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"encoding/json"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

type codexEvent struct {
	Type     string     `json:"type"`
	ThreadID string     `json:"thread_id"`
	Item     *codexItem `json:"item"`
	Message  string     `json:"message"`
	Error    struct {
		Message string `json:"message"`
	} `json:"error"`
	Usage *codexUsage `json:"usage"`
}

type codexItem struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Text    string `json:"text"`
	Message string `json:"message"`
	Name    string `json:"name"`
	Command string `json:"command"`
	raw     json.RawMessage
}

func (it *codexItem) UnmarshalJSON(b []byte) error {
	type alias codexItem
	var a alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*it = codexItem(a)
	it.raw = append(json.RawMessage(nil), b...)
	return nil
}

var codexToolItem = map[string]bool{
	"command_execution": true, "mcp_tool_call": true,
	"file_change": true, "web_search": true, "patch_apply": true,
}

// An item names the call in its own bookkeeping; what describes the call is
// everything else, so the naming is not handed over as an argument.
func codexToolArgs(raw json.RawMessage) json.RawMessage {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return wirecore.ObjectArgs(raw)
	}
	for _, key := range []string{"id", "type", "name", "message"} {
		delete(obj, key)
	}
	trimmed, err := json.Marshal(obj)
	if err != nil {
		return wirecore.ObjectArgs(raw)
	}
	return wirecore.ObjectArgs(trimmed)
}

type codexUsage struct {
	InputTokens           int `json:"input_tokens"`
	CachedInputTokens     int `json:"cached_input_tokens"`
	OutputTokens          int `json:"output_tokens"`
	ReasoningOutputTokens int `json:"reasoning_output_tokens"`
}

func (u *codexUsage) neutral() llm.Usage {
	if u == nil {
		return llm.Usage{}
	}
	return llm.RawUsage{
		Prompt: u.InputTokens, Completion: u.OutputTokens,
		CacheRead: u.CachedInputTokens, Reasoning: u.ReasoningOutputTokens,
		Reported: true,
	}.Normalise()
}

func codexParse(p *Provider, line []byte, st *state) ([]*llm.Frame, error) {
	var evt codexEvent
	if err := unmarshal(line, &evt, p.ID()); err != nil {
		return nil, err
	}
	switch evt.Type {
	case "thread.started":
		if evt.ThreadID != "" {
			st.began(p, evt.ThreadID)
		}
		return []*llm.Frame{{Role: llm.RoleAssistant, Session: st.session}}, nil

	case "item.completed":
		if evt.Item == nil {
			return nil, nil
		}
		switch evt.Item.Type {
		case "agent_message":
			if evt.Item.Text == "" {
				return nil, nil
			}
			st.lastWasTool = false
			return []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: evt.Item.Text}}}}, nil
		case "reasoning":
			if evt.Item.Text == "" {
				return nil, nil
			}
			st.lastWasTool = false
			return []*llm.Frame{{Delta: []llm.Part{&llm.Thinking{Text: evt.Item.Text}}}}, nil
		case "error":
			return []*llm.Frame{{Err: &llm.Error{
				Kind: llm.KindUpstream, Provider: p.ID(), Sent: true,
				Message: firstNonEmpty(evt.Item.Message, "the item failed"), Body: httpcore.ErrorBody([]byte(evt.Item.Message)),
			}}}, nil
		}
		if codexToolItem[evt.Item.Type] {
			return st.toolFrame(
				firstNonEmpty(evt.Item.ID, evt.Item.Type),
				firstNonEmpty(evt.Item.Name, evt.Item.Type),
				codexToolArgs(evt.Item.raw)), nil
		}
		if len(evt.Item.raw) > 0 {
			return []*llm.Frame{{
				Delta: []llm.Part{&llm.Raw{Vendor: "codex", Type: evt.Item.Type, JSON: evt.Item.raw}},
				Diagnostics: llm.Diagnostics{{
					Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
					Severity: llm.SeverityInfo, Provider: p.ID(), Field: evt.Item.Type,
					Detail: "item carried verbatim",
				}},
			}}, nil
		}
		return nil, nil

	case "turn.completed":
		u := evt.Usage.neutral()
		f := &llm.Frame{Usage: &u, TurnEnd: true, Session: st.session}
		f.Stop = st.ending()
		return []*llm.Frame{f}, nil

	case "turn.failed", "error":
		said := firstNonEmpty(evt.Message, evt.Error.Message)
		return []*llm.Frame{{Err: &llm.Error{
			Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Retryable: true,
			Message: firstNonEmpty(said, "the turn failed"), Body: httpcore.ErrorBody([]byte(said)),
		}}}, nil
	}
	return nil, nil
}

type antigravityEvent struct {
	Type      string          `json:"type"`
	SessionID string          `json:"session_id"`
	Text      string          `json:"text"`
	Content   string          `json:"content"`
	Name      string          `json:"name"`
	ID        string          `json:"id"`
	Arguments json.RawMessage `json:"arguments"`
	Error     string          `json:"error"`
	Message   string          `json:"message"`
	Stats     *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CandidatesTokens int `json:"candidates_tokens"`
		ThoughtsTokens   int `json:"thoughts_tokens"`
		CachedTokens     int `json:"cached_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"stats"`
}

func antigravityParse(p *Provider, line []byte, st *state) ([]*llm.Frame, error) {
	var evt antigravityEvent
	if err := unmarshal(line, &evt, p.ID()); err != nil {
		return nil, err
	}
	if evt.SessionID != "" {
		st.began(p, evt.SessionID)
	}
	switch evt.Type {
	case "init":
		return []*llm.Frame{{Role: llm.RoleAssistant, Session: st.session}}, nil

	case "message":
		text := firstNonEmpty(evt.Text, evt.Content)
		if text == "" {
			return nil, nil
		}
		st.lastWasTool = false
		return []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: text}}}}, nil

	case "tool_use":
		return st.toolFrame(evt.ID, evt.Name, evt.Arguments), nil

	case "result":
		u := llm.Usage{}
		if evt.Stats != nil {
			u = llm.RawUsage{
				Prompt: evt.Stats.PromptTokens, Completion: evt.Stats.CandidatesTokens,
				Reasoning: evt.Stats.ThoughtsTokens, CacheRead: evt.Stats.CachedTokens,
				ReasoningOutsideCompletion: true, Reported: true,
			}.Normalise()
		}
		f := &llm.Frame{Usage: &u, TurnEnd: true, Session: st.session}
		f.Stop = st.ending()
		return []*llm.Frame{f}, nil

	case "error":
		said := firstNonEmpty(evt.Error, evt.Message)
		return []*llm.Frame{{Err: &llm.Error{
			Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Retryable: true,
			Message: firstNonEmpty(said, "the invocation failed"), Body: httpcore.ErrorBody([]byte(said)),
		}}}, nil
	}
	return nil, nil
}

type copilotEvent struct {
	Type         string          `json:"type"`
	SessionID    string          `json:"session_id"`
	ThreadID     string          `json:"thread_id"`
	ID           string          `json:"id"`
	Text         string          `json:"text"`
	Content      json.RawMessage `json:"content"`
	Delta        json.RawMessage `json:"delta"`
	ContentBlock json.RawMessage `json:"content_block"`
	Name         string          `json:"name"`
	Arguments    json.RawMessage `json:"arguments"`
	Error        struct {
		Message string `json:"message"`
	} `json:"error"`
	Message string `json:"message"`
	Usage   *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

func copilotParse(p *Provider, line []byte, st *state) ([]*llm.Frame, error) {
	var evt copilotEvent
	if err := unmarshal(line, &evt, p.ID()); err != nil {
		return nil, err
	}
	if id := firstNonEmpty(evt.SessionID, evt.ThreadID); id != "" {
		st.began(p, id)
	}
	switch evt.Type {
	case "session.started", "session.start", "session", "ready", "thread.started", "init",
		"message_start":
		return []*llm.Frame{{Role: llm.RoleAssistant, Session: st.session}}, nil

	case "message", "message.delta", "assistant_message", "item.assistant_message", "text", "content",
		"content_block_delta", "content_block_start", "content.delta", "text.delta":
		text := firstNonEmpty(evt.Text, jsonText(evt.Delta), jsonText(evt.Content), jsonText(evt.ContentBlock))
		if text == "" {
			return nil, nil
		}
		st.lastWasTool = false
		return []*llm.Frame{{Delta: []llm.Part{&llm.Text{Text: text}}}}, nil

	case "tool_use", "tool.use", "tool.call", "tool_call", "item.tool_use":
		return st.toolFrame(evt.ID, evt.Name, evt.Arguments), nil

	case "completion", "completed", "complete", "done", "finished", "end", "turn.completed", "result",
		"message_delta", "message_stop":
		u := llm.Usage{}
		if evt.Usage != nil {
			prompt := evt.Usage.InputTokens
			if prompt == 0 {
				prompt = evt.Usage.PromptTokens
			}
			u = llm.RawUsage{Prompt: prompt, Completion: evt.Usage.OutputTokens, Reported: true}.Normalise()
		}
		f := &llm.Frame{Usage: &u, TurnEnd: true, Session: st.session}
		f.Stop = st.ending()
		return []*llm.Frame{f}, nil

	case "error", "failed":
		said := firstNonEmpty(evt.Error.Message, evt.Message)
		return []*llm.Frame{{Err: &llm.Error{
			Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Retryable: true,
			Message: firstNonEmpty(said, "the invocation failed"), Body: httpcore.ErrorBody([]byte(said)),
		}}}, nil
	}
	return nil, nil
}

func jsonText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) == nil {
		var out string
		for _, b := range blocks {
			out += b.Text
		}
		return out
	}
	var obj struct {
		Text    string `json:"text"`
		Content string `json:"content"`
	}
	if json.Unmarshal(raw, &obj) == nil {
		return firstNonEmpty(obj.Text, obj.Content)
	}
	return ""
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
