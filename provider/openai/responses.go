// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"strings"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire/openairesp"
)

func (p *Provider) responsesBody(a *llm.Attempt, stream bool) ([]byte, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	req := a.Request
	body := map[string]any{"model": p.model(a)}

	system, msgs := req.SystemAndMessages()
	if len(system) > 0 {
		body["instructions"] = llm.TextOf(system)
		diags = append(diags, llm.Flattened(system, p.ID(),
			"this endpoint takes instructions as text, so what was not text did not travel")...)
	}
	input, d := responsesInput(msgs, p.ID())
	diags = append(diags, d...)
	if len(input) == 0 && body["instructions"] == nil {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "no input survived translation")
	}
	body["input"] = input

	if u, key := caller(req.Metadata); u != "" {
		body["user"] = u
		if rest := unsentMeta(req.Metadata, key); len(rest) > 0 {
			diags = append(diags, metaDropped(p.ID(), rest))
		}
	} else if rest := unsentMeta(req.Metadata, ""); len(rest) > 0 {
		diags = append(diags, metaDropped(p.ID(), rest))
	}
	if req.MaxOutputTokens != nil {
		body["max_output_tokens"] = *req.MaxOutputTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if req.Effort != "" || (req.Thinking != nil && req.Thinking.Enabled) {
		reasoning := map[string]any{}
		for k, v := range wirecore.NestedFrom(req.Extra, "reasoning") {
			reasoning[k] = v
		}
		if req.Effort != "" {
			reasoning["effort"] = string(req.Effort)
		}
		if _, named := reasoning["summary"]; !named {
			reasoning["summary"] = "auto"
		}
		body["reasoning"] = reasoning
	}
	text := map[string]any{}
	for k, v := range wirecore.NestedFrom(req.Extra, "text") {
		text[k] = v
	}
	if req.ResponseFormat != nil {
		format := map[string]any{"type": req.ResponseFormat.Type}
		if len(req.ResponseFormat.Schema) > 0 {
			format = map[string]any{
				"type": "json_schema", "name": nameOr(req.ResponseFormat.Name, "response"),
				"strict": req.ResponseFormat.Strict, "schema": req.ResponseFormat.Schema,
			}
		}
		text["format"] = format
	}
	if len(text) > 0 {
		body["text"] = text
	}
	tools := a.Tools
	if len(tools) == 0 {
		tools = req.Tools
	}
	for _, t := range tools {
		schema := t.Schema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		fn := map[string]any{"type": "function", "name": t.Name, "parameters": schema}
		if t.Strict {
			fn["strict"] = true
		}
		if t.Description != "" {
			fn["description"] = t.Description
		}
		body["tools"] = append(bodyTools(body), fn)
	}
	for _, server := range req.MCPServers {
		if server.Enabled != nil && !*server.Enabled {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropMCPServer, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "enabled",
				Detail: "this backend cannot carry a tool server it is told not to use, so the server was left out",
			})
			continue
		}
		if len(server.DeniedTools) > 0 {
			return nil, diags, llm.Errorf(llm.KindUnsupported,
				"tool server %q refuses individual tools, which this backend cannot express; name the tools to allow instead",
				server.Name)
		}
		if server.DeferLoading != nil {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropMCPServer, Class: llm.ClassCapability, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "defer_loading",
				Detail: "this backend decides for itself when to load a tool server's descriptions, so the request to hold them back was not sent",
			})
		}
		body["tools"] = append(bodyTools(body), openairesp.EncodeMCPTool(server))
	}
	if req.ToolChoice != nil {
		if tc := responsesToolChoice(req.ToolChoice); tc != nil {
			body["tool_choice"] = tc
		}
	}
	if a.Session != "" {
		body["previous_response_id"] = a.Session
	}
	if stream {
		body["stream"] = true
	}
	diags = append(diags, p.extend(body, req)...)
	diags = append(diags, llm.UnsentSampling(req, p.ID(),
		llm.FieldTemperature, llm.FieldTopP, llm.FieldMaxOutputTokens)...)
	for _, x := range diags {
		a.Note(x)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode request: %v", err)
	}
	return raw, diags, nil
}

func responsesToolChoice(tc *llm.ToolChoice) any {
	switch tc.Mode {
	case llm.ToolAuto:
		return "auto"
	case llm.ToolNone:
		return "none"
	case llm.ToolRequired:
		return "required"
	case llm.ToolSpecific:
		if tc.Name == "" {
			return nil
		}
		return map[string]any{"type": "function", "name": tc.Name}
	default:
		return nil
	}
}

func bodyTools(body map[string]any) []map[string]any {
	if v, ok := body["tools"].([]map[string]any); ok {
		return v
	}
	return nil
}

func responsesInput(msgs []llm.Message, provider string) ([]map[string]any, llm.Diagnostics) {
	var diags llm.Diagnostics
	var out []map[string]any
	for _, m := range msgs {
		role := string(m.Role)
		if m.Name != "" {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: provider, Field: "name",
				Detail: "this backend has no field to name a participant, so the name was not sent",
			})
		}
		before := len(out)
		var content []map[string]any
		for _, part := range m.Content {
			switch v := part.(type) {
			case *llm.Text:
				kind := "input_text"
				if m.Role == llm.RoleAssistant {
					kind = "output_text"
				}
				content = append(content, map[string]any{"type": kind, "text": v.Text})
			case *llm.Refusal:
				content = append(content, map[string]any{"type": "refusal", "refusal": v.Text})
			case *llm.Image:
				url := v.URL
				if len(v.Data) > 0 {
					url = "data:" + nameOr(v.MIME, "image/png") + ";base64," + base64.StdEncoding.EncodeToString(v.Data)
				}
				content = append(content, map[string]any{"type": "input_image", "image_url": url})
			case *llm.ToolUse:
				if len(content) > 0 {
					out = append(out, map[string]any{"type": "message", "role": role, "content": content})
					content = nil
				}
				args := string(v.Args)
				if args == "" {
					args = "{}"
				}
				out = append(out, map[string]any{"type": "function_call", "call_id": v.ID, "name": v.Name, "arguments": args})
			case *llm.ToolResult:
				if len(content) > 0 {
					out = append(out, map[string]any{"type": "message", "role": role, "content": content})
					content = nil
				}
				out = append(out, map[string]any{"type": "function_call_output", "call_id": v.ID, "output": llm.TextOf(v.Content)})
				diags = append(diags, llm.Flattened(v.Content, provider, toolResultText)...)
			case *llm.Thinking:
				if len(content) > 0 {
					out = append(out, map[string]any{"type": "message", "role": role, "content": content})
					content = nil
				}
				item := map[string]any{"type": "reasoning", "summary": []map[string]any{}}
				if v.Text != "" {
					item["summary"] = []map[string]any{{"type": "summary_text", "text": v.Text}}
				}
				if v.Signature != "" {
					item["encrypted_content"] = v.Signature
				}
				out = append(out, item)
			case *llm.Raw:
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityLoss, Field: v.Type, Detail: "content block has no shape on this endpoint",
				})
			}
		}
		switch {
		case len(content) > 0:
			out = append(out, map[string]any{"type": "message", "role": role, "content": content})
		case len(out) == before:
			diags = append(diags, llm.DroppedMessage(m.Role, provider,
				"this turn reached no item on this endpoint, so the conversation arrives a turn shorter"))
		}
	}
	return out, diags
}

type responsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	InputTokensDetails struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokens        int `json:"output_tokens"`
	OutputTokensDetails struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
	TotalTokens int `json:"total_tokens"`
}

func (u *responsesUsage) neutral() llm.Usage {
	return llm.RawUsage{
		Prompt: u.InputTokens, Completion: u.OutputTokens,
		CacheRead: u.InputTokensDetails.CachedTokens,
		Reasoning: u.OutputTokensDetails.ReasoningTokens, Reported: true,
	}.Normalise()
}

type responsesItem struct {
	Type    string `json:"type"`
	ID      string `json:"id"`
	Role    string `json:"role"`
	Content []struct {
		Type    string `json:"type"`
		Text    string `json:"text"`
		Refusal string `json:"refusal"`
	} `json:"content"`
	Summary []struct {
		Text string `json:"text"`
	} `json:"summary"`
	EncryptedContent string `json:"encrypted_content"`
	CallID           string `json:"call_id"`
	Name             string `json:"name"`
	Arguments        string `json:"arguments"`
	Result           string `json:"result"`
	OutputFormat     string `json:"output_format"`
	RevisedPrompt    string `json:"revised_prompt"`

	raw json.RawMessage
}

func (it *responsesItem) UnmarshalJSON(b []byte) error {
	type alias responsesItem
	var a alias
	if err := json.Unmarshal(b, &a); err != nil {
		return err
	}
	*it = responsesItem(a)
	it.raw = append(json.RawMessage(nil), b...)
	return nil
}

func imagePart(it responsesItem, index *int) *llm.Image {
	data, err := base64.StdEncoding.DecodeString(it.Result)
	if err != nil {
		return nil
	}
	idx := *index
	*index++
	format := it.OutputFormat
	if format == "" {
		format = "png"
	}
	return &llm.Image{
		ID: it.ID, MIME: "image/" + format, Data: data,
		Prompt: it.RevisedPrompt, Index: &idx,
	}
}

var responsesKnown = map[string]bool{
	"id": true, "model": true, "status": true, "output": true,
	"usage": true, "incomplete_details": true, "error": true,
	"created_at": true, "service_tier": true,
}

type responsesBodyOut struct {
	ID                string          `json:"id"`
	Model             string          `json:"model"`
	CreatedAt         int64           `json:"created_at"`
	ServiceTier       string          `json:"service_tier"`
	Status            string          `json:"status"`
	Output            []responsesItem `json:"output"`
	Usage             *responsesUsage `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

func itemParts(items []responsesItem, toolIndex *int) []llm.Part {
	var out []llm.Part
	imageIndex := new(int)
	for _, it := range items {
		switch it.Type {
		case "reasoning":
			th := &llm.Thinking{Signature: it.EncryptedContent}
			for _, s := range it.Summary {
				th.Text += s.Text
			}
			out = append(out, th)
		case "message":
			for _, c := range it.Content {
				switch c.Type {
				case "output_text", "text":
					if c.Text != "" {
						out = append(out, &llm.Text{Text: c.Text})
					}
				case "refusal":
					if c.Refusal != "" {
						out = append(out, &llm.Refusal{Text: c.Refusal})
					}
				}
			}
		case "function_call":
			idx := *toolIndex
			*toolIndex++
			args := it.Arguments
			if args == "" {
				args = "{}"
			}
			out = append(out, &llm.ToolUse{ID: it.CallID, Name: it.Name, Args: json.RawMessage(args), Index: &idx})
		case "image_generation_call":
			if img := imagePart(it, imageIndex); img != nil {
				out = append(out, img)
			}
		default:
			if len(it.raw) > 0 {
				out = append(out, &llm.Raw{Vendor: openairesp.Name, Type: it.Type, JSON: it.raw})
			}
		}
	}
	return out
}

func stopFor(status string, reason string, hasTool bool) llm.StopReason {
	switch status {
	case "incomplete":
		switch reason {
		case "max_output_tokens":
			return llm.StopLength
		case "content_filter":
			return llm.StopContentFilter
		default:
			return llm.StopPause
		}
	case "failed":
		return llm.StopError
	default:
		if hasTool {
			return llm.StopToolUse
		}
		return llm.StopEnd
	}
}

func (p *Provider) completeResponses(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	body, diags, err := p.responsesBody(a, false)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.endpoint("/responses", p.opts.ResponsesPath), Body: body, Header: p.header(a), Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	raw, rerr := io.ReadAll(resp.Body)
	if rerr != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "response body: " + rerr.Error(), Err: rerr}
	}
	var out responsesBodyOut
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "malformed response: " + err.Error(), Err: err}
	}
	if out.Error != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Status: 200, Code: out.Error.Code, Message: out.Error.Message}
	}
	idx := 0
	parts := itemParts(out.Output, &idx)
	reason := ""
	if out.IncompleteDetails != nil {
		reason = out.IncompleteDetails.Reason
	}
	res := &llm.Response{
		ID: out.ID, Provider: p.ID(), Model: out.Model, Requested: a.Request.Model,
		Created: created(out.CreatedAt, p.now()), Session: out.ID, Diagnostics: diags,
		Extra: wirecore.Extras(raw, responsesKnown, a, openairesp.Name),
		Choices: []llm.Choice{{
			Index:   0,
			Message: llm.Message{Role: llm.RoleAssistant, Content: parts},
			Stop:    stopFor(out.Status, reason, llm.HasKind(parts, llm.KindToolUse)),
		}},
	}
	if out.Usage != nil {
		u := out.Usage.neutral()
		u.Tier = out.ServiceTier
		res.Usage = u
	}
	return res, nil
}

func (p *Provider) streamResponses(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	body, _, err := p.responsesBody(a, true)
	if err != nil {
		return nil, err
	}
	h := p.header(a)
	h.Set("Accept", "text/event-stream")
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.endpoint("/responses", p.opts.ResponsesPath), Body: body, Header: h, Sign: p.opts.Sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	sc := wirecore.NewScanner(resp.Body, a.Limits.MaxFrameBytes, a.Limits.MaxBufferBytes)
	st := &responsesParser{provider: p.ID(), sc: sc, attempt: a}
	return llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now()},
		st.next,
		func() error { return resp.Body.Close() },
	), nil
}

type responsesParser struct {
	provider    string
	sc          *wirecore.Scanner
	done        bool
	toolIdx     int
	imageIdx    int
	attempt     *llm.Attempt
	textSent    int
	refusalSent int
}

func (r *responsesParser) tail(it responsesItem) *llm.Frame {
	var text, refusal string
	for _, c := range it.Content {
		switch c.Type {
		case "output_text", "text":
			text += c.Text
		case "refusal":
			refusal += c.Refusal
		}
	}
	var parts []llm.Part
	if len(text) > r.textSent {
		parts = append(parts, &llm.Text{Text: text[r.textSent:]})
	}
	if len(refusal) > r.refusalSent {
		parts = append(parts, &llm.Refusal{Text: refusal[r.refusalSent:]})
	}
	r.textSent, r.refusalSent = len(text), len(refusal)
	if len(parts) == 0 {
		return nil
	}
	return &llm.Frame{Delta: parts}
}

func (r *responsesParser) next() (*llm.Frame, error) {
	for {
		if r.done {
			return nil, nil
		}
		if !r.sc.Scan() {
			if err := r.sc.Err(); err != nil {
				return nil, &llm.Error{Kind: llm.KindTransport, Provider: r.provider, Sent: true, Message: err.Error(), Err: err}
			}
			r.done = true
			return nil, nil
		}
		ev := r.sc.Event()
		data := bytes.TrimSpace(ev.Data)
		if len(data) == 0 || string(data) == "[DONE]" {
			continue
		}
		var env struct {
			Type         string          `json:"type"`
			Delta        string          `json:"delta"`
			ItemID       string          `json:"item_id"`
			Item         *responsesItem  `json:"item"`
			Response     json.RawMessage `json:"response"`
			Code         string          `json:"code"`
			Message      string          `json:"message"`
			PartialImage string          `json:"partial_image_b64"`
			PartialIndex int             `json:"partial_image_index"`
			OutputFormat string          `json:"output_format"`
		}
		if err := json.Unmarshal(data, &env); err != nil {
			return nil, &llm.Error{
				Kind: llm.KindUpstream, Provider: r.provider, Sent: true, Code: llm.WireFrameMalformed,
				Message: "malformed stream frame: " + err.Error(), Err: err,
			}
		}
		name := env.Type
		if name == "" {
			name = ev.Name
		}
		switch name {
		case "response.created", "response.in_progress":
			var out responsesBodyOut
			if len(env.Response) > 0 {
				_ = json.Unmarshal(env.Response, &out)
			}
			if out.ID == "" {
				continue
			}
			return &llm.Frame{
				ID: out.ID, Model: out.Model, Session: out.ID,
				Extra: wirecore.Extras(env.Response, responsesKnown, r.attempt, openairesp.Name),
			}, nil
		case "response.output_text.delta":
			r.textSent += len(env.Delta)
			return &llm.Frame{Delta: []llm.Part{&llm.Text{Text: env.Delta}}}, nil
		case "response.refusal.delta":
			r.refusalSent += len(env.Delta)
			return &llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: env.Delta}}}, nil
		case "response.reasoning_summary_text.delta":
			return &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: env.Delta}}}, nil
		case "response.output_item.added":
			if env.Item != nil && env.Item.Type == "message" {
				r.textSent, r.refusalSent = 0, 0
			}
			if env.Item != nil && env.Item.Type == "function_call" {
				idx := r.toolIdx
				r.toolIdx++
				return &llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: env.Item.CallID, Name: env.Item.Name, Index: &idx}}}, nil
			}
		case "response.image_generation_call.partial_image":
			data, err := base64.StdEncoding.DecodeString(env.PartialImage)
			if err != nil {
				continue
			}
			idx := env.PartialIndex
			return &llm.Frame{Delta: []llm.Part{&llm.Image{
				ID: env.ItemID, MIME: "image/" + nameOr(env.OutputFormat, "png"),
				Data: data, Index: &idx, Partial: true,
			}}}, nil
		case "response.output_item.done":
			if env.Item == nil {
				continue
			}
			switch env.Item.Type {
			case "image_generation_call":
				if img := imagePart(*env.Item, &r.imageIdx); img != nil {
					return &llm.Frame{Delta: []llm.Part{img}}, nil
				}
			case "reasoning":
				if env.Item.EncryptedContent != "" {
					return &llm.Frame{Delta: []llm.Part{
						&llm.Thinking{Signature: env.Item.EncryptedContent},
					}}, nil
				}
			case "message":
				if f := r.tail(*env.Item); f != nil {
					return f, nil
				}
			case "function_call":
			default:
				if len(env.Item.raw) > 0 {
					return &llm.Frame{Delta: []llm.Part{&llm.Raw{
						Vendor: openairesp.Name, Type: env.Item.Type, JSON: env.Item.raw,
					}}}, nil
				}
			}
		case "response.function_call_arguments.delta":
			idx := r.toolIdx - 1
			if idx < 0 {
				idx = 0
			}
			return &llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: &idx, Args: json.RawMessage(env.Delta)}}}, nil
		case "response.completed", "response.incomplete", "response.failed":
			var out responsesBodyOut
			if len(env.Response) > 0 {
				_ = json.Unmarshal(env.Response, &out)
			}
			r.done = true
			frame := &llm.Frame{
				ID: out.ID, Model: out.Model, Session: out.ID, TurnEnd: true,
				Extra: wirecore.Extras(env.Response, responsesKnown, r.attempt, openairesp.Name),
			}
			reason := ""
			if out.IncompleteDetails != nil {
				reason = out.IncompleteDetails.Reason
			}
			frame.Stop = stopFor(strings.TrimPrefix(name, "response."), reason, r.toolIdx > 0)
			if out.Usage != nil {
				u := out.Usage.neutral()
				u.Tier = out.ServiceTier
				frame.Usage = &u
			}
			if name == "response.failed" && out.Error != nil {
				frame.Err = &llm.Error{Kind: llm.KindUpstream, Provider: r.provider, Sent: true, Code: out.Error.Code, Message: out.Error.Message}
			}
			return frame, nil
		case "error":
			r.done = true
			return &llm.Frame{Err: &llm.Error{
				Kind: llm.KindUpstream, Provider: r.provider, Sent: true,
				Code: env.Code, Message: env.Message,
			}}, nil
		}
	}
}
