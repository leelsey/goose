// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package bedrock talks to AWS Bedrock through the Converse API.
package bedrock

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/leelsey/goose/internal/cloudauth"
	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

const maxBreakpoints = 4

type Credentials struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

type Options struct {
	ID          string
	Region      string
	Model       string
	BaseURL     string
	Credentials CredentialSource
	MaxRetries  int
	Timeout     time.Duration
	Header      http.Header
	Now         func() time.Time
}

type Provider struct {
	opts Options
	env  llm.Env
	call *httpcore.Caller
}

func New(o Options) (*Provider, error) {
	if o.ID == "" {
		o.ID = "bedrock"
	}
	if o.Region == "" {
		return nil, llm.Errorf(llm.KindInvalidRequest, "bedrock: region is required")
	}
	// The region is spliced into the host the call is signed for and sent to,
	// so a character no region's name has could move the call to another host.
	if strings.Trim(o.Region, "abcdefghijklmnopqrstuvwxyz0123456789-") != "" {
		return nil, llm.Errorf(llm.KindInvalidRequest, "bedrock: region %q is not a region's name", o.Region)
	}
	if o.Credentials == nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "bedrock: credentials are required")
	}
	if o.BaseURL == "" {
		o.BaseURL = "https://bedrock-runtime." + o.Region + ".amazonaws.com"
	}
	p := &Provider{opts: o}
	p.call = &httpcore.Caller{Provider: o.ID, MaxRetries: o.MaxRetries, HTTP: httpcore.NewClient(o.Timeout), Refine: refine}
	return p, nil
}

// refine reads what the status does not say. Converse refuses a model it does
// not know, and one it serves only through an inference profile, with the 400
// a malformed request gets, and only a malformed request fails the same on
// every backend. It refuses a model the account has not been granted, or
// cannot subscribe to, with the 403 a credential without rights gets, though
// every other model on the account may still be served.
func refine(e *llm.Error) {
	var unserved []string
	switch e.Kind {
	case llm.KindInvalidRequest:
		unserved = []string{"model identifier is invalid", "on-demand throughput", "inference profile"}
	case llm.KindPermission:
		unserved = []string{"access to the model with the specified model id", "to enable access to this model"}
	}
	said := strings.ToLower(e.Message)
	for _, s := range unserved {
		if strings.Contains(said, s) {
			e.Kind = llm.KindNotFound
			return
		}
	}
}

func (p *Provider) ID() string { return p.opts.ID }

func (p *Provider) Bind(env llm.Env) error {
	p.env = env
	if env.HTTP != nil {
		p.call.HTTP = env.HTTP
	}
	return nil
}

func (p *Provider) Capabilities() llm.Capabilities {
	return llm.Capabilities{
		Streaming:          true,
		Usage:              llm.UsageShape{StreamUsage: true},
		SystemPrompt:       true,
		Tools:              true,
		ParallelTools:      true,
		ToolChoice:         true,
		ToolResults:        true,
		ToolErrors:         true,
		ImageIn:            true,
		Thinking:           true,
		ThinkingSignatures: true,
		Signer:             llm.FamilyAnthropic,
		RedactedThinking:   true,
		Candidates:         1,
		CacheBreakpoints:   maxBreakpoints,
		Ceiling:            llm.CeilingMaxTokens,
	}
}

func (p *Provider) now() time.Time {
	switch {
	case p.opts.Now != nil:
		return p.opts.Now()
	case p.env.Now != nil:
		return p.env.Now()
	default:
		return time.Now()
	}
}

func (p *Provider) model(a *llm.Attempt) string {
	switch {
	case a.Model != "":
		return a.Model
	case p.opts.Model != "":
		return p.opts.Model
	case a.Request != nil:
		return a.Request.Model
	default:
		return ""
	}
}

func (p *Provider) url(a *llm.Attempt, action string) string {
	return httpcore.JoinURL(p.opts.BaseURL, "/model/"+url.PathEscape(p.model(a))+"/"+action)
}

func (p *Provider) sign(ctx context.Context, r *http.Request, body []byte) error {
	creds, err := p.opts.Credentials(ctx)
	if err != nil {
		return err
	}
	return cloudauth.SignV4(r, body, cloudauth.Creds{
		AccessKeyID:     creds.AccessKeyID,
		SecretAccessKey: creds.SecretAccessKey,
		SessionToken:    creds.SessionToken,
	}, p.opts.Region, "bedrock", p.now())
}

func (p *Provider) header() http.Header {
	h := http.Header{}
	for k, vs := range p.opts.Header {
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	h.Set("Content-Type", "application/json")
	return h
}

func (p *Provider) body(a *llm.Attempt) ([]byte, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	req := a.Request
	body := map[string]any{}

	tools := a.Tools
	if len(tools) == 0 {
		tools = req.Tools
	}
	cache := p.caching(a, tools, &diags)

	system, msgs := req.SystemAndMessages()
	if len(system) > 0 {
		blocks, d := contentBlocks(system, &diags)
		diags = append(diags, d...)
		if req.SystemCache != nil && cache.take() {
			blocks = append(blocks, cachePoint())
		}
		body["system"] = blocks
	}

	var turns []map[string]any
	if len(req.Metadata) > 0 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "metadata",
			Detail: "this backend has no field to name the end user, so the name was not sent",
		})
	}
	for _, m := range msgs {
		role := "user"
		if m.Role == llm.RoleAssistant {
			role = "assistant"
		}
		if m.Name != "" {
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageRequest,
				Severity: llm.SeverityInfo, Provider: p.ID(), Field: "name",
				Detail: "this backend has no field to name a participant, so the name was not sent",
			})
		}
		blocks, d := contentBlocks(m.Content, &diags)
		diags = append(diags, d...)
		if len(blocks) == 0 {
			diags = append(diags, llm.DroppedMessage(m.Role, p.ID(),
				"this turn reached no block on this endpoint, so the conversation arrives a turn shorter"))
			continue
		}
		if m.Cache != nil && cache.take() {
			blocks = append(blocks, cachePoint())
		}
		if n := len(turns); n > 0 && turns[n-1]["role"] == role {
			turns[n-1]["content"] = append(turns[n-1]["content"].([]map[string]any), blocks...)
			continue
		}
		turns = append(turns, map[string]any{"role": role, "content": blocks})
	}
	if len(turns) == 0 {
		return nil, diags, llm.Errorf(llm.KindInvalidRequest, "no messages survived translation")
	}
	body["messages"] = turns

	thinking, ceiling, d := p.thinking(req.Thinking, req.MaxOutputTokens,
		len(tools) > 0 && forced(req.ToolChoice), unsafeToResume(turns))
	diags = append(diags, d...)
	if thinking != nil {
		body["additionalModelRequestFields"] = map[string]any{"thinking": thinking}
	}
	temperature, topP := req.Temperature, req.TopP
	if thinking != nil {
		temperature, topP, d = p.besideThinking(temperature, topP)
		diags = append(diags, d...)
	}

	cfg := map[string]any{}
	if ceiling != nil {
		cfg["maxTokens"] = *ceiling
	}
	if temperature != nil {
		cfg["temperature"] = *temperature
	}
	if topP != nil {
		cfg["topP"] = *topP
	}
	if len(req.Stop) > 0 {
		cfg["stopSequences"] = req.Stop
	}
	if len(cfg) > 0 {
		body["inferenceConfig"] = cfg
	}

	if len(tools) > 0 {
		specs := make([]map[string]any, 0, len(tools))
		for _, t := range tools {
			schema := t.Schema
			if len(schema) == 0 {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			spec := map[string]any{"name": t.Name, "inputSchema": map[string]any{"json": schema}}
			if t.Description != "" {
				spec["description"] = t.Description
			}
			specs = append(specs, map[string]any{"toolSpec": spec})
			if t.Cache != nil && cache.take() {
				specs = append(specs, cachePoint())
			}
		}
		cfg := map[string]any{"tools": specs}
		if req.ToolChoice != nil {
			if tc := toolChoice(req.ToolChoice); tc != nil {
				cfg["toolChoice"] = tc
			} else {
				diags = append(diags, llm.Diagnostic{
					Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StageRequest,
					Severity: llm.SeverityLoss, Provider: p.ID(), Field: "tool_choice",
					Detail: "this backend cannot be told not to call a tool, so the model may still call one",
				})
			}
		}
		body["toolConfig"] = cfg
		if llm.StrictTools(tools) {
			diags = append(diags, llm.ToolStrictness(p.ID()))
		}
	} else if names := historyTools(msgs); len(names) > 0 {
		// Converse refuses toolUse and toolResult blocks without a toolConfig,
		// and a client may send a tool loop's history with no tools, to close
		// it with a summary for instance. The tools the history names are
		// declared again, with no schema, so the conversation travels as it was.
		// Converse cannot be told that none may be called, so the description
		// asks instead.
		specs := make([]map[string]any, 0, len(names))
		for _, name := range names {
			specs = append(specs, map[string]any{"toolSpec": map[string]any{
				"name":        name,
				"description": "Not offered on this turn: declared only because earlier turns use it.",
				"inputSchema": map[string]any{"json": json.RawMessage(`{"type":"object","properties":{}}`)},
			}})
		}
		body["toolConfig"] = map[string]any{"tools": specs}
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityLoss, Provider: p.ID(), Field: "tools", Count: len(names),
			Detail: "this backend refuses a tool history without the tools it names, so they were declared again and the model may call one the request did not offer",
		})
	}

	diags = append(diags, llm.UnsentSampling(req, p.ID(),
		llm.FieldTemperature, llm.FieldTopP, llm.FieldStop, llm.FieldMaxOutputTokens)...)

	for _, x := range diags {
		a.Note(x)
	}
	if err := a.Check(); err != nil {
		return nil, diags, err
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode request: %v", err)
	}
	return raw, diags, nil
}

const (
	defaultThinkingBudget = 2048
	minThinkingBudget     = 1024
	minThinkingTopP       = 0.95
	// thinkingCeiling and answerRoom are the anthropic provider's: the least
	// any Claude that thinks writes in one answer, and the room an answer has
	// after a budget where the caller set no ceiling.
	thinkingCeiling = 32000
	answerRoom      = 4096
)

// thinking is the thinking Claude takes here for what was asked, as on
// Anthropic's own API: at least 1024 tokens, fewer than the output ceiling,
// none before a tool call it is made to make, and none when the turn it would
// resume into does not lead with the thinking Claude requires of it. A request
// outside that is refused whole. The ceiling returned is the one to send.
func (p *Provider) thinking(th *llm.ThinkingConfig, ceiling *int, forced, unsafeResume bool) (map[string]any, *int, llm.Diagnostics) {
	if th == nil || !th.Enabled {
		return nil, ceiling, nil
	}
	off := func(detail string) (map[string]any, *int, llm.Diagnostics) {
		return nil, ceiling, llm.Diagnostics{{
			Code: llm.DropThinking, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityLoss, Provider: p.ID(), Field: "thinking",
			Detail: detail,
		}}
	}
	if forced {
		return off("this backend cannot think before a tool call it is made to make, so thinking was not enabled")
	}
	if unsafeResume {
		return off("the turn whose tool call this request answers does not lead with thinking this backend can verify, which it requires of that turn once thinking is on, so thinking was not enabled")
	}
	// The models a caller names the adaptive mode for refuse a budget of any
	// size, so the mode is sent as it was named.
	if th.Adaptive {
		return map[string]any{"type": "adaptive"}, ceiling, nil
	}
	var diags llm.Diagnostics
	most := math.MaxInt
	switch {
	case ceiling != nil:
		most = *ceiling - 1
	case th.Budget > 0:
		// With no ceiling sent, Converse gives the model the largest it has,
		// which a budget can pass and is then refused whole. One is sent as on
		// Anthropic's own API, so the budget is kept as far as any Claude that
		// thinks can write and the answer after it has the same room.
		most = min(max(th.Budget, minThinkingBudget), thinkingCeiling-answerRoom)
		ceiling = llm.Ptr(most + answerRoom)
		diags = append(diags, llm.Diagnostic{
			Code: llm.CapCeilingSupplied, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "maxTokens",
			Detail: "this backend thinks for fewer tokens than the output ceiling, so one was sent to fit the budget, and the answer stops at the default after it rather than where the model would have",
		})
	}
	if most < minThinkingBudget {
		return off("this backend thinks for at least 1024 tokens and for fewer than the output ceiling, which this ceiling leaves no room between, so thinking was not enabled")
	}
	budget := th.Budget
	switch {
	case budget <= 0:
		budget = defaultThinkingBudget
		if ceiling != nil {
			budget = min(budget, max(*ceiling/2, minThinkingBudget))
		}
	case budget < minThinkingBudget || budget > most:
		budget = min(max(budget, minThinkingBudget), most)
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "thinking.budget",
			Detail: "this backend thinks for at least 1024 tokens and for fewer than the output ceiling, so the budget was moved to the nearest it takes",
		})
	}
	return map[string]any{"type": "enabled", "budget_tokens": budget}, ceiling, diags
}

// unsafeToResume reports whether turns end with a tool result answering an
// assistant turn that does not lead with reasoning. Claude requires that turn
// to start with its thinking whenever thinking is enabled, however the turn
// came to lose it: dropped here unsigned, stripped before it arrived, or never
// there.
func unsafeToResume(turns []map[string]any) bool {
	n := len(turns)
	if n < 2 || turns[n-1]["role"] != "user" || turns[n-2]["role"] != "assistant" {
		return false
	}
	answers := slices.ContainsFunc(turns[n-1]["content"].([]map[string]any), func(b map[string]any) bool {
		_, ok := b["toolResult"]
		return ok
	})
	if !answers {
		return false
	}
	lead := turns[n-2]["content"].([]map[string]any)
	if len(lead) == 0 {
		return true
	}
	_, ok := lead[0]["reasoningContent"]
	return !ok
}

// besideThinking is the sampling Claude takes here while it thinks, as on
// Anthropic's own API: no temperature but 1 and no top_p under 0.95. Anything
// else beside thinking is refused whole.
func (p *Provider) besideThinking(temperature, topP *float64) (*float64, *float64, llm.Diagnostics) {
	var diags llm.Diagnostics
	unsent := func(field, detail string) {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: field,
			Detail: detail,
		})
	}
	if temperature != nil && *temperature != 1 {
		temperature = nil
		unsent(llm.FieldTemperature, "this backend samples at its own temperature while it thinks, so the one asked for was not sent")
	}
	if topP != nil && *topP < minThinkingTopP {
		topP = llm.Ptr(minThinkingTopP)
		unsent(llm.FieldTopP, "this backend takes a top_p of at least 0.95 while it thinks, so 0.95 was sent")
	}
	return temperature, topP, diags
}

// forced says a tool choice leaves the model no answer but a tool call.
func forced(tc *llm.ToolChoice) bool {
	return tc != nil && (tc.Mode == llm.ToolRequired || tc.Mode == llm.ToolSpecific)
}

func historyTools(msgs []llm.Message) []string {
	var names []string
	for _, m := range msgs {
		for _, part := range m.Content {
			var name string
			switch v := part.(type) {
			case *llm.ToolUse:
				name = v.Name
			case *llm.ToolResult:
				name = v.Name
			}
			if name != "" && !slices.Contains(names, name) {
				names = append(names, name)
			}
		}
	}
	return names
}

func toolChoice(tc *llm.ToolChoice) map[string]any {
	switch tc.Mode {
	case llm.ToolAuto:
		return map[string]any{"auto": map[string]any{}}
	case llm.ToolRequired:
		return map[string]any{"any": map[string]any{}}
	case llm.ToolSpecific:
		return map[string]any{"tool": map[string]any{"name": tc.Name}}
	default:
		return nil
	}
}

func contentBlocks(parts []llm.Part, diags *llm.Diagnostics) ([]map[string]any, llm.Diagnostics) {
	var out []map[string]any
	var local llm.Diagnostics
	for _, p := range parts {
		switch v := p.(type) {
		case *llm.Text:
			if v.Text != "" {
				out = append(out, map[string]any{"text": v.Text})
			}
		case *llm.Refusal:
			if v.Text != "" {
				out = append(out, map[string]any{"text": v.Text})
			}
		case *llm.Thinking:
			if len(v.Redacted) > 0 {
				out = append(out, map[string]any{"reasoningContent": map[string]any{
					"redactedContent": string(v.Redacted),
				}})
				continue
			}
			// A block stripped of a foreign signature, or one that never had
			// Claude's own, cannot be replayed as this backend's own reasoning:
			// sent with an empty signature it is indistinguishable from one
			// Claude never verified, which Claude refuses the same as anthropic.
			if v.Signature == "" {
				local = append(local, llm.Diagnostic{
					Code: llm.DropSignature, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityInfo,
					Detail:   "thinking without its signature cannot be replayed to this backend",
				})
				continue
			}
			out = append(out, map[string]any{"reasoningContent": map[string]any{
				"reasoningText": map[string]any{"text": v.Text, "signature": v.Signature},
			}})
		case *llm.ToolUse:
			input := v.Args
			if len(input) == 0 || !json.Valid(input) {
				input = json.RawMessage(`{}`)
			}
			out = append(out, map[string]any{"toolUse": map[string]any{
				"toolUseId": v.ID, "name": v.Name, "input": input,
			}})
		case *llm.ToolResult:
			inner, d := contentBlocks(v.Content, diags)
			local = append(local, d...)
			// Converse refuses a tool result with nothing in it, as null or as an
			// empty list, and a tool that printed nothing is routine. The result
			// says so in words rather than failing the whole turn.
			if len(inner) == 0 {
				inner = []map[string]any{{"text": "(no content)"}}
			}
			status := "success"
			if v.IsError {
				status = "error"
			}
			out = append(out, map[string]any{"toolResult": map[string]any{
				"toolUseId": v.ID, "content": inner, "status": status,
			}})
		case *llm.Image:
			if len(v.Data) == 0 {
				local = append(local, llm.Diagnostic{
					Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageRequest,
					Severity: llm.SeverityLoss, Detail: "this backend takes image bytes, not a URL",
				})
				continue
			}
			format := strings.TrimPrefix(v.MIME, "image/")
			if format == "" {
				format = "png"
			}
			out = append(out, map[string]any{"image": map[string]any{
				"format": format,
				"source": map[string]any{"bytes": base64.StdEncoding.EncodeToString(v.Data)},
			}})
		case *llm.Raw:
			local = append(local, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageRequest,
				Severity: llm.SeverityLoss, Field: v.Type, Detail: "content block has no shape in the Converse API",
			})
		}
	}
	return out, local
}

type converseResponse struct {
	Output struct {
		Message struct {
			Role    string            `json:"role"`
			Content []json.RawMessage `json:"content"`
		} `json:"message"`
	} `json:"output"`
	StopReason string `json:"stopReason"`
	Usage      struct {
		InputTokens           int `json:"inputTokens"`
		OutputTokens          int `json:"outputTokens"`
		TotalTokens           int `json:"totalTokens"`
		CacheReadInputTokens  int `json:"cacheReadInputTokens"`
		CacheWriteInputTokens int `json:"cacheWriteInputTokens"`
	} `json:"usage"`
	Metrics struct {
		LatencyMS int64 `json:"latencyMs"`
	} `json:"metrics"`
	Message string `json:"message"`
	Type    string `json:"__type"`
}

func decodeUsage(in, out, cacheRead, cacheWrite int, latency int64) llm.Usage {
	u := llm.RawUsage{
		Prompt: in, Completion: out, CacheRead: cacheRead, CacheWrite: cacheWrite,
		CacheOutsidePrompt: true, Reported: true,
	}.Normalise()
	u.DurationMS = latency
	return u
}

func decodeStop(s string) llm.StopReason {
	switch s {
	case "end_turn":
		return llm.StopEnd
	case "max_tokens":
		return llm.StopLength
	case "stop_sequence":
		return llm.StopSequence
	case "tool_use":
		return llm.StopToolUse
	case "content_filtered", "guardrail_intervened":
		return llm.StopContentFilter
	case "":
		return llm.StopNone
	default:
		return llm.StopUnknown
	}
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	body, diags, err := p.body(a)
	if err != nil {
		return nil, err
	}
	resp, err := p.call.Do(ctx, httpcore.Request{
		URL: p.url(a, "converse"), Body: body, Header: p.header(), Sign: p.sign, OnDiscard: httpcore.BookDiscards(a), Limits: a.Limits,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out converseResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, &llm.Error{Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Message: "malformed response: " + err.Error(), Err: err}
	}
	if out.Message != "" && len(out.Output.Message.Content) == 0 {
		return nil, &llm.Error{
			Kind: llm.KindUpstream, Provider: p.ID(), Sent: true, Status: 200,
			Code: out.Type, Message: out.Message, Body: httpcore.ErrorBody([]byte(out.Message)),
		}
	}
	parts, d := decodeBlocks(out.Output.Message.Content)
	diags = append(diags, d...)
	return &llm.Response{
		Provider: p.ID(), Model: p.model(a), Requested: a.Request.Model, Created: p.now(),
		Diagnostics: diags,
		Usage: decodeUsage(out.Usage.InputTokens, out.Usage.OutputTokens,
			out.Usage.CacheReadInputTokens, out.Usage.CacheWriteInputTokens, out.Metrics.LatencyMS),
		Choices: []llm.Choice{{
			Index:   0,
			Message: llm.Message{Role: llm.RoleAssistant, Content: parts},
			Stop:    decodeStop(out.StopReason),
		}},
	}, nil
}

func decodeBlocks(blocks []json.RawMessage) ([]llm.Part, llm.Diagnostics) {
	var diags llm.Diagnostics
	var out []llm.Part
	idx := 0
	for _, raw := range blocks {
		var b struct {
			Text    string `json:"text"`
			ToolUse *struct {
				ID    string          `json:"toolUseId"`
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			} `json:"toolUse"`
			Reasoning *struct {
				ReasoningText *struct {
					Text      string `json:"text"`
					Signature string `json:"signature"`
				} `json:"reasoningText"`
				RedactedContent string `json:"redactedContent"`
			} `json:"reasoningContent"`
			Image *struct {
				Format string `json:"format"`
				Source struct {
					Bytes string `json:"bytes"`
				} `json:"source"`
			} `json:"image"`
		}
		if err := json.Unmarshal(raw, &b); err != nil {
			diags = append(diags, llm.Diagnostic{
				Code: llm.WireFrameMalformed, Class: llm.ClassTransport, Stage: llm.StageResponse,
				Severity: llm.SeverityLoss, Detail: "content block could not be read",
			})
			continue
		}
		switch {
		case b.ToolUse != nil:
			i := idx
			idx++
			out = append(out, &llm.ToolUse{ID: b.ToolUse.ID, Name: b.ToolUse.Name, Args: b.ToolUse.Input, Index: &i})
		case b.Reasoning != nil:
			if b.Reasoning.RedactedContent != "" {
				data := []byte(b.Reasoning.RedactedContent)
				out = append(out, &llm.Thinking{Redacted: data})
				continue
			}
			if b.Reasoning.ReasoningText != nil {
				out = append(out, &llm.Thinking{Text: b.Reasoning.ReasoningText.Text,
					Signature: b.Reasoning.ReasoningText.Signature, Signer: llm.FamilyAnthropic})
			}
		case b.Image != nil:
			data, _ := base64.StdEncoding.DecodeString(b.Image.Source.Bytes)
			out = append(out, &llm.Image{MIME: "image/" + b.Image.Format, Data: data})
		case b.Text != "":
			out = append(out, &llm.Text{Text: b.Text})
		default:
			out = append(out, &llm.Raw{Vendor: "bedrock", Type: "contentBlock", JSON: raw})
			diags = append(diags, llm.Diagnostic{
				Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
				Severity: llm.SeverityInfo, Detail: "content block carried verbatim",
			})
		}
	}
	return out, diags
}
