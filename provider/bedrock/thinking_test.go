// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
)

// Claude here thinks for at least 1024 tokens and for fewer than the output
// ceiling, as on Anthropic's own API. A budget outside that is refused whole,
// and the refusal stops the chain. With no ceiling asked for, Converse gives
// the model the largest it has, which a budget can pass, so one is sent as
// the anthropic provider sends one, leaving the answer the same room.
func TestTheThinkingBudgetFitsWhatClaudeTakes(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	for _, c := range []struct {
		budget, ceiling, want int
		moved                 bool
		maxTokens             int
	}{
		{0, 0, 2048, false, 0},
		{0, 8000, 2048, false, 8000},
		{0, 3000, 1500, false, 3000},
		{0, 1500, 1024, false, 1500},
		{0, 1000, 0, false, 1000},
		{-1, 1024, 0, false, 1024},
		{512, 0, 1024, true, 1024 + answerRoom},
		{4000, 0, 4000, false, 4000 + answerRoom},
		{32768, 0, thinkingCeiling - answerRoom, true, thinkingCeiling},
		{200000, 0, thinkingCeiling - answerRoom, true, thinkingCeiling},
		{4096, 2000, 1999, true, 2000},
		{4096, 8000, 4096, false, 8000},
	} {
		req := hello()
		if c.ceiling > 0 {
			req.MaxOutputTokens = llm.Ptr(c.ceiling)
		}
		req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: c.budget}
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		var sent struct {
			Fields struct {
				Thinking *struct {
					Budget int `json:"budget_tokens"`
				} `json:"thinking"`
			} `json:"additionalModelRequestFields"`
			Inference struct {
				MaxTokens int `json:"maxTokens"`
			} `json:"inferenceConfig"`
		}
		body := (*bodies)[len(*bodies)-1]
		if err := json.Unmarshal([]byte(body), &sent); err != nil {
			t.Fatal(err)
		}
		moved := false
		for _, d := range resp.Diagnostics {
			moved = moved || d.Field == "thinking.budget"
		}
		switch th := sent.Fields.Thinking; {
		case c.want == 0 && (th != nil || !resp.Diagnostics.Has(llm.DropThinking)):
			t.Errorf("budget %d, ceiling %d: body = %s, diagnostics = %v: no budget Claude takes fits under it", c.budget, c.ceiling, body, resp.Diagnostics.Codes())
		case c.want != 0 && (th == nil || th.Budget != c.want):
			t.Errorf("budget %d, ceiling %d: body = %s, want a budget of %d", c.budget, c.ceiling, body, c.want)
		case sent.Inference.MaxTokens != c.maxTokens:
			t.Errorf("budget %d, ceiling %d: body = %s, want maxTokens %d", c.budget, c.ceiling, body, c.maxTokens)
		case moved != c.moved:
			t.Errorf("budget %d, ceiling %d: diagnostics = %+v: a budget moved must be said to have moved, and only then", c.budget, c.ceiling, resp.Diagnostics)
		}
	}
}

// Claude here refuses thinking beside a temperature, a narrow top_p or a forced
// tool call, as on Anthropic's own API, and other wires allow each of them, so
// one gives way to the other rather than the request being refused.
func TestThinkingAndWhatClaudeRefusesBesideItGiveWay(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	tools := []llm.Tool{{Name: "f"}}
	for name, c := range map[string]struct {
		change   func(*llm.Request)
		thinking bool
		want     map[string]string
		reported string
	}{
		"temperature":        {func(r *llm.Request) { r.Temperature = llm.Ptr(0.7) }, true, map[string]string{"temperature": ""}, llm.FieldTemperature},
		"temperature of one": {func(r *llm.Request) { r.Temperature = llm.Ptr(1.0) }, true, map[string]string{"temperature": "1"}, ""},
		"narrow top_p":       {func(r *llm.Request) { r.TopP = llm.Ptr(0.9) }, true, map[string]string{"topP": "0.95"}, llm.FieldTopP},
		"wide top_p":         {func(r *llm.Request) { r.TopP = llm.Ptr(0.97) }, true, map[string]string{"topP": "0.97"}, ""},
		"forced tool": {func(r *llm.Request) {
			r.Tools, r.ToolChoice, r.Temperature = tools, &llm.ToolChoice{Mode: llm.ToolRequired}, llm.Ptr(0.2)
		}, false, map[string]string{"temperature": "0.2", "toolChoice": `{"any":{}}`}, "thinking"},
		"named tool": {func(r *llm.Request) {
			r.Tools, r.ToolChoice = tools, &llm.ToolChoice{Mode: llm.ToolSpecific, Name: "f"}
		}, false, map[string]string{"toolChoice": `{"tool":{"name":"f"}}`}, "thinking"},
		"tool left to choose": {func(r *llm.Request) {
			r.Tools, r.ToolChoice = tools, &llm.ToolChoice{Mode: llm.ToolAuto}
		}, true, map[string]string{"toolChoice": `{"auto":{}}`}, ""},
	} {
		req := hello()
		req.MaxOutputTokens = llm.Ptr(8000)
		req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: 2048}
		c.change(req)
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		var sent struct {
			Fields     map[string]json.RawMessage `json:"additionalModelRequestFields"`
			Inference  map[string]json.RawMessage `json:"inferenceConfig"`
			ToolConfig map[string]json.RawMessage `json:"toolConfig"`
		}
		body := (*bodies)[len(*bodies)-1]
		if err := json.Unmarshal([]byte(body), &sent); err != nil {
			t.Fatal(err)
		}
		if _, on := sent.Fields["thinking"]; on != c.thinking {
			t.Errorf("%s: body = %s: thinking sent %v, want %v", name, body, on, c.thinking)
		}
		for field, want := range c.want {
			got := sent.Inference[field]
			if field == "toolChoice" {
				got = sent.ToolConfig[field]
			}
			if string(got) != want {
				t.Errorf("%s: %s = %q, want %q", name, field, got, want)
			}
		}
		reported := ""
		for _, d := range resp.Diagnostics {
			switch d.Field {
			case llm.FieldTemperature, llm.FieldTopP, "thinking":
				reported = d.Field
			}
		}
		if reported != c.reported {
			t.Errorf("%s: diagnostics = %+v: what gave way must be reported, and nothing else", name, resp.Diagnostics)
		}
	}
}

// Converse hands back redacted thinking, and Claude refuses a tool loop whose
// last turn has lost the thinking it opened with, so the block goes back to
// the backend that wrote it.
func TestRedactedThinkingIsReplayedToTheBackendThatWroteIt(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model:    "anthropic.claude-sonnet-4:0",
		Thinking: &llm.ThinkingConfig{Enabled: true, Budget: 2048},
		Tools:    []llm.Tool{{Name: "f"}},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			{Role: llm.RoleAssistant, Content: []llm.Part{
				&llm.Thinking{Redacted: []byte("RVJlZGFjdGVk")},
				&llm.ToolUse{ID: "t1", Name: "f", Args: json.RawMessage(`{}`)},
			}},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.ToolResult{ID: "t1", Name: "f", Content: []llm.Part{&llm.Text{Text: "42"}}}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if body := (*bodies)[0]; !strings.Contains(body, `{"reasoningContent":{"redactedContent":"RVJlZGFjdGVk"}},{"toolUse"`) {
		t.Fatalf("body = %s: the turn must open with the thinking it was written with", body)
	}
	if resp.Diagnostics.Has(llm.DropThinking) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

// Claude requires the turn whose tool call a request answers to lead with its
// thinking once thinking is on, here as on Anthropic's own API. A turn whose
// thinking was unsigned, stripped before it arrived, or never written opens
// with toolUse instead and is refused beside thinking, wherever the tool
// result sits, so thinking is turned off and said to be. Regression test for
// r4-prov-2.
func TestThinkingIsDisabledWhereverTheResumedTurnDoesNotLeadWithIt(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	ask := llm.Message{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "weather?"}}}
	call := &llm.ToolUse{ID: "t1", Name: "f", Args: json.RawMessage(`{}`)}
	result := &llm.ToolResult{ID: "t1", Name: "f", Content: []llm.Part{&llm.Text{Text: "42"}}}
	for name, c := range map[string]struct {
		msgs     []llm.Message
		thinking bool
	}{
		"unsigned thinking": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "let me check"}, call}},
			{Role: llm.RoleUser, Content: []llm.Part{result}}}, false},
		"thinking stripped before it arrived": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{call}},
			{Role: llm.RoleTool, Content: []llm.Part{result}}}, false},
		"a tool result the user's next words follow": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "let me check"}, call}},
			{Role: llm.RoleTool, Content: []llm.Part{result}},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "and tomorrow?"}}}}, false},
		"signed thinking leading the turn": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "let me check", Signature: "sig"}, call}},
			{Role: llm.RoleTool, Content: []llm.Part{result}}}, true},
	} {
		req := hello()
		req.Messages = c.msgs
		req.Tools = []llm.Tool{{Name: "f"}}
		req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: 2048}
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		body := (*bodies)[len(*bodies)-1]
		if on := strings.Contains(body, `"thinking"`); on != c.thinking {
			t.Errorf("%s: body = %s: thinking sent %v, want %v", name, body, on, c.thinking)
		}
		if resp.Diagnostics.Has(llm.DropThinking) == c.thinking {
			t.Errorf("%s: diagnostics = %v: thinking turned off must be reported, and only then", name, resp.Diagnostics.Codes())
		}
	}
}
