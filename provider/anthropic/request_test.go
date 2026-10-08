// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

// A field of this wire's own travels with the beta flags the caller opened it
// with, beside the operator's. Where the caller's flags are not sent, or the
// field is another wire's, it is not sent either, and strict refuses the loss.
func TestAnExtraFieldTravelsWithTheCallersBetas(t *testing.T) {
	var seen []http.Header
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		seen, bodies = append(seen, r.Header.Clone()), append(bodies, string(b))
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	}))
	defer srv.Close()
	for name, c := range map[string]struct {
		from        string
		noClient    bool
		sent        bool
		wantedBetas string
	}{
		"this wire's own field":           {anthropicmsg.Name, false, true, "operator,context-management-2025-06-27"},
		"another wire's field":            {"openai", false, false, "operator,context-management-2025-06-27"},
		"a surface that refuses the flag": {anthropicmsg.Name, true, false, "operator"},
	} {
		p, err := New(Options{BaseURL: srv.URL, APIKey: "k", Betas: []string{"operator"}, NoClientBetas: c.noClient})
		if err != nil {
			t.Fatal(err)
		}
		req := hello()
		req.Extra = map[string]json.RawMessage{"context_management": json.RawMessage(`{"edits":[{"type":"clear_tool_uses_20250919"}]}`)}
		req.ExtraFrom = c.from
		req.Betas = []string{"context-management-2025-06-27", "operator"}
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		body, header := bodies[len(bodies)-1], seen[len(seen)-1]
		var sent map[string]json.RawMessage
		if err := json.Unmarshal([]byte(body), &sent); err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(header.Values("anthropic-beta"), ","); got != c.wantedBetas {
			t.Errorf("%s: anthropic-beta = %q, want %q, each once", name, got, c.wantedBetas)
		}
		if _, ok := sent["context_management"]; ok != c.sent {
			t.Errorf("%s: body = %s: the field was sent %v, want %v", name, body, ok, c.sent)
		}
		strict := (llm.LossPolicy{Strict: true}).Check(resp.Diagnostics)
		switch {
		case c.sent && resp.Diagnostics.Has(llm.DropExtraField):
			t.Errorf("%s: diagnostics = %+v: a field that was sent was reported lost", name, resp.Diagnostics)
		case !c.sent && (!resp.Diagnostics.Has(llm.DropExtraField) || strict == nil):
			t.Errorf("%s: diagnostics = %+v: a field the backend never saw was served under strict", name, resp.Diagnostics)
		}
	}
}

// An explicit budget, such as one Gemini takes, can sit outside what this
// backend does, and sent as it is the request is refused and the chain stops.
// A ceiling goose supplied itself is raised to fit the budget instead, leaving
// the answer after it the default, and the interleaved flag lets a budget run
// past the ceiling.
func TestAnExplicitBudgetIsMovedToTheNearestTheBackendTakes(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	for _, c := range []struct {
		budget, ceiling     int
		interleaved         bool
		want, wantMaxTokens int
		moved               bool
	}{
		{512, 4000, false, 1024, 4000, true},
		{8192, 0, false, 8192, 8192 + defaultMaxTokens, false},
		{defaultMaxTokens, 0, false, defaultMaxTokens, 2 * defaultMaxTokens, false},
		{4000, 0, false, 4000, 4000 + defaultMaxTokens, false},
		{512, 0, false, 1024, 1024 + defaultMaxTokens, true},
		{40000, 0, false, thinkingCeiling - defaultMaxTokens, thinkingCeiling, true},
		{math.MaxInt, 0, false, thinkingCeiling - defaultMaxTokens, thinkingCeiling, true},
		{2048, 2048, false, 2047, 2048, true},
		{2048, 8000, false, 2048, 8000, false},
		{512, 1000, false, 0, 1000, false},
		{8192, 4000, true, 8192, 4000, false},
		{512, 1000, true, 1024, 1000, true},
	} {
		req := hello()
		if c.ceiling > 0 {
			req.MaxOutputTokens = llm.Ptr(c.ceiling)
		}
		if c.interleaved {
			req.Betas = []string{"interleaved-thinking-2025-05-14"}
		}
		req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: c.budget}
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		var sent struct {
			MaxTokens int `json:"max_tokens"`
			Thinking  *struct {
				Budget int `json:"budget_tokens"`
			} `json:"thinking"`
		}
		body := (*bodies)[len(*bodies)-1]
		if err := json.Unmarshal([]byte(body), &sent); err != nil {
			t.Fatal(err)
		}
		moved := false
		for _, d := range resp.Diagnostics {
			moved = moved || d.Field == "thinking.budget"
		}
		switch {
		case c.want == 0 && (sent.Thinking != nil || !resp.Diagnostics.Has(llm.DropThinking)):
			t.Errorf("budget %d, ceiling %d: body = %s, diagnostics = %v: nothing this backend takes fits", c.budget, c.ceiling, body, resp.Diagnostics.Codes())
		case c.want != 0 && (sent.Thinking == nil || sent.Thinking.Budget != c.want):
			t.Errorf("budget %d, ceiling %d: body = %s, want a budget of %d", c.budget, c.ceiling, body, c.want)
		case sent.MaxTokens != c.wantMaxTokens:
			t.Errorf("budget %d, ceiling %d: body = %s, want max_tokens %d", c.budget, c.ceiling, body, c.wantMaxTokens)
		case moved != c.moved:
			t.Errorf("budget %d, ceiling %d: diagnostics = %+v: a budget moved must be said to have moved, and only then", c.budget, c.ceiling, resp.Diagnostics)
		}
	}
}

// This backend refuses thinking beside a temperature, a top_k, a narrow top_p
// or a forced tool call, which other wires allow, so each gives way to the
// other rather than the request being refused.
func TestThinkingAndWhatItRefusesBesideItGiveWay(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
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
		"top_k":              {func(r *llm.Request) { r.TopK = llm.Ptr(40) }, true, map[string]string{"top_k": ""}, llm.FieldTopK},
		"narrow top_p":       {func(r *llm.Request) { r.TopP = llm.Ptr(0.9) }, true, map[string]string{"top_p": "0.95"}, llm.FieldTopP},
		"wide top_p":         {func(r *llm.Request) { r.TopP = llm.Ptr(0.97) }, true, map[string]string{"top_p": "0.97"}, ""},
		"forced tool": {func(r *llm.Request) {
			r.Tools, r.ToolChoice, r.Temperature = tools, &llm.ToolChoice{Mode: llm.ToolRequired}, llm.Ptr(0.2)
		}, false, map[string]string{"temperature": "0.2", "tool_choice": `{"type":"any"}`}, "thinking"},
		"named tool": {func(r *llm.Request) {
			r.Tools, r.ToolChoice = tools, &llm.ToolChoice{Mode: llm.ToolSpecific, Name: "f"}
		}, false, map[string]string{"tool_choice": `{"name":"f","type":"tool"}`}, "thinking"},
		"tool left to choose": {func(r *llm.Request) {
			r.Tools, r.ToolChoice = tools, &llm.ToolChoice{Mode: llm.ToolAuto}
		}, true, map[string]string{"tool_choice": `{"type":"auto"}`}, ""},
	} {
		req := hello()
		req.MaxOutputTokens = llm.Ptr(8000)
		req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: 2048}
		c.change(req)
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		body := (*bodies)[len(*bodies)-1]
		var sent map[string]json.RawMessage
		if err := json.Unmarshal([]byte(body), &sent); err != nil {
			t.Fatal(err)
		}
		if _, on := sent["thinking"]; on != c.thinking {
			t.Errorf("%s: body = %s: thinking sent %v, want %v", name, body, on, c.thinking)
		}
		for field, want := range c.want {
			if got := string(sent[field]); got != want {
				t.Errorf("%s: %s = %q, want %q", name, field, got, want)
			}
		}
		reported := ""
		for _, d := range resp.Diagnostics {
			switch d.Field {
			case llm.FieldTemperature, llm.FieldTopK, llm.FieldTopP, "thinking":
				reported = d.Field
			}
		}
		if reported != c.reported {
			t.Errorf("%s: diagnostics = %+v: what gave way must be reported, and nothing else", name, resp.Diagnostics)
		}
	}
}

// A tool server's toolset forces a tool call the same as a function tool
// does, so a choice forced over servers alone must also turn thinking off,
// not just when a function tool is offered beside them.
func TestThinkingGivesWayToAForcedChoiceOverAToolServerAlone(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	req := hello()
	req.MaxOutputTokens = llm.Ptr(8000)
	req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: 2048}
	req.MCPServers = []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp"}}
	req.ToolChoice = &llm.ToolChoice{Mode: llm.ToolRequired}
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[len(*bodies)-1]
	var sent map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &sent); err != nil {
		t.Fatal(err)
	}
	if _, on := sent["thinking"]; on {
		t.Fatalf("body = %s: thinking was sent beside a choice forced over a tool server alone, which this backend refuses", body)
	}
	if got, want := string(sent["tool_choice"]), `{"type":"any"}`; got != want {
		t.Fatalf("tool_choice = %s, want %s", got, want)
	}
	if !resp.Diagnostics.Has(llm.DropThinking) {
		t.Fatalf("diagnostics = %+v: thinking gave way without being reported", resp.Diagnostics)
	}
}

// A client of this wire opens a field with its own beta flag, and an image
// kept in the Files API is one such field: it reaches the backend with the
// flag, not as an image with neither bytes nor an address.
func TestAFilesImageReachesTheBackendWithItsBeta(t *testing.T) {
	var header http.Header
	var body string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		header, body = r.Header.Clone(), string(b)
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	}))
	defer srv.Close()
	p, err := New(Options{BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	req, _, _, err := anthropicmsg.New(anthropicmsg.Options{}).Decode(&wire.Inbound{
		Method: "POST", Header: http.Header{"Anthropic-Beta": {"files-api-2025-04-14"}},
		Body: []byte(`{"model":"claude-test","max_tokens":64,"messages":[{"role":"user","content":[
			{"type":"image","source":{"type":"file","file_id":"file_abc"}},
			{"type":"text","text":"what is this?"}]}]}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(body, `"source":{"type":"file","file_id":"file_abc"},"type":"image"`) {
		t.Fatalf("body = %s: the image never reached the backend", body)
	}
	if got := header.Values("anthropic-beta"); len(got) != 1 || got[0] != "files-api-2025-04-14" {
		t.Fatalf("anthropic-beta = %v: the file cannot be read without the flag", got)
	}
	if resp.Diagnostics.Has(llm.DropImage) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestThinkingWithoutABudgetGetsOneTheBackendTakes(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	for _, budget := range []int{0, -1} {
		for ceiling, want := range map[int]int{8000: 4000, 1500: 1024, 1000: 0, 1024: 0} {
			req := hello()
			req.MaxOutputTokens = llm.Ptr(ceiling)
			req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: budget}
			resp, err := p.Complete(context.Background(), attempt(req))
			if err != nil {
				t.Fatal(err)
			}
			var sent struct {
				Thinking *struct {
					Budget int `json:"budget_tokens"`
				} `json:"thinking"`
			}
			body := (*bodies)[len(*bodies)-1]
			if err := json.Unmarshal([]byte(body), &sent); err != nil {
				t.Fatal(err)
			}
			chose := false
			for _, d := range resp.Diagnostics {
				chose = chose || d.Field == "thinking.budget"
			}
			switch {
			case want == 0 && sent.Thinking != nil:
				t.Errorf("budget %d, ceiling %d: body = %s: no budget of at least 1024 fits under it, and one that does not is refused", budget, ceiling, body)
			case want == 0 && !resp.Diagnostics.Has(llm.DropThinking):
				t.Errorf("budget %d, ceiling %d: diagnostics = %v: thinking went unasked for and nothing said so", budget, ceiling, resp.Diagnostics.Codes())
			case want == 0 && chose:
				t.Errorf("budget %d, ceiling %d: diagnostics = %+v: a budget was said to be chosen for thinking that was not enabled", budget, ceiling, resp.Diagnostics)
			case want != 0 && (sent.Thinking == nil || sent.Thinking.Budget != want):
				t.Errorf("budget %d, ceiling %d: body = %s, want a budget of %d", budget, ceiling, body, want)
			case want != 0 && chose != (budget < 0):
				t.Errorf("budget %d, ceiling %d: diagnostics = %+v: only a request to decide is answered with a chosen budget", budget, ceiling, resp.Diagnostics)
			}
		}
	}
}

// A tool loop's own turn is the one Claude requires thinking to lead when
// thinking is enabled, and a signature this backend cannot verify was
// already stripped from that turn's thinking. Sending thinking on beside a
// turn that now opens with tool_use gets the whole request refused; goose
// must turn thinking off instead, and say why. Regression test for
// r3-httpproviders-2.
func TestThinkingIsDisabledWhenTheTurnItWouldResumeLostItsLead(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	req := hello()
	req.Messages = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "weather?"}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{Text: "let me check"},
			&llm.ToolUse{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`)},
		}},
		{Role: llm.RoleUser, Content: []llm.Part{
			&llm.ToolResult{ID: "call_1", Name: "get_weather", Content: []llm.Part{&llm.Text{Text: "20C"}}},
		}},
	}
	req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: 2048}
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Thinking json.RawMessage `json:"thinking"`
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	body := (*bodies)[len(*bodies)-1]
	if err := json.Unmarshal([]byte(body), &sent); err != nil {
		t.Fatal(err)
	}
	if sent.Thinking != nil {
		t.Fatalf("body = %s: thinking must not be sent when the turn it would resume lost its lead thinking to an unsigned drop", body)
	}
	if !resp.Diagnostics.Has(llm.DropThinking) {
		t.Fatalf("diagnostics = %+v: thinking was turned off and nothing said so", resp.Diagnostics)
	}
	if len(sent.Messages) != 3 {
		t.Fatalf("messages = %+v", sent.Messages)
	}
	// The assistant turn itself still reaches the backend, tool_use and all.
	assistant := sent.Messages[1]
	if assistant.Role != "assistant" || len(assistant.Content) != 1 || assistant.Content[0]["type"] != "tool_use" {
		t.Fatalf("assistant turn = %+v", assistant)
	}
}

// The turn Claude requires thinking to lead is judged as this backend receives
// it: a tool result in a message of its own, or one the user's next words were
// folded in beside, still ends the conversation in that tool loop, and a turn
// whose thinking was stripped before it arrived leads with tool_use all the
// same. Each is refused beside thinking, so thinking is turned off and said to
// be. Regression test for r4-prov-1.
func TestThinkingIsDisabledWhereverTheResumedTurnDoesNotLeadWithIt(t *testing.T) {
	p, bodies := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"id":"m","model":"c","content":[],"stop_reason":"end_turn","usage":{}}`))
	})
	ask := llm.Message{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "weather?"}}}
	call := &llm.ToolUse{ID: "call_1", Name: "get_weather", Args: json.RawMessage(`{}`)}
	result := &llm.ToolResult{ID: "call_1", Name: "get_weather", Content: []llm.Part{&llm.Text{Text: "20C"}}}
	unsigned := llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "let me check"}, call}}
	for name, c := range map[string]struct {
		msgs     []llm.Message
		thinking bool
	}{
		"a tool result in its own message": {[]llm.Message{ask, unsigned,
			{Role: llm.RoleTool, Content: []llm.Part{result}}}, false},
		"a tool result the user's next words follow": {[]llm.Message{ask, unsigned,
			{Role: llm.RoleUser, Content: []llm.Part{result}},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "and tomorrow?"}}}}, false},
		"thinking stripped before it arrived": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{call}},
			{Role: llm.RoleUser, Content: []llm.Part{result}}}, false},
		"signed thinking leading the turn": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "let me check", Signature: "sig"}, call}},
			{Role: llm.RoleTool, Content: []llm.Part{result}}}, true},
		"a tool loop already answered": {[]llm.Message{ask,
			{Role: llm.RoleAssistant, Content: []llm.Part{call}},
			{Role: llm.RoleTool, Content: []llm.Part{result}},
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "20C"}}},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "thanks"}}}}, true},
	} {
		req := hello()
		req.Messages = c.msgs
		req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: 2048}
		resp, err := p.Complete(context.Background(), attempt(req))
		if err != nil {
			t.Fatal(err)
		}
		body := (*bodies)[len(*bodies)-1]
		var sent map[string]json.RawMessage
		if err := json.Unmarshal([]byte(body), &sent); err != nil {
			t.Fatal(err)
		}
		if _, on := sent["thinking"]; on != c.thinking {
			t.Errorf("%s: body = %s: thinking sent %v, want %v", name, body, on, c.thinking)
		}
		if resp.Diagnostics.Has(llm.DropThinking) == c.thinking {
			t.Errorf("%s: diagnostics = %v: thinking turned off must be reported, and only then", name, resp.Diagnostics.Codes())
		}
	}
}
