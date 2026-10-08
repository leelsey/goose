// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/route"
)

func pair(t *testing.T, first, second llm.Provider, opts ...Option) *Client {
	t.Helper()
	c, err := New(append([]Option{
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: first.ID()}, {Provider: second.ID()},
		}}),
	}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func withTool(r *llm.Request) *llm.Request {
	r.Tools = []llm.Tool{{Name: "act", Schema: []byte(`{"type":"object"}`)}}
	return r
}

// A link that would have to strip the tools is the chain's last resort, not a
// refusal. While the link that carries them is failing, or skipped by its
// breaker, the one without them still answers and says what it lost.
func TestToolsFallToALinkWithoutThemOnceTheOthersHaveFailed(t *testing.T) {
	cli := &capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}
	api := &failing{capped{id: "api", caps: full()}}
	c := pair(t, cli, api, WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}))
	for round := range 2 {
		cli.seen, api.seen = nil, nil
		resp, err := c.Chat(context.Background(), withTool(req()))
		if err != nil {
			t.Fatalf("round %d: err = %v: the link without tools was healthy and never asked", round, err)
		}
		if resp.Provider != "cli" || len(cli.seen.Request.Tools) != 0 || !resp.Diagnostics.Has(llm.DropToolDefinition) {
			t.Fatalf("round %d: served by %q with tools %+v, diagnostics %v",
				round, resp.Provider, cli.seen.Request.Tools, resp.Diagnostics.Codes())
		}
		switch round {
		case 0:
			if api.seen == nil || len(resp.Attempts) != 2 || resp.Attempts[0].Provider != "api" {
				t.Fatalf("attempts = %+v: the link that carries the tools goes first", resp.Attempts)
			}
			if resp.Attempts[0].Index != 1 || resp.Attempts[1].Index != 0 {
				t.Fatalf("attempts = %+v: a link keeps its place in the route on the record", resp.Attempts)
			}
		case 1:
			if api.seen != nil || !resp.Diagnostics.Has(llm.PolicyBreakerOpen) {
				t.Fatalf("diagnostics = %v: the failing link was to be skipped", resp.Diagnostics.Codes())
			}
		}
	}
}

// The last resort is reached only once the links that carry the tools have
// failed. Refused before anything was sent, it could not have served either,
// and the failure the caller is owed is theirs: a busy backend a retry may
// cure, not a request no backend could take.
func TestALastResortsRefusalLeavesTheOthersFailureStanding(t *testing.T) {
	forced := func() *llm.Request {
		r := withTool(req())
		r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolRequired}
		return r
	}
	pictured := func() *llm.Request {
		r := withTool(req())
		r.Messages[0].Content = append(r.Messages[0].Content, &llm.Image{MIME: "image/png", Data: []byte("x")})
		return r
	}
	for _, tc := range []struct {
		what    string
		request func() *llm.Request
		open    bool
	}{
		{"a call the caller insists on", forced, false},
		{"a call the caller insists on, the carrier skipped", forced, true},
		{"an image the last resort cannot see", pictured, false},
	} {
		for _, stream := range []bool{false, true} {
			cli := &capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}
			api := &failing{capped{id: "api", caps: full()}}
			c := pair(t, cli, api, WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}))
			if tc.open {
				c.breaker.failed("api", 0)
			}
			var err error
			if stream {
				_, err = c.Stream(context.Background(), tc.request())
			} else {
				_, err = c.Chat(context.Background(), tc.request())
			}
			e := llm.AsError(err)
			if e == nil || e.Kind != llm.KindOverloaded || cli.seen != nil {
				t.Fatalf("%s, stream %v: err = %v: the last resort's refusal hid why the link that carries the tools failed",
					tc.what, stream, err)
			}
			if !tc.open && !e.Sent {
				t.Fatalf("%s, stream %v: the failure the caller is owed is one that reached a backend", tc.what, stream)
			}
		}
	}
}

// Where the caller's format has no field for a tool call, no link can hand one
// back, so the route's own order stands and the first link strips the tools.
func TestToolsAFormatCannotRenderLeaveTheRouteInOrder(t *testing.T) {
	cli := &capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}
	api := &capped{id: "api", caps: full()}
	c := pair(t, cli, api,
		WithTools("grp", []llm.Tool{{Name: "act", Schema: []byte(`{"type":"object"}`)}}),
		WithDefaultTools("grp"))
	resp, err := c.Chat(context.Background(), req(), WithRenders(llm.Renders{Candidates: 1}))
	if err != nil {
		t.Fatalf("err = %v: the link without tools could answer", err)
	}
	if resp.Provider != "cli" || api.seen != nil || len(resp.Attempts) != 1 {
		t.Fatalf("served by %q after %+v", resp.Provider, resp.Attempts)
	}
}

// A tool server the backend dials for itself is a tool like any other: a link
// that cannot dial it is reached only once the link that can has been tried.
func TestToolServersAreLeftToALinkThatDialsThem(t *testing.T) {
	caps := full()
	caps.MCPServers = true
	servers := func() *llm.Request {
		r := req()
		r.MCPServers = []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp"}}
		return r
	}

	cli := &capped{id: "cli", caps: full()}
	api := &capped{id: "api", caps: caps}
	c := pair(t, cli, api, WithLossPolicy(llm.LossPolicy{Strict: true}))
	resp, err := c.Chat(context.Background(), servers())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Provider != "api" || cli.seen != nil || len(api.seen.Request.MCPServers) != 1 {
		t.Fatalf("served by %q: the link that dials the server was not the one asked", resp.Provider)
	}

	down := &failing{capped{id: "api", caps: caps}}
	cli.seen = nil
	c = pair(t, cli, down)
	resp, err = c.Chat(context.Background(), servers())
	if err != nil {
		t.Fatalf("err = %v: the link that cannot dial it still answers once the other has failed", err)
	}
	if resp.Provider != "cli" || !resp.Diagnostics.Has(llm.DropMCPServer) {
		t.Fatalf("served by %q, diagnostics %v", resp.Provider, resp.Diagnostics.Codes())
	}
}

// Tools that travel as written for one wire reach only a backend on it. They
// are left to such a link like any other tool, and a call the caller insists on
// among them is never answered without them.
func TestCarriedToolsAreLeftToALinkOnTheirWire(t *testing.T) {
	caps := full()
	caps.Extensions = []string{"openai-responses"}
	carried := func(choice string) *llm.Request {
		r := req()
		r.Extra = map[string]json.RawMessage{
			"tools":       json.RawMessage(`[{"type":"custom","name":"apply_patch"}]`),
			"tool_choice": json.RawMessage(choice),
		}
		r.ExtraFrom = "openai-responses"
		return r
	}

	for _, choice := range []string{`"required"`, `"auto"`, `{"type":"custom","name":"apply_patch"}`} {
		claude := &capped{id: "claude", caps: full()}
		gpt := &capped{id: "gpt", caps: caps}
		resp, err := pair(t, claude, gpt).Chat(context.Background(), carried(choice))
		if err != nil {
			t.Fatalf("choice %s: %v", choice, err)
		}
		if resp.Provider != "gpt" || claude.seen != nil || len(gpt.seen.Request.Extra) != 2 {
			t.Fatalf("choice %s: served by %q, and the link on the tools' wire was not the one asked", choice, resp.Provider)
		}
	}

	claude := &capped{id: "claude", caps: full()}
	down := &failing{capped{id: "gpt", caps: caps}}
	_, err := pair(t, claude, down).Chat(context.Background(), carried(`"required"`))
	if e := llm.AsError(err); e == nil || claude.seen != nil {
		t.Fatalf("err = %v: a call the caller insists on was answered without the tool", err)
	}
	resp, err := pair(t, claude, down).Chat(context.Background(), carried(`"auto"`))
	if err != nil || resp.Provider != "claude" || !resp.Diagnostics.Has(llm.DropExtraField) {
		t.Fatalf("err = %v: a choice left to the model is answered without the tools once their link has failed", err)
	}
}

// A backend that takes tools but cannot be told to call one would answer a
// forced call as it pleased, so it is refused there and the chain walks on.
func TestAForcedCallIsNotLeftToABackendThatCannotBeToldToMakeIt(t *testing.T) {
	loose := full()
	loose.ToolChoice = false
	for _, choice := range []*llm.ToolChoice{{Mode: llm.ToolRequired}, {Mode: llm.ToolSpecific, Name: "act"}} {
		local := &capped{id: "local", caps: loose}
		api := &capped{id: "api", caps: full()}
		r := withTool(req())
		r.ToolChoice = choice
		resp, err := pair(t, local, api, WithLossPolicy(llm.LossPolicy{Strict: true})).Chat(context.Background(), r)
		if err != nil {
			t.Fatalf("choice %+v: %v", choice, err)
		}
		if resp.Provider != "api" || local.seen != nil || api.seen.Request.ToolChoice == nil {
			t.Fatalf("choice %+v: served by %q, and the call it insists on was not honoured", choice, resp.Provider)
		}

		local.seen = nil
		c, err := New(WithProvider(local))
		if err != nil {
			t.Fatal(err)
		}
		r = withTool(req())
		r.ToolChoice = choice
		if _, err := c.Chat(context.Background(), r); llm.AsError(err) == nil || llm.AsError(err).Kind != llm.KindUnsupported || local.seen != nil {
			t.Fatalf("choice %+v: err = %v: a backend that cannot be held to the call answered anyway", choice, err)
		}
	}
}

// A choice over a tool server's tools alone governs those tools. Dropped, a
// caller's "none" would leave the model free to run a tool it was told not to.
func TestAToolChoiceOverToolServersAloneTravels(t *testing.T) {
	caps := full()
	caps.MCPServers = true
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	r := req()
	r.MCPServers = []llm.MCPServer{{Name: "dice", URL: "https://tools.test/mcp"}}
	r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolNone}
	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if tc := p.seen.Request.ToolChoice; tc == nil || tc.Mode != llm.ToolNone {
		t.Fatalf("choice = %+v: it governs the server's tools", tc)
	}
	if resp.Diagnostics.Has(llm.DropToolDefinition) {
		t.Fatalf("diagnostics = %+v: nothing was dropped", resp.Diagnostics)
	}

	var body map[string]json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x",`+
			`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()
	claude, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	if c, err = New(WithProvider(claude)); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), r); err != nil {
		t.Fatal(err)
	}
	if string(body["tool_choice"]) != `{"type":"none"}` || body["mcp_servers"] == nil {
		t.Fatalf("tool_choice = %s, mcp_servers = %s: the choice did not travel with the servers",
			body["tool_choice"], body["mcp_servers"])
	}
}

// billing fails having been sent and having counted what it read: unstreamed
// with what it had answered, streamed on its first event, as a backend that
// began and then fell over does.
type billing struct{ capped }

func (b *billing) fault() *llm.Error {
	return &llm.Error{Kind: llm.KindUpstream, Provider: b.id, Status: 500, Sent: true, Retryable: true, Message: "fell over"}
}

func (b *billing) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	b.seen = a
	return &llm.Response{Usage: llm.Usage{Prompt: 9, Reported: true}}, b.fault()
}

func (b *billing) Stream(_ context.Context, a *llm.Attempt) (*llm.Stream, error) {
	b.seen = a
	return llm.StaticStream(llm.StreamMeta{Provider: b.id}, []*llm.Frame{
		{Err: b.fault(), Usage: &llm.Usage{Prompt: 9, Reported: true}},
	}), nil
}

// failedWith makes a call that fails and says what the failure counted.
// Unstreamed, the answer handed back beside it must be the failed link's own.
func failedWith(t *testing.T, c *Client, r *llm.Request, stream bool) (*llm.Error, int) {
	t.Helper()
	if stream {
		_, err := c.Stream(context.Background(), r)
		e := llm.AsError(err)
		if e == nil || e.Usage == nil {
			t.Fatalf("err = %+v: the failure says nothing of what was counted", err)
		}
		return e, e.Usage.Prompt
	}
	resp, err := c.Chat(context.Background(), r)
	e := llm.AsError(err)
	if e == nil || e.Usage == nil || resp == nil || resp.Usage.Prompt != e.Usage.Prompt {
		t.Fatalf("resp = %+v, err = %+v: the answer handed back is not the one that failed", resp, e)
	}
	return e, e.Usage.Prompt
}

// The last resort's own refusal does not hide the failure of the links ahead
// of it, but a failure of its own that reached the backend is not a refusal:
// it is the one that was paid for, and the caller is owed it, with what it
// counted, rather than the lead's refusal that sent nothing.
func TestALastResortsSentFailureIsNotHiddenBehindTheLeadsRefusal(t *testing.T) {
	for _, tc := range []struct {
		what string
		open bool
	}{
		{"the lead skipped by its breaker", true},
		{"the lead refused before it was sent", false},
	} {
		for _, stream := range []bool{false, true} {
			caps := full()
			if !tc.open {
				caps.MaxInputBytes = 1
			}
			api := &capped{id: "api", caps: caps}
			cli := &billing{capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}}
			c := pair(t, cli, api, WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}))
			if tc.open {
				c.breaker.failed("api", 0)
			}
			e, prompt := failedWith(t, c, withTool(req()), stream)
			if e.Provider != "cli" || !e.Sent || e.Status != 500 || prompt != 9 {
				t.Fatalf("%s, stream %v: err = %+v: a failure the last resort was sent and billed for was hidden behind a refusal that sent nothing",
					tc.what, stream, e)
			}
			if api.seen != nil || cli.seen == nil {
				t.Fatalf("%s, stream %v: the links were not tried in the order the tools set", tc.what, stream)
			}
		}
	}
}

// Between links that lead, too, one refused before it was sent says only that
// it could not have served either. The failure of the link before it, which
// went and was billed, is the one the caller is owed, along with its own answer
// rather than another link's.
func TestALaterLinksRefusalDoesNotHideAnEarlierSentFailure(t *testing.T) {
	for _, stream := range []bool{false, true} {
		narrow := full()
		narrow.MaxInputBytes = 1
		first := &billing{capped{id: "first", caps: full()}}
		second := &capped{id: "second", caps: narrow}
		e, prompt := failedWith(t, pair(t, first, second), req(), stream)
		if e.Provider != "first" || !e.Sent || e.Status != 500 || prompt != 9 {
			t.Fatalf("stream %v: err = %+v: the failure that went and was billed was replaced by a refusal that sent nothing", stream, e)
		}
		if first.seen == nil || second.seen != nil {
			t.Fatalf("stream %v: the chain did not reach the later link and refuse it unsent", stream)
		}
	}
}
