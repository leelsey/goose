// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

type capped struct {
	id   string
	caps llm.Capabilities
	seen *llm.Attempt
}

func (c *capped) ID() string                     { return c.id }
func (c *capped) Capabilities() llm.Capabilities { return c.caps }

func (c *capped) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	c.seen = a
	return &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Content: []llm.Part{&llm.Text{Text: "ok"}}}, Stop: llm.StopEnd}}}, nil
}

func (c *capped) Stream(_ context.Context, a *llm.Attempt) (*llm.Stream, error) {
	c.seen = a
	return llm.StaticStream(llm.StreamMeta{Provider: c.id}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "ok"}}}, {Stop: llm.StopEnd},
	}), nil
}

func full() llm.Capabilities {
	return llm.Capabilities{
		Streaming: true, Tools: true, ToolChoice: true, ImageIn: true, Thinking: true,
		Effort: true, ResponseFormat: true, Sessions: true, Candidates: 4,
		CacheBreakpoints: 4, Usage: llm.UsageShape{StreamUsage: true},
		SystemPrompt: true, ToolResults: true, ParallelTools: true,
		ThinkingSignatures: true, RedactedThinking: true,
		Extensions: []string{"openai"},
	}
}

func TestUnservableRequestFailsBeforeAnythingIsSpent(t *testing.T) {
	p := &capped{id: "one", caps: llm.Capabilities{Streaming: true}}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	r := req()
	r.Candidates = 3
	_, err = c.Chat(context.Background(), r)
	if err == nil {
		t.Fatal("a backend that returns one completion cannot answer a request for three")
	}
	e := llm.AsError(err)
	if e.Kind != llm.KindUnsupported || e.Sent {
		t.Fatalf("error = %+v: nothing left the process, so nothing was billed", e)
	}
	if p.seen != nil {
		t.Fatal("the request reached the backend anyway")
	}
	records := mem.Records()
	if len(records) != 1 || records[0].Sent {
		t.Fatalf("records = %+v: the refusal is on the record and costs nothing", records)
	}
}

func TestAnImageIsNotSentToABackendThatCannotRead(t *testing.T) {
	p := &capped{id: "one", caps: llm.Capabilities{Streaming: true, Candidates: 1}}
	c, _ := New(WithProvider(p))
	r := req()
	r.Messages[0].Content = append(r.Messages[0].Content, &llm.Image{MIME: "image/png", Data: []byte("x")})
	if _, err := c.Chat(context.Background(), r); err == nil {
		t.Fatal("answering without the picture answers a different question")
	}
}

func TestTheChainTriesTheNextLinkWhenOneCannotServe(t *testing.T) {
	narrow := &capped{id: "narrow", caps: llm.Capabilities{Streaming: true}}
	wide := &capped{id: "wide", caps: full()}
	c, err := New(
		WithProvider(narrow), WithProvider(wide),
		WithRoutes(route.Rule{Match: route.Always(), Chain: []route.Target{{Provider: "narrow"}, {Provider: "wide"}}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	r := req()
	r.Candidates = 3
	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if wide.seen == nil {
		t.Fatal("a link that cannot serve should hand on to one that can")
	}
	if len(resp.Attempts) != 2 || resp.Attempts[0].Sent {
		t.Fatalf("attempts = %+v", resp.Attempts)
	}
}

func TestWhatWillNotBeHonouredIsStrippedAndReported(t *testing.T) {
	p := &capped{id: "one", caps: llm.Capabilities{Streaming: true, Candidates: 1, ImageIn: true}}
	c, _ := New(WithProvider(p))
	r := req()
	r.Tools = []llm.Tool{{Name: "search", Schema: json.RawMessage(`{}`)}}
	r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolAuto}
	r.ResponseFormat = &llm.ResponseFormat{Type: "json_object"}
	r.Thinking = &llm.ThinkingConfig{Enabled: true}
	r.Effort = llm.EffortHigh
	r.SystemCache = &llm.CacheHint{}

	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	sent := p.seen.Request
	if len(sent.Tools) != 0 || sent.ToolChoice != nil || sent.ResponseFormat != nil ||
		sent.Thinking != nil || sent.Effort != "" || sent.SystemCache != nil {
		t.Fatalf("request = %+v: what the backend will not honour must not travel", sent)
	}
	if len(r.Tools) != 1 || r.Effort != llm.EffortHigh {
		t.Fatal("the caller's request was edited")
	}
	codes := llm.Diagnostics(resp.Diagnostics).Codes()
	for _, want := range []string{llm.DropToolDefinition, llm.DropExtraField, llm.DropThinking, llm.CapCacheBreakpoint} {
		found := false
		for _, got := range codes {
			if got == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("diagnostics = %v, want %s among them", codes, want)
		}
	}
}

func TestAnUnmeasurableStreamIsReportedBeforeItRuns(t *testing.T) {
	p := &capped{id: "one", caps: llm.Capabilities{Streaming: true, Candidates: 1}}
	c, _ := New(WithProvider(p))
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	found := false
	for _, d := range s.Diagnostics() {
		if d.Code == llm.CapNoStreamUsage {
			found = true
		}
	}
	if !found {
		t.Fatalf("diagnostics = %+v: a call that books zero must say why", s.Diagnostics())
	}
}

func TestAnAbandonedStreamStillReleasesItsSlot(t *testing.T) {
	p := &gated{capped: capped{id: "one", caps: full()}, slots: make(chan struct{}, 1)}
	c, _ := New(WithProvider(p))
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := c.Stream(ctx, req()); err != nil {
		t.Fatal(err)
	}
	if len(p.slots) != 1 {
		t.Fatal("the slot was never taken")
	}
	cancel()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if len(p.slots) == 0 {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("an abandoned stream held the slot, and with it the provider's child process")
}

type gated struct {
	capped
	slots chan struct{}
}

func (g *gated) Acquire(ctx context.Context) (func(), error) {
	select {
	case g.slots <- struct{}{}:
		return func() { <-g.slots }, nil
	default:
		return func() {}, &llm.Error{Kind: llm.KindSaturated, Message: "busy"}
	}
}

type failing struct {
	capped
}

func (f *failing) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	f.seen = a
	return nil, &llm.Error{Kind: llm.KindOverloaded, Message: "busy", Sent: true, Retryable: true}
}

func (f *failing) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	_, err := f.Complete(ctx, a)
	return nil, err
}

func TestClientKeyDoesNotFollowAFailover(t *testing.T) {
	first := &failing{capped{id: "anthropic", caps: llm.Capabilities{Streaming: true, KeyScheme: llm.SchemeAnthropic}}}
	second := &capped{id: "gateway", caps: llm.Capabilities{Streaming: true}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(),
			Chain: []route.Target{{Provider: "anthropic"}, {Provider: "gateway"}}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req(),
		WithCredential(llm.Credential{Key: "sk-ant-caller-own", Scheme: llm.SchemeAnthropic}))
	if err != nil {
		t.Fatal(err)
	}
	if first.seen.Credential.Key != "sk-ant-caller-own" {
		t.Fatalf("the backend the key was minted for did not receive it: %+v", first.seen.Credential)
	}
	if second.seen.Credential.Key != "" {
		t.Fatalf("credential = %+v: the caller's key reached a second vendor", second.seen.Credential)
	}
	var withheld bool
	for _, d := range resp.Diagnostics {
		if d.Code == llm.PolicyKeyWithheld {
			withheld = true
		}
	}
	if !withheld {
		t.Fatalf("diagnostics = %+v: withholding a credential changes who pays, so it is on the record", resp.Diagnostics)
	}
}

func TestClientKeyTravelsBetweenBackendsOfTheSameVendor(t *testing.T) {
	first := &failing{capped{id: "primary", caps: llm.Capabilities{Streaming: true, KeyScheme: llm.SchemeAnthropic}}}
	second := &capped{id: "secondary", caps: llm.Capabilities{Streaming: true, KeyScheme: llm.SchemeAnthropic}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(),
			Chain: []route.Target{{Provider: "primary"}, {Provider: "secondary"}}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req(),
		WithCredential(llm.Credential{Key: "sk-ant-caller-own", Scheme: llm.SchemeAnthropic})); err != nil {
		t.Fatal(err)
	}
	if second.seen.Credential.Key != "sk-ant-caller-own" {
		t.Fatal("two entries for one vendor are a failover the caller's key survives")
	}
}

func TestUnplaceableClientKeyStopsAtTheFirstLink(t *testing.T) {
	first := &failing{capped{id: "one", caps: llm.Capabilities{Streaming: true}}}
	second := &capped{id: "two", caps: llm.Capabilities{Streaming: true}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(),
			Chain: []route.Target{{Provider: "one"}, {Provider: "two"}}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req(),
		WithCredential(llm.Credential{Key: "sk-unlabelled", Headers: map[string]string{"x-extra": "v"}})); err != nil {
		t.Fatal(err)
	}
	if first.seen.Credential.Key != "sk-unlabelled" {
		t.Fatal("the link the request was routed to must still receive it")
	}
	if second.seen.Credential.Key != "" || len(second.seen.Credential.Headers) != 0 {
		t.Fatalf("credential = %+v: an unplaceable key must not follow a failover", second.seen.Credential)
	}
}

func TestToolResultsAreRefusedWhereTheyCannotBeReplayed(t *testing.T) {
	caps := full()
	caps.ToolResults = false
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	req := &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleTool, Content: []llm.Part{&llm.ToolResult{ID: "1", Content: []llm.Part{&llm.Text{Text: "42"}}}}},
	}}
	_, err = c.Chat(context.Background(), req)
	e := llm.AsError(err)
	if e == nil || e.Kind != llm.KindUnsupported {
		t.Fatalf("err = %v: a history whose tool calls went unanswered is a different conversation", err)
	}
	if e.Sent {
		t.Fatal("nothing was sent, so nothing was billed")
	}
}

func TestSystemPromptIsFoldedWhereThereIsNoChannelForIt(t *testing.T) {
	caps := full()
	caps.SystemPrompt = false
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model:    "m",
		System:   []llm.Part{&llm.Text{Text: "be brief"}},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	sent := p.seen.Request
	if len(sent.System) != 0 {
		t.Fatalf("system = %+v: the block has nowhere of its own to go", sent.System)
	}
	if len(sent.Messages) != 2 || llm.TextOf(sent.Messages[0].Content) != "be brief" {
		t.Fatalf("messages = %+v: the instruction must lead the conversation it governs", sent.Messages)
	}
	if !resp.Diagnostics.Has(llm.DropSystemPrompt) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestUnknownFieldsAreStrippedForABackendThatRefusesThem(t *testing.T) {
	caps := full()
	caps.Extensions = nil
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model:     "m",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
		Extra:     map[string]json.RawMessage{"safety_identifier": json.RawMessage(`"x"`)},
		ExtraFrom: "openai",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.seen.Request.Extra) != 0 {
		t.Fatalf("extra = %+v: a backend that refuses unknown fields refuses the whole request", p.seen.Request.Extra)
	}
	if !resp.Diagnostics.Has(llm.DropExtraField) {
		t.Fatalf("diagnostics = %+v: a field that vanishes at parse is the failure this layer exists to stop", resp.Diagnostics)
	}
}

func TestThinkingSignaturesAreStrippedForAnotherVendor(t *testing.T) {
	caps := full()
	caps.ThinkingSignatures = false
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model: "m",
		Messages: []llm.Message{{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{Text: "because", Signature: "sig-from-another-vendor"},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := p.seen.Request.Messages[0].Content[0].(*llm.Thinking)
	if got.Signature != "" {
		t.Fatal("a signature means nothing to the backend that did not issue it")
	}
	if got.Text != "because" {
		t.Fatalf("thinking = %q: only the signature travels no further", got.Text)
	}
	if !resp.Diagnostics.Has(llm.DropSignature) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestRedactedThinkingIsDroppedWhereThereIsNoBlockForIt(t *testing.T) {
	caps := full()
	caps.RedactedThinking = false
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model: "m",
		Messages: []llm.Message{{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{Redacted: []byte("opaque")},
			&llm.Text{Text: "the answer"},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	content := p.seen.Request.Messages[0].Content
	if len(content) != 1 || llm.TextOf(content) != "the answer" {
		t.Fatalf("content = %+v: the block goes, the answer stays", content)
	}
	if !resp.Diagnostics.Has(llm.DropThinking) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestParallelToolHistoryIsReportedNotRemoved(t *testing.T) {
	caps := full()
	caps.ParallelTools = false
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model: "m",
		Messages: []llm.Message{{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.ToolUse{ID: "1", Name: "a"}, &llm.ToolUse{ID: "2", Name: "b"},
		}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.seen.Request.Messages[0].Content) != 2 {
		t.Fatal("only the caller can judge whether a history that ran two together matters")
	}
	if !resp.Diagnostics.Has(llm.CapParallelTools) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestRequestLargerThanTheBackendTakesIsRefused(t *testing.T) {
	caps := full()
	caps.MaxInputBytes = 32
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	long := make([]byte, 1024)
	for i := range long {
		long[i] = 'x'
	}
	_, err = c.Chat(context.Background(), &llm.Request{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: string(long)}}}},
	})
	if e := llm.AsError(err); e == nil || e.Kind != llm.KindUnsupported {
		t.Fatalf("err = %v: sending it would be paid for and refused", err)
	}
	if p.seen != nil {
		t.Fatal("the request must not reach the backend")
	}
}

func TestCacheHintsBeyondTheBudgetAreDropped(t *testing.T) {
	caps := full()
	caps.CacheBreakpoints = 2
	p := &capped{id: "one", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	hint := func() *llm.CacheHint { return &llm.CacheHint{} }
	resp, err := c.Chat(context.Background(), &llm.Request{
		Model:       "m",
		SystemCache: hint(),
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "a"}}, Cache: hint()},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "b"}}, Cache: hint()},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	sent := p.seen.Request
	kept := 0
	if sent.SystemCache != nil {
		kept++
	}
	for _, m := range sent.Messages {
		if m.Cache != nil {
			kept++
		}
	}
	if kept != 2 {
		t.Fatalf("breakpoints = %d, want 2: over the budget the backend refuses the whole request", kept)
	}
	if sent.SystemCache == nil {
		t.Fatal("the earliest breakpoint covers the longest stable prefix and is the one worth keeping")
	}
	if !resp.Diagnostics.Has(llm.DropCacheHint) {
		t.Fatalf("diagnostics = %+v", resp.Diagnostics)
	}
}

func TestAnAllowedTruncationIsReportedAndNotFailed(t *testing.T) {
	p := &stub{id: "one", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	}}
	c, err := New(
		WithProvider(p),
		WithLossPolicy(llm.LossPolicy{Allow: []string{llm.WireNoFinish}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for range s.Frames() {
	}
	if s.Err() != nil {
		t.Fatalf("err = %v: the operator chose to live with this backend", s.Err())
	}
	if !llm.Diagnostics(s.Diagnostics()).Has(llm.WireNoFinish) {
		t.Fatalf("diagnostics = %+v: living with it is not the same as not being told", s.Diagnostics())
	}
}

func TestATruncationIsAFailureByDefault(t *testing.T) {
	p := &stub{id: "one", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	}}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), &llm.Request{
		Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for range s.Frames() {
	}
	if s.Err() == nil {
		t.Fatal("a stream cut off mid-answer is not a whole one")
	}
}

func TestASessionThatCannotTravelIsReported(t *testing.T) {
	one := &capped{id: "one", caps: full()}
	two := &capped{id: "two", caps: full()}
	c, err := New(
		WithProvider(one), WithProvider(two),
		WithRoutes(route.Rule{Name: "r", Match: route.Always(), Chain: []route.Target{{Provider: "two"}}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req(),
		WithSession("sess_1"), WithSessionProvider("one"))
	if err != nil {
		t.Fatal(err)
	}
	if two.seen.Session != "" {
		t.Fatalf("session = %q: it was minted by another backend", two.seen.Session)
	}
	if !resp.Diagnostics.Has(llm.CapSessionElsewhere) {
		t.Fatalf("diagnostics = %v", resp.Diagnostics.Codes())
	}
}

func TestAMultiTurnBackendSaysWhatItsFigureCovers(t *testing.T) {
	caps := full()
	caps.Usage.MultiTurn = true
	p := &capped{id: "cli", caps: caps}
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Diagnostics.Has(llm.AcctMultiTurn) {
		t.Fatalf("diagnostics = %v", resp.Diagnostics.Codes())
	}
}

func TestACancelledCallStillLeavesARow(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p := &capped{id: "one", caps: full()}
	var rows int
	sink := usage.SinkFunc(func(ctx context.Context, _ usage.Record) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		rows++
		return nil
	})
	c, err := New(WithProvider(p), WithUsage(sink))
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	_, _ = c.Chat(ctx, req())
	if rows == 0 {
		t.Fatal("the call was billed and left no record")
	}
}

func TestABackendThatPricesAndDidNotSaysSo(t *testing.T) {
	caps := full()
	caps.Usage.Cost = true
	priced := &capped{id: "cli", caps: caps}
	c, err := New(WithProvider(priced))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if !hasAnomaly(resp.Usage, llm.AnomalyPriceMissing) {
		t.Fatalf("usage = %+v", resp.Usage)
	}

	plain := &capped{id: "http", caps: full()}
	c2, err := New(WithProvider(plain))
	if err != nil {
		t.Fatal(err)
	}
	resp2, err := c2.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if hasAnomaly(resp2.Usage, llm.AnomalyPriceMissing) {
		t.Fatalf("usage = %+v", resp2.Usage)
	}
}

func hasAnomaly(u llm.Usage, code string) bool {
	for _, a := range u.Anomalies {
		if a.Code == code {
			return true
		}
	}
	return false
}

func TestAFailedToolResultThatCannotTravelIsReported(t *testing.T) {
	caps := full()
	caps.ToolErrors = false
	p := &capped{id: "one", caps: caps}
	c, _ := New(WithProvider(p))

	r := req()
	r.Messages = append(r.Messages, llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.Part{&llm.ToolResult{ID: "t1", Content: []llm.Part{&llm.Text{Text: "boom"}}, IsError: true}},
	})
	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Diagnostics.Has(llm.DropToolError) {
		t.Fatalf("diagnostics = %+v: the failure flag went missing without a word", resp.Diagnostics)
	}

	caps.ToolErrors = true
	q := &capped{id: "two", caps: caps}
	c2, _ := New(WithProvider(q))
	resp, err = c2.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Diagnostics.Has(llm.DropToolError) {
		t.Fatalf("diagnostics = %+v: this backend has the field", resp.Diagnostics)
	}

	ok := req()
	ok.Messages = append(ok.Messages, llm.Message{
		Role:    llm.RoleUser,
		Content: []llm.Part{&llm.ToolResult{ID: "t1", Content: []llm.Part{&llm.Text{Text: "fine"}}}},
	})
	resp, err = c.Chat(context.Background(), ok)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Diagnostics.Has(llm.DropToolError) {
		t.Fatalf("diagnostics = %+v: nothing failed", resp.Diagnostics)
	}
}

func TestWhatTheCallersFormatCannotReadIsRefusedAtTheDoor(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, _ := New(WithProvider(p))
	one := llm.Renders{ToolCalls: true, Candidates: 1}

	r := req()
	r.Candidates = 3
	if _, err := c.Chat(context.Background(), r, WithRenders(one)); err == nil {
		t.Fatal("a format that carries one completion cannot be handed three")
	}
	if p.seen != nil {
		t.Fatal("the request reached the backend anyway")
	}

	r = req()
	r.Thinking = &llm.ThinkingConfig{Enabled: true}
	if _, err := c.Chat(context.Background(), r, WithRenders(one)); err == nil {
		t.Fatal("thinking a format cannot carry would be billed and not delivered")
	}

	r = req()
	r.Candidates = 3
	if _, err := c.Chat(context.Background(), r); err != nil {
		t.Fatalf("a caller that named no format was refused anyway: %v", err)
	}
}

func TestAToolChoiceNamingNoToolIsReported(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, _ := New(WithProvider(p))
	r := req()
	r.Tools = []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`)}}
	r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolSpecific}
	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Diagnostics.Has(llm.DropToolDefinition) {
		t.Fatalf("diagnostics = %+v: the choice went missing without a word", resp.Diagnostics)
	}
	if p.seen.Request.ToolChoice != nil {
		t.Fatalf("choice = %+v: it cannot be sent", p.seen.Request.ToolChoice)
	}

	r = req()
	r.Tools = []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`)}}
	r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolSpecific, Name: "f"}
	if _, err := c.Chat(context.Background(), r); err != nil {
		t.Fatal(err)
	}
	if p.seen.Request.ToolChoice == nil {
		t.Fatal("a choice that names a tool must reach the backend")
	}
}

// A backend that takes no tools cannot answer a request that offers them as
// well as one that can. While a later link takes them, the request is left to
// it rather than stripped here; a tool call the caller insists on is never
// stripped, since an answer without one is an answer to another question.
func TestToolsAreLeftToALinkThatCanOfferThem(t *testing.T) {
	cli := &capped{id: "cli", caps: llm.Capabilities{Streaming: true, SystemPrompt: true}}
	api := &capped{id: "api", caps: full()}
	c, err := New(
		WithProvider(cli), WithProvider(api),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "cli"}, {Provider: "api"},
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, choice := range []*llm.ToolChoice{
		{Mode: llm.ToolSpecific, Name: "act"}, {Mode: llm.ToolRequired}, {Mode: llm.ToolAuto}, nil,
	} {
		cli.seen, api.seen = nil, nil
		r := req()
		r.Tools = []llm.Tool{{Name: "act", Schema: []byte(`{"type":"object"}`)}}
		r.ToolChoice = choice
		resp, err := c.Chat(context.Background(), r)
		if err != nil {
			t.Fatalf("choice %+v: %v", choice, err)
		}
		if cli.seen != nil || resp.Provider != "api" {
			t.Fatalf("choice %+v: served by %q with the tools stripped, and the link that has them was not tried",
				choice, resp.Provider)
		}
		if len(api.seen.Request.Tools) != 1 {
			t.Fatalf("choice %+v: tools = %+v", choice, api.seen.Request.Tools)
		}
	}
}

func TestAToolCallTheCallerInsistsOnIsNotStripped(t *testing.T) {
	cli := &capped{id: "cli", caps: llm.Capabilities{Streaming: true}}
	c, err := New(WithProvider(cli))
	if err != nil {
		t.Fatal(err)
	}
	r := req()
	r.Tools = []llm.Tool{{Name: "act", Schema: []byte(`{"type":"object"}`)}}
	r.ToolChoice = &llm.ToolChoice{Mode: llm.ToolSpecific, Name: "act"}
	_, err = c.Chat(context.Background(), r)
	if e := llm.AsError(err); e == nil || e.Kind != llm.KindUnsupported || e.Sent {
		t.Fatalf("err = %v: a backend with no tools cannot make the call the caller insists on", err)
	}
	if cli.seen != nil {
		t.Fatal("the request reached the backend anyway")
	}
}

func TestAToolServerTheBackendCannotDialIsReported(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	c, _ := New(WithProvider(p))
	r := req()
	r.MCPServers = []llm.MCPServer{{Name: "docs", URL: "https://tools.test/mcp"}}
	resp, err := c.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Diagnostics.Has(llm.DropMCPServer) {
		t.Fatalf("diagnostics = %+v: the tools went missing without a word", resp.Diagnostics)
	}
	if len(p.seen.Request.MCPServers) != 0 {
		t.Fatalf("servers = %+v: a backend that cannot use them must not be sent them", p.seen.Request.MCPServers)
	}
	if len(r.MCPServers) != 1 {
		t.Fatalf("the caller's request was written through: %+v", r.MCPServers)
	}

	caps := full()
	caps.MCPServers = true
	q := &capped{id: "two", caps: caps}
	c2, _ := New(WithProvider(q))
	resp, err = c2.Chat(context.Background(), r)
	if err != nil {
		t.Fatal(err)
	}
	if resp.Diagnostics.Has(llm.DropMCPServer) {
		t.Fatalf("diagnostics = %+v: a backend that dials them loses nothing", resp.Diagnostics)
	}
	if len(q.seen.Request.MCPServers) != 1 {
		t.Fatalf("servers = %+v", q.seen.Request.MCPServers)
	}
}
