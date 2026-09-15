// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

func TestSaturationIsReportedAsAPolicyDecision(t *testing.T) {
	p := &gated{capped: capped{id: "one", caps: full()}, slots: make(chan struct{})}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err == nil {
		t.Fatal("a shed request is not a served one")
	}
	rows := mem.Records()
	if len(rows) != 1 || rows[0].Sent {
		t.Fatalf("records = %+v: nothing was sent, so nothing was billed", rows)
	}
	if !llm.Diagnostics(rows[0].Diags).Has(llm.PolicySaturated) {
		t.Fatalf("diagnostics = %+v", rows[0].Diags)
	}
}

func TestAbandonedStreamIsStillBooked(t *testing.T) {
	p := &capped{id: "one", caps: full()}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	c.abandon = 20 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := c.Stream(ctx, &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err != nil {
		t.Fatal(err)
	}
	cancel()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if rows := mem.Records(); len(rows) == 1 {
			if !rows[0].Sent || rows[0].Outcome != llm.OutcomeCancelled {
				t.Fatalf("row = %+v", rows[0])
			}
			if rows[0].Usage.Reported {
				t.Fatal("what an abandoned stream cost is not known, and a zero that says so is not a measurement")
			}
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("records = %+v: a delivered call with an unknown cost is still a delivered call", mem.Records())
}

func TestAClosedStreamBooksTheFiguresRatherThanTheFallback(t *testing.T) {
	p := &stub{id: "one", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "one"}}},
		{Stop: llm.StopEnd},
		{Usage: &llm.Usage{Prompt: 7, Completion: 3, Reported: true}, TurnEnd: true},
	}}
	mem := usage.NewMemory()
	c, err := New(WithProvider(p), WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	c.abandon = time.Second
	ctx, cancel := context.WithCancel(context.Background())
	s, err := c.Stream(ctx, &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}})
	if err != nil {
		t.Fatal(err)
	}
	for range s.Frames() {
	}
	cancel()
	_ = s.Close()

	time.Sleep(2 * time.Second)
	rows := mem.Records()
	if len(rows) != 1 {
		t.Fatalf("records = %d, want 1: the call is booked once", len(rows))
	}
	if rows[0].Usage.Prompt != 7 || rows[0].Usage.Completion != 3 {
		t.Fatalf("usage = %+v: the row that knows the figures is the one to keep", rows[0].Usage)
	}
}

func TestAStreamThatFailsAfterItsFirstFrameIsNotFailedOver(t *testing.T) {
	boom := &llm.Error{Kind: llm.KindUpstream, Provider: "first", Message: "cut off", Sent: true, Retryable: true}
	first := &stub{id: "first", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half"}}},
		{Err: boom},
	}}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
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
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if s.Err() == nil {
		t.Fatal("a stream cut off mid-answer is not a whole one")
	}
	if text != "half" {
		t.Fatalf("text = %q: the client reads the stream it was given", text)
	}
	if second.calls != 0 {
		t.Fatal("the second link was tried after the client had started reading the first")
	}
}

func TestAStreamThatFailsBeforeItsFirstFrameIsFailedOver(t *testing.T) {
	first := &stub{id: "first", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "first", Message: "busy", Sent: true, Retryable: true,
	}}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	c, err := New(
		WithProvider(first), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "first"}, {Provider: "second"},
		}}),
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
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if text != "whole" {
		t.Fatalf("text = %q: nothing had been written, so the next link could still serve", text)
	}
	if s.Meta().Provider != "second" {
		t.Fatalf("provider = %q", s.Meta().Provider)
	}
}
