// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/agentcli"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/bedrock"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/ollama"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

// foreign carries a block from a wire none of these backends speaks, which
// every adapter notes as a loss while it builds its request.
func foreign() *llm.Request {
	return &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{
		&llm.Text{Text: "hi"},
		&llm.Raw{Vendor: "elsewhere", Type: "mystery", JSON: json.RawMessage(`{"type":"mystery"}`)},
	}}}}
}

// A loss an adapter finds while building its request is known before anything
// is sent, so a policy that will not take it refuses it there, unsent.
func TestEveryBackendRefusesALossItFindsWhileBuildingBeforeSending(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, `{"error":{"message":"this should never have been sent"}}`, http.StatusInternalServerError)
	}))
	defer srv.Close()
	dir := t.TempDir()
	ran := filepath.Join(dir, "ran")
	script := filepath.Join(dir, "agent.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\ntouch '"+ran+"'\n"), 0o755); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name   string
		format string
		build  func() (llm.Provider, error)
	}{
		{"anthropic", "", func() (llm.Provider, error) {
			return anthropic.New(anthropic.Options{ID: "anthropic", BaseURL: srv.URL, APIKey: "k"})
		}},
		{"openai chat", "", func() (llm.Provider, error) {
			return openai.New(openai.Options{ID: "openai", BaseURL: srv.URL, APIKey: "k"})
		}},
		{"openai responses", "openai-responses", func() (llm.Provider, error) {
			return openai.New(openai.Options{ID: "openai", BaseURL: srv.URL, APIKey: "k", Dialect: openai.OpenAIDialect()})
		}},
		{"google", "", func() (llm.Provider, error) {
			return google.New(google.Options{ID: "google", BaseURL: srv.URL, APIKey: "k"})
		}},
		{"bedrock", "", func() (llm.Provider, error) {
			return bedrock.New(bedrock.Options{Region: "us-east-1", BaseURL: srv.URL,
				Credentials: func(context.Context) (bedrock.Credentials, error) {
					return bedrock.Credentials{AccessKeyID: "AK", SecretAccessKey: "SK"}, nil
				}})
		}},
		{"ollama", "", func() (llm.Provider, error) {
			return ollama.New(ollama.Options{ID: "ollama", BaseURL: srv.URL})
		}},
		{"agentcli", "", func() (llm.Provider, error) {
			return agentcli.New(agentcli.Options{Kind: agentcli.Claude, Binary: script})
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := tc.build()
			if err != nil {
				t.Fatal(err)
			}
			if c, ok := p.(llm.Closer); ok {
				t.Cleanup(func() { _ = c.Close(context.Background()) })
			}
			hits.Store(0)
			for _, stream := range []bool{false, true} {
				var noted llm.Diagnostics
				a := &llm.Attempt{
					Request: foreign(), Provider: p.ID(), Model: "m", Format: tc.format, Stream: stream,
					Policy: llm.LossPolicy{Strict: true}, Recorder: &noted,
				}
				if stream {
					_, err = p.Stream(context.Background(), a)
				} else {
					_, err = p.Complete(context.Background(), a)
				}
				if e := llm.AsError(err); e == nil || e.Kind != llm.KindLoss || e.Sent {
					t.Fatalf("stream %v: err = %v: a loss known before sending was not refused there", stream, err)
				}
				if !noted.Has(llm.DropUnknownBlock) {
					t.Fatalf("stream %v: noted = %v: the refused loss is not on the attempt's record", stream, noted.Codes())
				}
			}
			if n := hits.Load(); n != 0 {
				t.Fatalf("the backend was sent %d requests it was always going to have refused", n)
			}
			if _, err := os.Stat(ran); err == nil {
				t.Fatal("the agent was run for a request it was always going to have refused")
			}
		})
	}
}

// The chain treats such a refusal as the rejection it is: nothing was paid for,
// the breaker holds nothing against the backend, and the next link, which may
// carry the request whole, is tried.
func TestALossFoundWhileBuildingIsRefusedUnsentAndTheChainWalksOn(t *testing.T) {
	named := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Name: "alice", Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	ownField := func() *llm.Request {
		r := req()
		r.Extra = map[string]json.RawMessage{"service_tier": json.RawMessage(`"standard_only"`)}
		r.ExtraFrom = "anthropic"
		return r
	}
	for _, tc := range []struct {
		what   string
		req    func() *llm.Request
		stream bool
	}{
		{"a participant's name, unstreamed", named, false},
		{"a participant's name, streamed", named, true},
		{"a field of the backend's own wire the operator keeps back, unstreamed", ownField, false},
		{"a field of the backend's own wire the operator keeps back, streamed", ownField, true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				http.Error(w, `{"type":"error","error":{"type":"api_error","message":"no"}}`, http.StatusInternalServerError)
			}))
			defer srv.Close()
			claude, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: srv.URL, APIKey: "k", NoClientBetas: true})
			if err != nil {
				t.Fatal(err)
			}
			wide := &capped{id: "wide", caps: full()}
			mem := usage.NewMemory()
			c, err := New(
				WithProvider(claude), WithProvider(wide),
				WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
					{Provider: "claude"}, {Provider: "wide"},
				}}),
				WithLossPolicy(llm.LossPolicy{Strict: true}),
				WithUsage(mem),
			)
			if err != nil {
				t.Fatal(err)
			}
			if tc.stream {
				s, err := c.Stream(context.Background(), tc.req())
				if err != nil {
					t.Fatal(err)
				}
				for range s.Frames() {
				}
				_ = s.Close()
				if s.Err() != nil || s.Meta().Provider != "wide" {
					t.Fatalf("served by %q, err = %v", s.Meta().Provider, s.Err())
				}
			} else if resp, err := c.Chat(context.Background(), tc.req()); err != nil || resp.Provider != "wide" {
				t.Fatalf("err = %v: the next link was not given the request", err)
			}
			if n := hits.Load(); n != 0 {
				t.Fatalf("claude was sent the request %d times and paid for it", n)
			}
			rows := mem.Records()
			if len(rows) == 0 || rows[0].Provider != "claude" || rows[0].Sent || rows[0].Outcome != llm.OutcomeRejected {
				t.Fatalf("rows = %+v: the refusal is a rejected attempt that cost nothing", rows)
			}
		})
	}
}

// whole answers only unstreamed, with a loss noted on the answer, so a stream
// of it is synthesised by the router.
type whole struct{ id string }

func (w *whole) ID() string                     { return w.id }
func (w *whole) Capabilities() llm.Capabilities { return llm.Capabilities{} }

func (w *whole) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	return &llm.Response{
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Text{Text: "the whole answer"},
		}}, Stop: llm.StopEnd}},
		Diagnostics: llm.Diagnostics{{
			Code: llm.DropImage, Class: llm.ClassContent, Stage: llm.StageResponse,
			Severity: llm.SeverityLoss, Detail: "an image the answer carried did not survive",
		}},
	}, nil
}

func (w *whole) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return nil, llm.Errorf(llm.KindUnsupported, "this backend does not stream")
}

// A backend that does not stream has handed over its whole answer before any of
// it is delivered, so a loss in it is refused before the first frame, as the
// unstreamed call refuses it, rather than after the last.
func TestAWholeAnswerServedAsAStreamIsRefusedBeforeAnyOfItIsDelivered(t *testing.T) {
	c, err := New(WithProvider(&whole{id: "whole"}), WithLossPolicy(llm.LossPolicy{Strict: true}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req()); llm.AsError(err) == nil || llm.AsError(err).Kind != llm.KindLoss {
		t.Fatalf("err = %v", err)
	}
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	var text string
	for f := range s.Frames() {
		text += llm.TextOf(f.Delta)
	}
	if e := llm.AsError(s.Err()); text != "" || e == nil || e.Kind != llm.KindLoss {
		t.Fatalf("delivered %q, err = %v: the refused answer was handed over before it was refused", text, s.Err())
	}
}

// carrying answers with a block it carried verbatim, as the Anthropic adapter
// does with a server tool's result, and text after it.
type carrying struct{}

func carried() (*llm.Raw, llm.Diagnostic) {
	return &llm.Raw{Vendor: "anthropic", Type: "web_search_tool_result", JSON: json.RawMessage(`{"type":"web_search_tool_result"}`)},
		llm.Diagnostic{
			Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageResponse,
			Severity: llm.SeverityInfo, Detail: "content block carried verbatim",
		}
}

func (carrying) ID() string                     { return "carrying" }
func (carrying) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (carrying) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	raw, note := carried()
	return &llm.Response{
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			raw, &llm.Text{Text: "per the result"},
		}}, Stop: llm.StopEnd}},
		Diagnostics: llm.Diagnostics{note},
	}, nil
}

func (carrying) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	raw, note := carried()
	return llm.StaticStream(llm.StreamMeta{Provider: "carrying"}, []*llm.Frame{
		{Role: llm.RoleAssistant},
		{Delta: []llm.Part{raw}, Diagnostics: llm.Diagnostics{note}},
		{Delta: []llm.Part{&llm.Text{Text: "per the result"}}},
		{Stop: llm.StopEnd},
	}), nil
}

// A block carried verbatim has lost nothing only where it can be written out.
// Strict refuses it for a destination that has no shape for it, and serves it
// to a caller that takes parts as they are.
func TestStrictRefusesACarriedBlockItsDestinationCannotWriteOut(t *testing.T) {
	c, err := New(WithProvider(carrying{}), WithLossPolicy(llm.LossPolicy{Strict: true}))
	if err != nil {
		t.Fatal(err)
	}
	chatWire := WithRenders(llm.Renders{ToolCalls: true, Candidates: 128})

	if _, err := c.Chat(context.Background(), req(), chatWire); llm.AsError(err) == nil || llm.AsError(err).Kind != llm.KindLoss {
		t.Fatalf("err = %v: an answer whose block the destination drops was served as whole", err)
	}
	s, err := c.Stream(context.Background(), req(), chatWire)
	if err != nil {
		t.Fatal(err)
	}
	var n int
	for range s.Frames() {
		n++
	}
	_ = s.Close()
	if e := llm.AsError(s.Err()); n != 0 || e == nil || e.Kind != llm.KindLoss {
		t.Fatalf("frames = %d, err = %v: the stream was handed over with a block its destination drops", n, s.Err())
	}

	resp, err := c.Chat(context.Background(), req())
	if err != nil || len(resp.Choices) == 0 || len(resp.Choices[0].Message.Content) != 2 {
		t.Fatalf("resp = %+v, err = %v: a library caller takes the block as it is", resp, err)
	}
	s, err = c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	n = 0
	for range s.Frames() {
		n++
	}
	_ = s.Close()
	if n != 4 || s.Err() != nil {
		t.Fatalf("frames = %d, err = %v", n, s.Err())
	}
}

// A loss refused in an answer is refused after the request went and was billed.
// The failure says so, and whose it was, so a caller reading only the failure
// does not take the call for one that cost nothing.
func TestALossRefusedInTheAnswerSaysTheRequestWent(t *testing.T) {
	for _, tc := range []struct {
		p    llm.Provider
		opts []CallOption
	}{
		{carrying{}, []CallOption{WithRenders(llm.Renders{ToolCalls: true, Candidates: 128})}},
		{&whole{id: "whole"}, nil},
	} {
		c, err := New(WithProvider(tc.p), WithLossPolicy(llm.LossPolicy{Strict: true}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.Chat(context.Background(), req(), tc.opts...)
		e := llm.AsError(err)
		if e == nil || e.Kind != llm.KindLoss || !e.Sent || e.Provider != tc.p.ID() || e.Model == "" || resp == nil {
			t.Fatalf("%s: err = %+v: a refusal of an answer that was sent for reads as though nothing was", tc.p.ID(), e)
		}
	}
}

// answering answers with parts its destination may have no field for,
// streamed part by part or, when it does not stream, whole.
type answering struct {
	streams bool
	parts   []llm.Part
}

func (answering) ID() string                       { return "answering" }
func (a answering) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: a.streams} }

func (a answering) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	return &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: a.parts}, Stop: llm.StopEnd,
	}}}, nil
}

func (a answering) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	frames := []*llm.Frame{{Role: llm.RoleAssistant}}
	for _, p := range a.parts {
		frames = append(frames, &llm.Frame{Delta: []llm.Part{p}})
	}
	return llm.StaticStream(llm.StreamMeta{Provider: "answering"}, append(frames, &llm.Frame{Stop: llm.StopEnd})), nil
}

// An image or a tool call the destination has no field for is a loss like a
// block it cannot write out, and strict refuses it before anything is handed
// over: a whole answer served as a stream is delivered whole or not at all,
// and a stream that opens with one is refused as a call, not as a stream that
// opened and then failed.
func TestStrictRefusesAnImageOrAToolCallItsDestinationCannotCarry(t *testing.T) {
	img := &llm.Image{MIME: "image/png", Data: []byte("png")}
	call := &llm.ToolUse{ID: "1", Name: "act", Args: json.RawMessage(`{}`)}
	chat := llm.Renders{ToolCalls: true, Candidates: 128}
	generate := llm.Renders{Thinking: true, Candidates: 1}
	for _, tc := range []struct {
		what string
		p    answering
		to   llm.Renders
	}{
		{"an image after text, answered whole", answering{parts: []llm.Part{&llm.Text{Text: "here"}, img}}, chat},
		{"a tool call after text, answered whole", answering{parts: []llm.Part{&llm.Text{Text: "here"}, call}}, generate},
		{"a stream that opens with an image", answering{streams: true, parts: []llm.Part{img, &llm.Text{Text: "here"}}}, chat},
	} {
		c, err := New(WithProvider(tc.p), WithLossPolicy(llm.LossPolicy{Strict: true}))
		if err != nil {
			t.Fatal(err)
		}
		s, err := c.Stream(context.Background(), req(), WithRenders(tc.to))
		if err != nil {
			t.Fatal(err)
		}
		var text string
		for f := range s.Frames() {
			text += llm.TextOf(f.Delta)
		}
		_ = s.Close()
		if e := llm.AsError(s.Err()); text != "" || s.Committed() || e == nil || e.Kind != llm.KindLoss {
			t.Fatalf("%s: delivered %q, err = %v: what the destination cannot carry was found only after part of the answer went",
				tc.what, text, s.Err())
		}
	}
}

// A whole answer served as a stream names the backend that served it and the
// model the caller asked for, as the unstreamed answer does, so a refusal of
// it does not read as nobody's.
func TestAWholeAnswerServedAsAStreamNamesItsBackend(t *testing.T) {
	c, err := New(WithProvider(&whole{id: "whole"}), WithLossPolicy(llm.LossPolicy{Strict: true}))
	if err != nil {
		t.Fatal(err)
	}
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	for range s.Frames() {
	}
	_ = s.Close()
	if m := s.Meta(); m.Provider != "whole" || m.Requested != req().Model {
		t.Fatalf("meta = %+v", m)
	}
	if e := llm.AsError(s.Err()); e == nil || e.Kind != llm.KindLoss || e.Provider != "whole" {
		t.Fatalf("err = %+v: the refusal names no backend", e)
	}
}

// emptied answers only unstreamed, with no candidate: only what the call cost,
// or only why the backend stopped it.
type emptied struct {
	id    string
	block bool
}

func (e *emptied) ID() string                     { return e.id }
func (e *emptied) Capabilities() llm.Capabilities { return llm.Capabilities{} }

func (e *emptied) Complete(context.Context, *llm.Attempt) (*llm.Response, error) {
	if e.block {
		return &llm.Response{Block: &llm.Block{Reason: "safety"}, Usage: llm.Usage{Prompt: 3, Reported: true}}, nil
	}
	return &llm.Response{Usage: llm.Usage{Prompt: 3, Reported: true}}, nil
}

func (e *emptied) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return nil, llm.Errorf(llm.KindUnsupported, "this backend does not stream")
}

// An answer with no candidate is a success unstreamed, and served as a stream
// it is the same success: not a stream cut off, held against the backend and
// sent, and billed, again to the next link.
func TestAnAnswerWithNoCandidateIsServedAsAStreamToo(t *testing.T) {
	for _, block := range []bool{false, true} {
		first := &emptied{id: "first", block: block}
		second := &stub{id: "second"}
		c := pair(t, first, second, WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}))
		if _, err := c.Chat(context.Background(), req()); err != nil {
			t.Fatal(err)
		}
		s, err := c.Stream(context.Background(), req())
		if err != nil {
			t.Fatal(err)
		}
		for range s.Frames() {
		}
		_ = s.Close()
		if s.Err() != nil || s.Meta().Provider != "first" || second.calls != 0 {
			t.Fatalf("block %v: err = %v, served by %q: a whole answer was read as one cut off", block, s.Err(), s.Meta().Provider)
		}
		if _, ok := c.breaker.allow("first"); !ok {
			t.Fatalf("block %v: the backend that answered was taken out of the chain", block)
		}
	}
}
