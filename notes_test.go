// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/usage"
)

// building notes what building its request lost and hands it back on its
// answer as well, as every adapter does for a caller of its own, and answers
// only unstreamed.
type building struct{}

func (building) ID() string                     { return "building" }
func (building) Capabilities() llm.Capabilities { return llm.Capabilities{} }

func (building) lost() llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.CapCeilingSupplied, Class: llm.ClassCapability, Stage: llm.StageRequest,
		Severity: llm.SeverityLoss, Provider: "building", Field: "max_tokens",
		Detail: "this backend needs an output ceiling, so the default was sent instead",
	}
}

func (b building) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	a.Note(b.lost())
	return &llm.Response{
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Text{Text: "ok"},
		}}, Stop: llm.StopEnd}},
		Diagnostics: llm.Diagnostics{b.lost()},
	}, nil
}

func (building) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return nil, llm.Errorf(llm.KindUnsupported, "this backend does not stream")
}

// What an adapter noted while building the request reaches the chain twice:
// on the attempt, weighed by the caller's policy, and on the answer as the
// adapter made it. The answer says it once, as the policy has it, so a loss the
// policy allowed does not read as one, and a stream made of the answer, and
// its row, say it once too.
func TestWhatAnAdapterNotedBuildingTheRequestIsSaidOnce(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if strings.HasSuffix(r.URL.Path, "/messages") {
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x",`+
				`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"c","model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()
	claude, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	gpt, err := openai.New(openai.Options{ID: "gpt", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	unbounded := func() *llm.Request {
		r := req()
		r.MaxOutputTokens = new(int)
		return r
	}
	sampled := func() *llm.Request {
		r := req()
		k := 3
		r.TopK = &k
		return r
	}
	for _, tc := range []struct {
		p       llm.Provider
		request func() *llm.Request
		code    string
	}{
		{claude, unbounded, llm.CapCeilingSupplied},
		{gpt, sampled, llm.DropExtraField},
	} {
		c, err := New(WithProvider(tc.p), WithLossPolicy(llm.LossPolicy{Allow: []string{tc.code}}))
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.Chat(context.Background(), tc.request())
		if err != nil {
			t.Fatal(err)
		}
		if got := resp.Diagnostics.Find(tc.code); len(got) != 1 || resp.Diagnostics.Worst() != llm.SeverityInfo {
			t.Fatalf("%s: diagnostics = %+v: said other than once, or as a loss the policy allowed", tc.p.ID(), resp.Diagnostics)
		}
	}

	for _, stream := range []bool{false, true} {
		mem := usage.NewMemory()
		c, err := New(WithProvider(building{}), WithUsage(mem),
			WithLossPolicy(llm.LossPolicy{Allow: []string{llm.CapCeilingSupplied}}))
		if err != nil {
			t.Fatal(err)
		}
		answer, row := said(t, c, mem, req(), stream)
		if n, m := len(answer.Find(llm.CapCeilingSupplied)), len(row.Find(llm.CapCeilingSupplied)); n != 1 || m != 1 || answer.Worst() != llm.SeverityInfo {
			t.Fatalf("stream %v: said %d times on the answer, %d on the row: %+v", stream, n, m, answer)
		}
	}
}

// folding answers by folding a stream of its own, handing the fold the
// attempt's recorder as an agent CLI does, and the stream runs past the buffer
// the fold is given. cut ends it in a failure rather than a finish.
type folding struct{ cut bool }

func (folding) ID() string                     { return "folding" }
func (folding) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (f folding) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	s, _ := f.Stream(ctx, a)
	return wirecore.Fold(s, wirecore.FoldOptions{MaxBufferBytes: 8, Recorder: a.Recorder})
}

func (f folding) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	last := &llm.Frame{Stop: llm.StopEnd}
	if f.cut {
		last = &llm.Frame{Err: &llm.Error{Kind: llm.KindUpstream, Provider: "folding", Sent: true, Message: "cut off"}}
	}
	return llm.StaticStream(llm.StreamMeta{Provider: "folding"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "more than the buffer holds"}}}, last,
	}), nil
}

// What a fold found itself, such as the buffer filling, goes to the recorder it
// was handed or else onto the answer, never both: the chain reads the two
// together. The row of a call that failed still says it.
func TestWhatAFoldFoundIsSaidOnce(t *testing.T) {
	c, err := New(WithProvider(folding{}))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if n := len(resp.Diagnostics.Find(llm.WireBufferFull)); n != 1 {
		t.Fatalf("the buffer filling is said %d times", n)
	}

	mem := usage.NewMemory()
	if c, err = New(WithProvider(folding{cut: true}), WithUsage(mem)); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Chat(context.Background(), req()); err == nil {
		t.Fatal("the stream was cut off")
	}
	if rows := mem.Records(); len(rows) != 1 || len(llm.Diagnostics(rows[0].Diags).Find(llm.WireBufferFull)) != 1 {
		t.Fatalf("rows = %+v: the failed call's row lost what the fold found", rows)
	}
}
