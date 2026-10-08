// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func idx(i int) *int { return &i }

func fold(t *testing.T, frames ...*llm.Frame) *llm.Response {
	t.Helper()
	s := llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "m"}, frames)
	resp, err := Fold(s, FoldOptions{})
	if err != nil {
		t.Fatalf("fold: %v", err)
	}
	return resp
}

func parts(r *llm.Response) []llm.Part {
	if len(r.Choices) == 0 {
		return nil
	}
	return r.Choices[0].Message.Content
}

func TestFoldKeepsRefusal(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.Refusal{Text: "I can't help with that."}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	if got := llm.RefusalOf(parts(r)); got != "I can't help with that." {
		t.Fatalf("refusal = %q", got)
	}
}

func TestFoldMergesParallelToolCallsByIndex(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: idx(0), ID: "call_a", Name: "search", Args: json.RawMessage(`{"q":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: idx(1), ID: "call_b", Name: "fetch", Args: json.RawMessage(`{"url":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: idx(0), Args: json.RawMessage(`"go"}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{Index: idx(1), Args: json.RawMessage(`"x"}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	tools := llm.ToolUses(parts(r))
	if len(tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(tools))
	}
	if string(tools[0].Args) != `{"q":"go"}` || string(tools[1].Args) != `{"url":"x"}` {
		t.Fatalf("args = %q / %q", tools[0].Args, tools[1].Args)
	}
}

func TestFoldSeparatesIndexlessToolCalls(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "call_a", Name: "one", Args: json.RawMessage(`{"a":1}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "call_b", Name: "two", Args: json.RawMessage(`{"b":2}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	tools := llm.ToolUses(parts(r))
	if len(tools) != 2 || string(tools[0].Args) != `{"a":1}` || string(tools[1].Args) != `{"b":2}` {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestFoldKeepsPerBlockSignatures(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "first"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "sig-1"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "answer"}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	th, ok := parts(r)[0].(*llm.Thinking)
	if !ok || th.Signature != "sig-1" || th.Text != "first" {
		t.Fatalf("thinking = %+v", parts(r)[0])
	}
}

// Who minted a signature decides which backend it may be replayed to, so the
// folded answer says so as the deltas did, and a reasoning item keeps the name
// its backend gave it and stays apart from the next one.
func TestFoldKeepsWhoSignedAThought(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "plan"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "EqQB", Signer: llm.FamilyAnthropic}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{ID: "rs_a", Text: "why"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{ID: "rs_b", Text: "how"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{ID: "rs_b", Signature: "enc", Signer: llm.FamilyOpenAI}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	got := parts(r)
	if len(got) != 3 {
		t.Fatalf("parts = %+v", got)
	}
	for i, want := range []llm.Thinking{
		{Text: "plan", Signature: "EqQB", Signer: llm.FamilyAnthropic},
		{ID: "rs_a", Text: "why"},
		{ID: "rs_b", Text: "how", Signature: "enc", Signer: llm.FamilyOpenAI},
	} {
		th, ok := got[i].(*llm.Thinking)
		if !ok || th.ID != want.ID || th.Text != want.Text || th.Signature != want.Signature || th.Signer != want.Signer {
			t.Fatalf("part %d = %+v, want %+v", i, got[i], want)
		}
	}
}

func TestFoldAttachesLateSignatureToClosedBlock(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "reasoned"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "hello"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Signature: "late"}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	th := parts(r)[0].(*llm.Thinking)
	if th.Signature != "late" {
		t.Fatalf("signature = %q, want it attached to the closed block", th.Signature)
	}
	if len(parts(r)) != 2 {
		t.Fatalf("parts = %d, want thinking + text", len(parts(r)))
	}
}

func TestFoldKeepsRedactedThinkingSeparate(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "open"}}},
		&llm.Frame{Delta: []llm.Part{&llm.Thinking{Redacted: []byte("opaque")}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	if len(parts(r)) != 2 {
		t.Fatalf("parts = %d: redacted thinking must not merge into a text block", len(parts(r)))
	}
}

func TestFoldKeepsAnonymousImagesApart(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.Image{MIME: "image/png", Data: []byte("aa"), Partial: true}}},
		&llm.Frame{Delta: []llm.Part{&llm.Image{Data: []byte("bb")}}},
		&llm.Frame{Delta: []llm.Part{&llm.Image{MIME: "image/png", Data: []byte("cc")}}},
		&llm.Frame{Stop: llm.StopEnd},
	)
	var imgs []*llm.Image
	for _, p := range parts(r) {
		if im, ok := p.(*llm.Image); ok {
			imgs = append(imgs, im)
		}
	}
	if len(imgs) != 2 {
		t.Fatalf("images = %d, want 2", len(imgs))
	}
	if string(imgs[0].Data) != "bb" || string(imgs[1].Data) != "cc" {
		t.Fatalf("image data = %q / %q", imgs[0].Data, imgs[1].Data)
	}
	if imgs[0].Partial || imgs[1].Partial {
		t.Fatal("a folded image is whole")
	}
}

func TestFoldMarksMissingFinishReason(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half an ans"}}},
	})
	r, err := Fold(s, FoldOptions{})
	if err == nil {
		t.Fatal("fold: a truncated stream must not fold into a complete answer")
	}
	if e := llm.AsError(err); e.Code != llm.WireNoFinish {
		t.Fatalf("error = %+v, want %s", e, llm.WireNoFinish)
	}
	if r.Choices[0].Stop != llm.StopUnknown {
		t.Fatalf("stop = %q: a truncated stream must not read as complete", r.Choices[0].Stop)
	}
	if !hasCode(r.Diagnostics, llm.WireNoFinish) {
		t.Fatalf("diagnostics = %+v", r.Diagnostics)
	}
}

func TestFoldKeepsPromptBlockWithoutInventingAChoice(t *testing.T) {
	r := fold(t, &llm.Frame{Block: &llm.Block{Reason: "SAFETY"}, Stop: llm.StopContentFilter})
	if len(r.Choices) != 0 {
		t.Fatalf("choices = %d: a blocked prompt has no candidate", len(r.Choices))
	}
	if r.Block == nil || r.Stop != llm.StopContentFilter {
		t.Fatalf("response = %+v", r)
	}
}

func TestFoldReturnsPartialWithError(t *testing.T) {
	boom := errors.New("connection reset")
	i := 0
	s := llm.NewStream(llm.StreamMeta{}, func() (*llm.Frame, error) {
		i++
		switch i {
		case 1:
			return &llm.Frame{Delta: []llm.Part{&llm.Text{Text: "partial"}}}, nil
		case 2:
			return &llm.Frame{Usage: &llm.Usage{Prompt: 90, Completion: 3, Reported: true}}, nil
		default:
			return nil, boom
		}
	}, nil)
	resp, err := Fold(s, FoldOptions{})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v", err)
	}
	if llm.TextOf(parts(resp)) != "partial" {
		t.Fatal("generated text was thrown away with the error")
	}
	if resp.Usage.Prompt != 90 {
		t.Fatalf("usage = %+v: the failed call was still billed", resp.Usage)
	}
}

func TestFoldSurfacesInBandError(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "start"}}},
		{Err: &llm.Error{Kind: llm.KindOverloaded, Message: "overloaded"}},
	})
	resp, err := Fold(s, FoldOptions{})
	if err == nil {
		t.Fatal("a 200 carrying an error body must not fold into an empty success")
	}
	if llm.TextOf(parts(resp)) != "start" {
		t.Fatal("partial content lost")
	}
}

func TestFoldCapsAccumulation(t *testing.T) {
	big := make([]byte, 0, 4096)
	for len(big) < 4096 {
		big = append(big, 'x')
	}
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: string(big)}}},
		{Delta: []llm.Part{&llm.Text{Text: string(big)}}},
		{Stop: llm.StopEnd},
	})
	resp, err := Fold(s, FoldOptions{MaxBufferBytes: 5000})
	if err != nil {
		t.Fatal(err)
	}
	if !hasCode(resp.Diagnostics, llm.WireBufferFull) {
		t.Fatal("an accumulation cap that drops content must say so")
	}
	if n := len(llm.TextOf(parts(resp))); n != 4096 {
		t.Fatalf("kept %d bytes, want the first frame intact", n)
	}
}

func TestFoldKeepsSeveralChoices(t *testing.T) {
	r := fold(t,
		&llm.Frame{Choice: idx(0), Delta: []llm.Part{&llm.Text{Text: "first"}}},
		&llm.Frame{Choice: idx(1), Delta: []llm.Part{&llm.Text{Text: "second"}}},
		&llm.Frame{Choice: idx(0), Stop: llm.StopEnd},
		&llm.Frame{Choice: idx(1), Stop: llm.StopEnd},
	)
	if len(r.Choices) != 2 {
		t.Fatalf("choices = %d, want 2", len(r.Choices))
	}
	if llm.TextOf(r.Choices[1].Message.Content) != "second" {
		t.Fatal("a second candidate was folded into the first")
	}
}

func hasCode(ds []llm.Diagnostic, code string) bool {
	for _, d := range ds {
		if d.Code == code {
			return true
		}
	}
	return false
}

func TestFoldContinuesACallWhoseIDRepeats(t *testing.T) {
	r := fold(t,
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "call_1", Name: "f", Args: []byte(`{"a":`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "call_1", Args: []byte(`1}`)}}},
		&llm.Frame{Delta: []llm.Part{&llm.ToolUse{ID: "call_2", Name: "g", Args: []byte(`{}`)}}},
		&llm.Frame{Stop: llm.StopToolUse},
	)
	var calls []*llm.ToolUse
	for _, p := range parts(r) {
		if t, ok := p.(*llm.ToolUse); ok {
			calls = append(calls, t)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("calls = %d", len(calls))
	}
	if string(calls[0].Args) != `{"a":1}` || calls[0].Name != "f" {
		t.Fatalf("first = %+v", calls[0])
	}
	if string(calls[1].Args) != `{}` || calls[1].Name != "g" {
		t.Fatalf("second = %+v", calls[1])
	}
}

func TestFoldGathersALongRunWithoutRewritingIt(t *testing.T) {
	frames := make([]*llm.Frame, 0, 2001)
	for range 1000 {
		frames = append(frames, &llm.Frame{Delta: []llm.Part{&llm.Text{Text: "ab"}}})
	}
	for range 1000 {
		frames = append(frames, &llm.Frame{Delta: []llm.Part{&llm.Thinking{Text: "cd"}}})
	}
	frames = append(frames, &llm.Frame{Stop: llm.StopEnd})
	r := fold(t, frames...)

	if got := llm.TextOf(parts(r)); len(got) != 2000 {
		t.Fatalf("text = %d bytes, want 2000", len(got))
	}
	var thinking int
	for _, p := range parts(r) {
		if th, ok := p.(*llm.Thinking); ok {
			thinking += len(th.Text)
		}
	}
	if thinking != 2000 {
		t.Fatalf("thinking = %d bytes: every delta of a run must survive being gathered", thinking)
	}
}

func TestFoldBoundsWhatAnUpstreamCanMint(t *testing.T) {
	f := NewFolder(FoldOptions{})
	for i := range maxChoices * 3 {
		idx := i
		f.Add(&llm.Frame{Choice: &idx, Delta: []llm.Part{&llm.Text{Text: "x"}}, Stop: llm.StopEnd})
	}
	resp := f.Response(llm.StreamMeta{Provider: "stub"}, llm.Usage{})
	if len(resp.Choices) > maxChoices {
		t.Fatalf("choices = %d: the candidate index is the upstream's to choose", len(resp.Choices))
	}
}

// The ceiling is there to stop an upstream filling memory. Saying so once for
// every delta past it fills memory faster than keeping the text would have.
func TestTheBufferCeilingIsSaidOnceAndNotOncePerDelta(t *testing.T) {
	const deltas = 4000
	frames := make([]*llm.Frame, 0, deltas+1)
	for range deltas {
		frames = append(frames, &llm.Frame{Choice: idx(0), Delta: []llm.Part{&llm.Text{Text: strings.Repeat("x", 64)}}})
	}
	frames = append(frames, &llm.Frame{Choice: idx(0), Stop: llm.StopEnd})
	s := llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "m"}, frames)
	resp, err := Fold(s, FoldOptions{MaxBufferBytes: 1024})
	if err != nil {
		t.Fatal(err)
	}
	if !hasCode(resp.Diagnostics, llm.WireBufferFull) {
		t.Fatal("an accumulation cap that drops content must say so")
	}
	var said int
	for _, d := range resp.Diagnostics {
		if d.Code == llm.WireBufferFull {
			said++
		}
	}
	if said != 1 {
		t.Errorf("the ceiling was reported %d times for one candidate and %d deltas past it: "+
			"the note costs more than the text it refused", said, deltas)
	}
}

// A field whose value is null carries nothing, so leaving it behind loses
// nothing. Reporting it as a loss makes every OpenAI answer read as one: each
// choice carries a logprobs of null unless the caller asked for them.
func TestAFieldWithNothingInItIsNotALoss(t *testing.T) {
	known := map[string]bool{"index": true, "message": true, "finish_reason": true}
	for _, tc := range []struct {
		what string
		body string
		loss bool
	}{
		{"null", `{"index":0,"message":{},"finish_reason":"stop","logprobs":null}`, false},
		{"and something in it", `{"index":0,"message":{},"finish_reason":"stop","logprobs":{"content":[]}}`, true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			got := len(Unmodelled([]byte(tc.body), known, "choice", nil)) > 0
			if got != tc.loss {
				t.Errorf("reported = %v, want %v", got, tc.loss)
			}
		})
	}
}

// What a frame said is said once in the answer it folds into. The stream keeps
// its own frames' notes as well, so taking both counts each one twice.
func TestAFramesNoteIsFoldedOnce(t *testing.T) {
	note := llm.Diagnostic{Code: llm.WireFrameMalformed, Class: llm.ClassTransport, Stage: llm.StageResponse}
	s := llm.StaticStream(llm.StreamMeta{Provider: "stub", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}, Diagnostics: llm.Diagnostics{note}},
		{Stop: llm.StopEnd},
	})
	resp, err := Fold(s, FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	var said int
	for _, d := range resp.Diagnostics {
		if d.Code == llm.WireFrameMalformed {
			said++
		}
	}
	if said != 1 {
		t.Errorf("one frame's note appears %d times in the answer", said)
	}
}

// A folder used on its own still hands back what its frames said: the answer
// it builds is the whole of what it was given, whoever else kept a copy.
func TestAFolderOnItsOwnKeepsWhatItsFramesSaid(t *testing.T) {
	note := llm.Diagnostic{Code: llm.WireFrameMalformed, Class: llm.ClassTransport, Stage: llm.StageResponse}
	f := NewFolder(FoldOptions{})
	f.Add(&llm.Frame{Delta: []llm.Part{&llm.Text{Text: "hi"}}, Diagnostics: llm.Diagnostics{note}})
	f.Add(&llm.Frame{Stop: llm.StopEnd})
	resp := f.Response(llm.StreamMeta{Provider: "stub"}, llm.Usage{})
	if !hasCode(resp.Diagnostics, llm.WireFrameMalformed) {
		t.Error("a frame's note was lost by a folder used on its own")
	}
}

// What the folder found itself goes to the recorder it was handed, where the
// attempt's other notes are, or failing one onto the answer. A caller reads
// both, so a note kept in each was read twice.
func TestWhatTheFolderFindsItselfIsKeptInOnePlace(t *testing.T) {
	frames := func() *llm.Stream {
		return llm.StaticStream(llm.StreamMeta{Provider: "stub"}, []*llm.Frame{
			{Choice: idx(0), Delta: []llm.Part{&llm.Text{Text: strings.Repeat("x", 64)}}},
			{Choice: idx(0), Stop: llm.StopEnd},
		})
	}
	var noted llm.Diagnostics
	resp, err := Fold(frames(), FoldOptions{MaxBufferBytes: 8, Recorder: &noted})
	if err != nil {
		t.Fatal(err)
	}
	if n, m := len(noted.Find(llm.WireBufferFull)), len(resp.Diagnostics.Find(llm.WireBufferFull)); n+m != 1 || n != 1 {
		t.Fatalf("recorded %d times and on the answer %d: said other than once, or not where the attempt's notes are", n, m)
	}
	if resp, err = Fold(frames(), FoldOptions{MaxBufferBytes: 8}); err != nil {
		t.Fatal(err)
	}
	if n := len(resp.Diagnostics.Find(llm.WireBufferFull)); n != 1 {
		t.Fatalf("on the answer %d times with nowhere else to say it", n)
	}
}
