// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

func frames(fs ...*Frame) *Stream { return StaticStream(StreamMeta{}, fs) }

func drain(s *Stream) []*Frame {
	var out []*Frame
	for f := range s.Frames() {
		out = append(out, f)
	}
	return out
}

func TestStreamMergesUsageWithinTurn(t *testing.T) {
	s := frames(
		&Frame{Usage: &Usage{Prompt: 1200, CacheRead: 900, Reported: true}},
		&Frame{Delta: []Part{&Text{Text: "hi"}}},
		&Frame{Stop: StopEnd},
		&Frame{Usage: &Usage{Completion: 42, Reported: true}},
	)
	drain(s)
	u := s.Usage()
	if u.Prompt != 1200 || u.Completion != 42 || u.CacheRead != 900 {
		t.Fatalf("usage = %+v", u)
	}
	if !u.Reported {
		t.Fatal("reported lost")
	}
}

func TestStreamBanksTurnsOnBoundary(t *testing.T) {
	s := frames(
		&Frame{Usage: &Usage{Prompt: 100, Completion: 10, Reported: true}},
		&Frame{TurnEnd: true},
		&Frame{Usage: &Usage{Prompt: 150, Completion: 20, Reported: true}},
		&Frame{TurnEnd: true},
	)
	drain(s)
	u := s.Usage()
	if u.Prompt != 250 || u.Completion != 30 {
		t.Fatalf("usage = %+v, want 250/30 summed across turns", u)
	}
	if u.Turns != 2 {
		t.Fatalf("turns = %d, want 2", u.Turns)
	}
}

func TestStreamIgnoresRestatedTurn(t *testing.T) {
	same := Usage{Prompt: 100, Completion: 10, Reported: true}
	s := frames(
		&Frame{Usage: &same},
		&Frame{TurnEnd: true},
		&Frame{Usage: &same},
		&Frame{TurnEnd: true},
	)
	drain(s)
	if u := s.Usage(); u.Prompt != 100 || u.Turns != 1 {
		t.Fatalf("usage = %+v, want one turn of 100", u)
	}
}

func TestStreamKeepsUsageAfterBreak(t *testing.T) {
	s := frames(
		&Frame{Usage: &Usage{Prompt: 900, Reported: true}},
		&Frame{Delta: []Part{&Text{Text: "a"}}},
		&Frame{Delta: []Part{&Text{Text: "b"}}},
	)
	for range s.Frames() {
		break
	}
	if u := s.Usage(); u.Prompt != 900 {
		t.Fatalf("usage = %+v: a client that hangs up still spent it", u)
	}
}

func TestStreamKeepsUsageAfterError(t *testing.T) {
	boom := errors.New("connection reset")
	i := 0
	s := NewStream(StreamMeta{}, func() (*Frame, error) {
		i++
		switch i {
		case 1:
			return &Frame{Usage: &Usage{Prompt: 500, Reported: true}}, nil
		default:
			return nil, boom
		}
	}, nil)
	drain(s)
	if !errors.Is(s.Err(), boom) {
		t.Fatalf("err = %v", s.Err())
	}
	if u := s.Usage(); u.Prompt != 500 {
		t.Fatalf("usage = %+v: a failed call was still billed", u)
	}
}

func TestStreamSurfacesInBandError(t *testing.T) {
	s := frames(&Frame{Err: &Error{Kind: KindOverloaded, Message: "overloaded"}})
	drain(s)
	if s.Err() == nil {
		t.Fatal("in-band error swallowed")
	}
}

func TestPrimeDoesNotCommit(t *testing.T) {
	s := frames(&Frame{ID: "resp_1", Model: "claude-x", Session: "s-9", Delta: []Part{&Text{Text: "x"}}})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	if s.Committed() {
		t.Fatal("priming must leave the request re-routable")
	}
	if m := s.Meta(); m.ID != "resp_1" || m.Model != "claude-x" || m.Session != "s-9" {
		t.Fatalf("meta = %+v: identity must be readable before the first frame is written", m)
	}
	got := drain(s)
	if len(got) != 1 {
		t.Fatalf("frames = %d, want the primed frame delivered exactly once", len(got))
	}
	if !s.Committed() {
		t.Fatal("delivering a frame commits the request")
	}
}

func TestPrimeReportsFailureBeforeCommit(t *testing.T) {
	boom := errors.New("503")
	s := NewStream(StreamMeta{}, func() (*Frame, error) { return nil, boom }, nil)
	if err := s.Prime(); !errors.Is(err, boom) {
		t.Fatalf("prime err = %v", err)
	}
	if s.Committed() {
		t.Fatal("a failure before the first frame is still a routing decision")
	}
}

// A backend that answers and then fails on its first event has delivered
// nothing, so that is a failure before the first frame too. What the event
// counted is still the stream's to report.
func TestPrimeReportsAFirstFrameThatIsOnlyAFailure(t *testing.T) {
	boom := &Error{Kind: KindOverloaded, Message: "overloaded", Sent: true}
	s := frames(&Frame{Err: boom, Usage: &Usage{Prompt: 9, Reported: true}})
	if err := s.Prime(); !errors.Is(err, boom) {
		t.Fatalf("prime err = %v: nothing was delivered, so the router can still go elsewhere", err)
	}
	if s.Committed() {
		t.Fatal("a failure before any content is still a routing decision")
	}
	if u := s.Usage(); u.Prompt != 9 {
		t.Fatalf("usage = %+v: the failed event was still counted", u)
	}

	s = frames(&Frame{Delta: []Part{&Text{Text: "half"}}, Err: boom})
	if err := s.Prime(); err != nil {
		t.Fatalf("prime err = %v: a frame with content in it is an answer that began", err)
	}
	if got := drain(s); len(got) != 1 || s.Err() == nil {
		t.Fatalf("frames = %d, err = %v", len(got), s.Err())
	}
}

// A backend opens with frames that name and count the answer before any of it
// arrives, and can fail after them. Nothing has been delivered until content
// has, so that failure is still one the router can walk past.
func TestPrimeHoldsWhatArrivesBeforeContent(t *testing.T) {
	boom := &Error{Kind: KindOverloaded, Message: "overloaded", Sent: true}
	opening := func() *Frame {
		return &Frame{ID: "msg_1", Model: "claude-x", Role: RoleAssistant, Usage: &Usage{Prompt: 12, Reported: true}}
	}
	for name, fs := range map[string][]*Frame{
		"after the opening frame": {opening(), {Err: boom}},
		"after the closing count": {
			{Usage: &Usage{Prompt: 12, Reported: true}, Total: true, TurnEnd: true, Stop: StopEnd},
			{Err: boom},
		},
	} {
		s := frames(fs...)
		if err := s.Prime(); !errors.Is(err, boom) {
			t.Fatalf("%s: prime err = %v: nothing had been delivered, so the router can still go elsewhere", name, err)
		}
		if s.Committed() {
			t.Fatalf("%s: a failure before any content is still a routing decision", name)
		}
		if u := s.Usage(); u.Prompt != 12 {
			t.Fatalf("%s: usage = %+v: what the opening counted is still reported", name, u)
		}
	}

	s := frames(opening(), &Frame{Delta: []Part{&Text{Text: "hi"}}}, &Frame{Stop: StopEnd})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	if s.Committed() || s.Meta().ID != "msg_1" {
		t.Fatalf("committed = %v, meta = %+v", s.Committed(), s.Meta())
	}
	if got := drain(s); len(got) != 3 || got[0].ID != "msg_1" || TextOf(got[1].Delta) != "hi" || s.Err() != nil {
		t.Fatalf("frames = %+v, err = %v: what was held is delivered once, in order", got, s.Err())
	}

	s = frames(opening(), &Frame{Stop: StopEnd, TurnEnd: true})
	if err := s.Prime(); err != nil {
		t.Fatalf("prime err = %v: an answer with nothing in it is still an answer", err)
	}
	if got := drain(s); len(got) != 2 || s.Err() != nil {
		t.Fatalf("frames = %d, err = %v", len(got), s.Err())
	}

	s = frames(opening())
	s.Tolerate(WireNoFinish)
	if err := s.Prime(); err != nil {
		t.Fatalf("prime err = %v: the caller chose to live with an unfinished answer", err)
	}

	var many []*Frame
	for range maxHeld + 4 {
		many = append(many, opening())
	}
	s = frames(append(many, &Frame{Err: boom})...)
	if err := s.Prime(); err != nil {
		t.Fatalf("prime err = %v: past the bound, the stream is handed over as it stands", err)
	}
	if got := drain(s); len(got) != len(many)+1 || !errors.Is(s.Err(), boom) {
		t.Fatalf("frames = %d, err = %v", len(got), s.Err())
	}
}

// A stream held to a policy stops at the first frame carrying a loss the policy
// refuses, without delivering it, and says why. What it delivered before that
// stays delivered, and a stream refused at its first frame has delivered none.
func TestAnEnforcedStreamStopsAtALossItRefuses(t *testing.T) {
	lost := Diagnostic{Code: DropRefusal, Class: ClassContent, Severity: SeverityLoss}
	s := frames(
		&Frame{Delta: []Part{&Text{Text: "a"}}},
		&Frame{Delta: []Part{&Text{Text: "b"}}, Diagnostics: Diagnostics{lost}},
		&Frame{Delta: []Part{&Text{Text: "c"}}},
	)
	ended := 0
	s.OnEnd(func(*Stream) { ended++ })
	s.Enforce(LossPolicy{Strict: true}, Renders{})
	if got := drain(s); len(got) != 1 || TextOf(got[0].Delta) != "a" {
		t.Fatalf("frames = %+v: delivery went on past the loss", got)
	}
	if e := AsError(s.Err()); e == nil || e.Kind != KindLoss {
		t.Fatalf("err = %v: a refused stream that ends cleanly reads as a whole answer", s.Err())
	}
	if ended != 1 {
		t.Fatalf("end hook fired %d times", ended)
	}

	s = frames(&Frame{Delta: []Part{&Text{Text: "b"}}, Diagnostics: Diagnostics{lost}})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	s.Enforce(LossPolicy{Deny: []string{DropRefusal}}, Renders{})
	if got := drain(s); len(got) != 0 || s.Committed() {
		t.Fatalf("frames = %d, committed = %v: the lossy first frame was delivered", len(got), s.Committed())
	}
	if e := AsError(s.Err()); e == nil || e.Kind != KindLoss {
		t.Fatalf("err = %v", s.Err())
	}

	s = frames(&Frame{Delta: []Part{&Text{Text: "b"}}, Diagnostics: Diagnostics{lost}}, &Frame{Stop: StopEnd})
	s.Enforce(LossPolicy{}, Renders{})
	if got := drain(s); len(got) != 2 || s.Err() != nil {
		t.Fatalf("frames = %d, err = %v: a policy that accepts the loss delivers it", len(got), s.Err())
	}

	s = frames(&Frame{Delta: []Part{&Text{Text: "b"}}}, &Frame{Stop: StopEnd})
	s.Enforce(LossPolicy{Strict: true}, Renders{})
	s.Note(lost)
	if got := drain(s); len(got) != 2 || s.Err() != nil {
		t.Fatalf("frames = %d, err = %v: a note added afterwards is the caller's own", len(got), s.Err())
	}

	// An answer with no content in it ends while it is primed, and is held to
	// the policy all the same.
	s = frames(&Frame{Stop: StopEnd, TurnEnd: true})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	s.Note(lost)
	s.Enforce(LossPolicy{Strict: true}, Renders{})
	if got := drain(s); len(got) != 0 {
		t.Fatalf("frames = %d: a refused answer was delivered because it had already ended", len(got))
	}
	if e := AsError(s.Err()); e == nil || e.Kind != KindLoss {
		t.Fatalf("err = %v", s.Err())
	}
}

// A stream held to a policy for a destination weighs what it hands over
// against what that destination can write out: a block carried from a wire it
// does not speak is refused before it is delivered, held frames included, and
// the refusal is of something that was sent.
func TestAnEnforcedStreamRefusesABlockItsDestinationCannotWriteOut(t *testing.T) {
	carried := func() *Stream {
		return frames(
			&Frame{Role: RoleAssistant},
			&Frame{Delta: []Part{&Raw{Vendor: "anthropic", Type: "web_search_tool_result", JSON: json.RawMessage(`{}`)}},
				Diagnostics: Diagnostics{{Code: DropUnknownBlock, Class: ClassContent, Severity: SeverityInfo}}},
			&Frame{Delta: []Part{&Text{Text: "so"}}},
			&Frame{Stop: StopEnd},
		)
	}
	s := carried()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	s.Enforce(LossPolicy{Strict: true}, Renders{ToolCalls: true, Candidates: 128})
	if got := drain(s); len(got) != 0 || s.Committed() {
		t.Fatalf("frames = %d: the stream was handed over with a block its destination drops", len(got))
	}
	if e := AsError(s.Err()); e == nil || e.Kind != KindLoss || e.Code != DropUnknownBlock || !e.Sent {
		t.Fatalf("err = %+v", e)
	}

	s = carried()
	s.Enforce(LossPolicy{Strict: true}, Renders{ToolCalls: true, Candidates: 1, Raw: []string{"anthropic"}})
	if got := drain(s); len(got) != 4 || s.Err() != nil {
		t.Fatalf("frames = %d, err = %v: the block's own wire writes it out whole", len(got), s.Err())
	}

	s = carried()
	s.Enforce(LossPolicy{Strict: true}, Renders{})
	if got := drain(s); len(got) != 4 || s.Err() != nil {
		t.Fatalf("frames = %d, err = %v: a library caller takes the block as it is", len(got), s.Err())
	}
}

// What an encoder loses in writing a stream out is held to the policy the
// stream was enforced with. A loss it will not take ends the stream there, as a
// refusal of something sent, and nothing after it is delivered. A stream never
// enforced, or a loss the policy takes, goes on.
func TestAStreamRefusesWhatItsEncoderLosesWhereThePolicyWillNot(t *testing.T) {
	lost := Diagnostic{Code: DropImage, Class: ClassContent, Stage: StageEncode, Severity: SeverityLoss}
	two := func() *Stream {
		return frames(&Frame{Delta: []Part{&Text{Text: "a"}}}, &Frame{Delta: []Part{&Text{Text: "b"}}}, &Frame{Stop: StopEnd})
	}

	s := two()
	ended := 0
	s.OnEnd(func(*Stream) { ended++ })
	s.Enforce(LossPolicy{Strict: true}, Renders{})
	var got []*Frame
	for f := range s.Frames() {
		got = append(got, f)
		if err := s.Check(Diagnostics{lost}); err != nil {
			if e := AsError(err); e.Kind != KindLoss || e.Code != DropImage || !e.Sent {
				t.Fatalf("err = %+v", e)
			}
			break
		}
	}
	if len(got) != 1 || ended != 1 {
		t.Fatalf("frames = %d, ended = %d: delivery went on past what the encoder lost", len(got), ended)
	}
	if e := AsError(s.Err()); e == nil || e.Code != DropImage {
		t.Fatalf("err = %v: the stream does not say why it ended", s.Err())
	}

	for _, s := range []*Stream{two(), func() *Stream {
		s := two()
		s.Enforce(LossPolicy{Strict: true, Allow: []string{DropImage}}, Renders{})
		return s
	}()} {
		n := 0
		for range s.Frames() {
			n++
			if err := s.Check(Diagnostics{lost}); err != nil {
				t.Fatalf("err = %v: nothing here refuses the loss", err)
			}
		}
		if n != 3 || s.Err() != nil {
			t.Fatalf("frames = %d, err = %v", n, s.Err())
		}
	}
}

func TestCloseBanksAndIsIdempotent(t *testing.T) {
	closed := 0
	s := NewStream(StreamMeta{}, func() (*Frame, error) {
		return &Frame{Usage: &Usage{Prompt: 7, Reported: true}}, nil
	}, func() error { closed++; return nil })
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if closed != 1 {
		t.Fatalf("closer ran %d times", closed)
	}
	if u := s.Usage(); u.Prompt != 7 || u.Turns != 1 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestStreamBanksTwoGenuineTurnsWithEqualCounts(t *testing.T) {
	u := Usage{Prompt: 100, Completion: 10, Reported: true}
	s := frames(
		&Frame{Usage: &u},
		&Frame{TurnEnd: true},
		&Frame{Delta: []Part{&Text{Text: "second turn output"}}},
		&Frame{Usage: &u},
		&Frame{TurnEnd: true},
	)
	drain(s)
	if got := s.Usage(); got.Prompt != 200 || got.Turns != 2 {
		t.Fatalf("usage = %+v, want two banked turns", got)
	}
}

func TestStreamRestatementIsRecorded(t *testing.T) {
	u := Usage{Prompt: 100, Completion: 10, Reported: true}
	s := frames(&Frame{Usage: &u}, &Frame{TurnEnd: true}, &Frame{Usage: &u}, &Frame{TurnEnd: true})
	drain(s)
	if len(s.Diagnostics()) == 0 || s.Diagnostics()[0].Code != AcctTurnRestate {
		t.Fatalf("diagnostics = %+v: a suppressed turn must leave a trace", s.Diagnostics())
	}
}

func TestOnEndFiresOnceHoweverTheStreamEnds(t *testing.T) {
	cases := map[string]func() *Stream{
		"drained": func() *Stream {
			return frames(&Frame{Usage: &Usage{Prompt: 3, Reported: true}})
		},
		"failed": func() *Stream {
			i := 0
			return NewStream(StreamMeta{}, func() (*Frame, error) {
				i++
				if i == 1 {
					return &Frame{Usage: &Usage{Prompt: 3, Reported: true}}, nil
				}
				return nil, errors.New("reset")
			}, nil)
		},
	}
	for name, build := range cases {
		s := build()
		ended := 0
		var final Usage
		s.OnEnd(func(x *Stream) { ended++; final = x.Usage() })
		drain(s)
		_ = s.Close()
		if ended != 1 {
			t.Fatalf("%s: end hook fired %d times", name, ended)
		}
		if final.Prompt != 3 || final.Turns != 1 {
			t.Fatalf("%s: final usage = %+v", name, final)
		}
	}
}

func TestOnEndFiresOnCloseWithoutIteration(t *testing.T) {
	s := frames(&Frame{Usage: &Usage{Prompt: 1, Reported: true}}, &Frame{Delta: []Part{&Text{Text: "hi"}}})
	ended := 0
	s.OnEnd(func(*Stream) { ended++ })
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	_ = s.Close()
	if ended != 1 {
		t.Fatalf("end hook fired %d times: an abandoned stream is still booked", ended)
	}
}

func TestZeroStreamIsInert(t *testing.T) {
	var s Stream
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	if got := drain(&s); got != nil {
		t.Fatalf("frames = %v", got)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestStreamCarriesFailedAttempts(t *testing.T) {
	s := frames(&Frame{Delta: []Part{&Text{Text: "ok"}}})
	s.Record(Attempted{Provider: "anthropic", Sent: true, Outcome: OutcomeError, Usage: Usage{Prompt: 12, Reported: true}})
	s.Record(Attempted{Provider: "bedrock", Sent: true, Outcome: OutcomeOK})
	drain(s)
	if len(s.Attempts()) != 2 || s.Attempts()[0].Provider != "anthropic" {
		t.Fatalf("attempts = %+v", s.Attempts())
	}
}

func TestTruncatedStreamIsReported(t *testing.T) {
	s := frames(&Frame{Delta: []Part{&Text{Text: "half an ans"}}})
	drain(s)
	e := AsError(s.Err())
	if e == nil || e.Code != WireNoFinish {
		t.Fatalf("err = %v: a stream that never said why it ended is not a whole answer", s.Err())
	}
	if !e.Retryable {
		t.Fatal("a truncated stream is worth another link of the chain")
	}
	if !hasDiagnostic(s.Diagnostics(), WireNoFinish) {
		t.Fatalf("diagnostics = %+v", s.Diagnostics())
	}
}

func TestFinishedStreamIsNotCalledTruncated(t *testing.T) {
	s := frames(&Frame{Delta: []Part{&Text{Text: "whole"}}}, &Frame{Stop: StopEnd})
	drain(s)
	if s.Err() != nil {
		t.Fatalf("err = %v", s.Err())
	}
}

func TestAbandonedStreamIsNotCalledTruncated(t *testing.T) {
	s := frames(
		&Frame{Delta: []Part{&Text{Text: "one"}}},
		&Frame{Delta: []Part{&Text{Text: "two"}}},
		&Frame{Stop: StopEnd},
	)
	for range s.Frames() {
		break
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close = %v", err)
	}
	if s.Err() != nil {
		t.Fatalf("err = %v: an abandoned stream is not a truncated one", s.Err())
	}
}

func TestEmptyStreamFailsAtPrime(t *testing.T) {
	s := frames()
	if err := s.Prime(); AsError(err) == nil || AsError(err).Code != WireNoFinish {
		t.Fatalf("prime = %v", err)
	}
}

func hasDiagnostic(ds []Diagnostic, code string) bool {
	for _, d := range ds {
		if d.Code == code {
			return true
		}
	}
	return false
}

func TestASummaryIsReconciledAgainstTheTurnsItSummarises(t *testing.T) {
	s := frames(
		&Frame{Usage: &Usage{Prompt: 100, Completion: 10, Reported: true}, TurnEnd: true},
		&Frame{Usage: &Usage{Prompt: 100, Completion: 10, Reported: true}, TurnEnd: true},
		&Frame{Usage: &Usage{Prompt: 200, Completion: 20, Reported: true}, Total: true, TurnEnd: true},
		&Frame{Stop: StopEnd},
	)
	drain(s)
	u := s.Usage()
	if u.Prompt != 200 || u.Completion != 20 {
		t.Fatalf("usage = %d/%d, want 200/20", u.Prompt, u.Completion)
	}
}

func TestASummaryThatKnowsMoreThanTheTurnsWins(t *testing.T) {
	s := frames(
		&Frame{Usage: &Usage{Prompt: 100, Completion: 10, Reported: true}, TurnEnd: true},
		&Frame{Usage: &Usage{Prompt: 140, Completion: 10, CacheRead: 40, Reported: true}, Total: true},
		&Frame{Stop: StopEnd},
	)
	drain(s)
	u := s.Usage()
	if u.Prompt != 140 || u.CacheRead != 40 {
		t.Fatalf("usage = %+v: the larger of each column is the safe reading", u)
	}
}

func TestASummaryAloneIsTheFigure(t *testing.T) {
	s := frames(
		&Frame{Delta: []Part{&Text{Text: "hi"}}},
		&Frame{Usage: &Usage{Prompt: 7, Completion: 3, Reported: true}, Total: true},
		&Frame{Stop: StopEnd},
	)
	drain(s)
	if u := s.Usage(); u.Prompt != 7 || u.Completion != 3 {
		t.Fatalf("usage = %+v", u)
	}
}

func TestOneCandidateFinishingIsNotAllOfThem(t *testing.T) {
	first, second := 0, 1
	s := frames(
		&Frame{Choice: &first, Delta: []Part{&Text{Text: "one"}}},
		&Frame{Choice: &second, Delta: []Part{&Text{Text: "two"}}},
		&Frame{Choice: &first, Stop: StopEnd},
	)
	drain(s)
	e := AsError(s.Err())
	if e == nil || e.Code != WireNoFinish {
		t.Fatalf("err = %v: the second candidate never said why it ended", s.Err())
	}
}

func TestEveryCandidateFinishingIsWhole(t *testing.T) {
	first, second := 0, 1
	s := frames(
		&Frame{Choice: &first, Delta: []Part{&Text{Text: "one"}}},
		&Frame{Choice: &second, Delta: []Part{&Text{Text: "two"}}},
		&Frame{Choice: &first, Stop: StopEnd},
		&Frame{Choice: &second, Stop: StopEnd},
	)
	drain(s)
	if s.Err() != nil {
		t.Fatalf("err = %v", s.Err())
	}
}

func TestATolerantCallerIsToldAndNotFailed(t *testing.T) {
	s := frames(&Frame{Delta: []Part{&Text{Text: "half an ans"}}})
	s.Tolerate(WireNoFinish)
	drain(s)
	if s.Err() != nil {
		t.Fatalf("err = %v: the caller chose to live with it", s.Err())
	}
	if !hasDiagnostic(s.Diagnostics(), WireNoFinish) {
		t.Fatal("living with it is not the same as not being told")
	}
}

func TestTheWatcherSeesEveryFrameInOrder(t *testing.T) {
	s := frames(
		&Frame{Delta: []Part{&Text{Text: "one"}}},
		&Frame{Delta: []Part{&Text{Text: "two"}}},
		&Frame{Stop: StopEnd},
	)
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}

	var watched, yielded []string
	s.OnFrame(func(f *Frame) { watched = append(watched, TextOf(f.Delta)) })
	for f := range s.Frames() {
		yielded = append(yielded, TextOf(f.Delta))
		if len(watched) != len(yielded) {
			t.Fatalf("watched %v, yielded %v: the watcher runs before the frame goes out", watched, yielded)
		}
	}
	if len(watched) != 3 || watched[0] != "one" || watched[1] != "two" {
		t.Fatalf("watched = %v", watched)
	}
}

func TestEndHooksAddRatherThanReplace(t *testing.T) {
	s := frames(&Frame{Delta: []Part{&Text{Text: "hi"}}}, &Frame{Stop: StopEnd})
	var order []string
	s.OnEnd(func(*Stream) { order = append(order, "router") })
	s.OnEnd(func(*Stream) { order = append(order, "embedder") })
	s.OnEnd(nil)
	drain(s)
	if len(order) != 2 || order[0] != "router" || order[1] != "embedder" {
		t.Fatalf("order = %v", order)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if len(order) != 2 {
		t.Fatalf("order = %v: the hooks run once however the stream ends", order)
	}
}

func TestASecondTurnCutOffIsStillReported(t *testing.T) {
	s := frames(
		&Frame{Delta: []Part{&Text{Text: "calling a tool"}}},
		&Frame{Stop: StopToolUse},
		&Frame{TurnEnd: true},
		&Frame{Delta: []Part{&Text{Text: "half an ans"}}},
	)
	drain(s)
	e := AsError(s.Err())
	if e == nil || e.Code != WireNoFinish {
		t.Fatalf("err = %v: a turn that finished cannot vouch for the one cut off after it", s.Err())
	}
}

func TestAWholeMultiTurnStreamIsNotCalledTruncated(t *testing.T) {
	s := frames(
		&Frame{Delta: []Part{&Text{Text: "calling a tool"}}},
		&Frame{Stop: StopToolUse},
		&Frame{TurnEnd: true},
		&Frame{Delta: []Part{&Text{Text: "the answer"}}},
		&Frame{Stop: StopEnd},
		&Frame{TurnEnd: true},
	)
	drain(s)
	if s.Err() != nil {
		t.Fatalf("err = %v: every turn said why it ended", s.Err())
	}
}

func TestAnAbandonedStreamIsClosedAndBooked(t *testing.T) {
	closed, ended := false, false
	queue := []*Frame{
		{Delta: []Part{&Text{Text: "one"}}},
		{Delta: []Part{&Text{Text: "two"}}},
		{Stop: StopEnd},
	}
	next := func() (*Frame, error) {
		if len(queue) == 0 {
			return nil, nil
		}
		f := queue[0]
		queue = queue[1:]
		return f, nil
	}
	s := NewStream(StreamMeta{Provider: "stub"}, next, func() error { closed = true; return nil })
	s.OnEnd(func(*Stream) { ended = true })

	for range s.Frames() {
		break
	}
	if !closed {
		t.Fatal("a consumer that stops reading leaves the body open unless the stream closes itself")
	}
	if !ended {
		t.Fatal("nothing else will book what the call cost")
	}
}

func TestClosingWhileReadingIsSafe(t *testing.T) {
	for range 50 {
		gate := make(chan struct{})
		n := 0
		next := func() (*Frame, error) {
			n++
			if n > 200 {
				return nil, nil
			}
			i := n % 4
			return &Frame{Choice: &i, Delta: []Part{&Text{Text: "x"}}}, nil
		}
		s := NewStream(StreamMeta{Provider: "stub"}, next, func() error { return nil })
		s.OnEnd(func(done *Stream) {
			_ = done.Usage()
			_ = done.Diagnostics()
			done.Record(Attempted{Provider: "stub"})
		})

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			<-gate
			for range s.Frames() {
			}
		}()
		go func() {
			defer wg.Done()
			<-gate
			_ = s.Close()
		}()
		close(gate)
		wg.Wait()
	}
}

func TestACandidateLeftOpenIsNotForgivenByALaterTurn(t *testing.T) {
	one, two := 0, 1
	s := frames(
		&Frame{Choice: &one, Delta: []Part{&Text{Text: "first"}}},
		&Frame{Choice: &two, Delta: []Part{&Text{Text: "second"}}},
		&Frame{Choice: &one, Stop: StopToolUse},
		// Candidate two never said why it stopped.
		&Frame{TurnEnd: true},
		&Frame{Choice: &one, Delta: []Part{&Text{Text: "the answer"}}},
		&Frame{Choice: &one, Stop: StopEnd},
		&Frame{TurnEnd: true},
	)
	drain(s)
	e := AsError(s.Err())
	if e == nil || e.Code != WireNoFinish {
		t.Fatalf("err = %v: one candidate was left open, and a later whole turn does not answer for it", s.Err())
	}
}

// The candidate index is the upstream's to choose, so it must not be a key
// goose will hold any number of — and a turn it could not track is not one it
// may call complete.
func TestCandidateIndicesAreBounded(t *testing.T) {
	var frames []*Frame
	for i := 0; i < maxChoices+50; i++ {
		frames = append(frames, &Frame{Choice: &[]int{i}[0], Delta: []Part{&Text{Text: "x"}}})
	}
	frames = append(frames, &Frame{Choice: &[]int{0}[0], Stop: StopEnd})
	s := StaticStream(StreamMeta{Model: "m"}, frames)
	for range s.Frames() {
	}
	if len(s.open) > maxChoices {
		t.Fatalf("open holds %d candidates: the upstream names them, so it must not choose the cost", len(s.open))
	}
	if s.Err() == nil {
		t.Fatal("candidates goose never tracked cannot be vouched for as complete")
	}
}

// A frame carrying an id, a model or a session says something, and a provider
// that drops "empty" frames must not drop those.
func TestAFrameThatNamesSomethingIsNotEmpty(t *testing.T) {
	for _, f := range []*Frame{
		{ID: "resp_1"},
		{Model: "gpt-4o"},
		{Session: "sess_1"},
		{Extra: map[string]json.RawMessage{"x": json.RawMessage(`1`)}},
		{StopSequence: "STOP"},
	} {
		if f.Empty() {
			t.Errorf("%+v reads as empty, and a provider filtering on that would drop it", f)
		}
	}
	if !(&Frame{}).Empty() {
		t.Error("a frame carrying nothing is empty")
	}
}

// Redaction is for what a backend said. goose's own account of a stream that
// stopped short is goose's, and a client needs it to tell a cut-off answer
// from a backend that failed.
func TestGoosesOwnAccountIsNotRedacted(t *testing.T) {
	s := StaticStream(StreamMeta{Provider: "up", Model: "m"}, []*Frame{
		{Delta: []Part{&Text{Text: "half"}}},
	})
	s.Redact(func(*Error) *Error {
		return &Error{Kind: KindUpstream, Message: "the backend could not be reached"}
	})
	for range s.Frames() {
	}
	e := AsError(s.Err())
	if e == nil {
		t.Fatal("a stream that never said why it stopped is not a complete one")
	}
	if e.Code != WireNoFinish {
		t.Fatalf("code = %q, want %q: the client cannot tell a cut-off answer from an outage", e.Code, WireNoFinish)
	}
}

// A frame belongs to whoever made it. Rewriting one on the way through would
// change what a provider reusing its own frames sends the next reader.
func TestRedactionDoesNotRewriteTheProvidersFrame(t *testing.T) {
	original := &Error{Kind: KindRateLimit, Provider: "up", Message: "the backend's own words"}
	frames := []*Frame{{Err: original}}
	s := StaticStream(StreamMeta{Provider: "up"}, frames)
	s.Redact(func(*Error) *Error { return &Error{Kind: KindRateLimit, Message: "generic"} })
	for f := range s.Frames() {
		if f.Err != nil && f.Err.Message != "generic" {
			t.Fatalf("the consumer saw %q", f.Err.Message)
		}
	}
	if original.Message != "the backend's own words" {
		t.Fatalf("the provider's own frame was rewritten: %q", original.Message)
	}
	if frames[0].Err != original {
		t.Fatal("the provider's slice now holds a different error")
	}
}

// Priming a stream twice hands back the same failure the first call did, and a
// proxy that must not repeat a backend's words gets none of them either time.
func TestPrimingTwiceTellsTheSameStory(t *testing.T) {
	s := NewStream(StreamMeta{Provider: "up"}, func() (*Frame, error) {
		return nil, &Error{Kind: KindUpstream, Provider: "up", Message: "SECRET-UPSTREAM-TEXT"}
	}, nil)
	s.Redact(func(*Error) *Error {
		return &Error{Kind: KindUpstream, Message: "the upstream failed to serve this request"}
	})
	first, second := s.Prime(), s.Prime()
	if first == nil {
		t.Fatal("a stream whose only frame is a failure did not fail")
	}
	if first.Error() != second.Error() {
		t.Fatalf("first = %q, second = %q", first, second)
	}
	for _, e := range []error{first, second} {
		if strings.Contains(e.Error(), "SECRET-UPSTREAM-TEXT") {
			t.Fatalf("the backend's own words reached the caller: %v", e)
		}
	}
}
