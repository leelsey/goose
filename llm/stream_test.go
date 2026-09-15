// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"errors"
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
	s := frames(&Frame{Usage: &Usage{Prompt: 1, Reported: true}})
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
