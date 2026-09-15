// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"encoding/json"
	"iter"
	"time"
)

type StreamMeta struct {
	ID        string
	Provider  string
	Model     string
	Requested string
	Created   time.Time
	Session   string
	Route     string
	Extra     map[string]json.RawMessage
}

type Stream struct {
	meta    StreamMeta
	next    func() (*Frame, error)
	closer  func() error
	onFrame func(*Frame)
	onEnd   []func(*Stream)

	pending   *Frame
	primed    bool
	done      bool
	closed    bool
	ended     bool
	committed bool
	content   bool
	open      map[int]bool
	stopped   map[int]bool
	tolerate  []string
	exhausted bool

	turn    Usage
	total   Usage
	summary Usage
	seen    []Usage

	err      error
	diags    Diagnostics
	attempts []Attempted
}

func NewStream(meta StreamMeta, next func() (*Frame, error), closer func() error) *Stream {
	return &Stream{meta: meta, next: next, closer: closer}
}

func (s *Stream) Meta() StreamMeta { return s.meta }

func (s *Stream) Committed() bool { return s.committed }

func (s *Stream) Prime() error {
	if s.primed || s.done {
		return s.err
	}
	s.primed = true
	if s.next == nil {
		s.done = true
		s.finish()
		return s.err
	}
	f, err := s.next()
	if err != nil {
		s.fail(err)
		return s.err
	}
	if f == nil {
		s.done, s.exhausted = true, true
		s.finish()
		return s.err
	}
	s.absorb(f)
	s.pending = f
	return nil
}

func (s *Stream) Frames() iter.Seq[*Frame] {
	return func(yield func(*Frame) bool) {
		if s.pending != nil {
			f := s.pending
			s.pending = nil
			s.committed = true
			s.observe(f)
			if !yield(f) {
				return
			}
		}
		for !s.done && s.next != nil {
			f, err := s.next()
			if err != nil {
				s.fail(err)
				return
			}
			if f == nil {
				s.done, s.exhausted = true, true
				s.finish()
				return
			}
			s.absorb(f)
			s.committed = true
			s.observe(f)
			if !yield(f) {
				return
			}
		}
	}
}

func (s *Stream) absorb(f *Frame) {
	if f == nil {
		return
	}
	if s.meta.ID == "" && f.ID != "" {
		s.meta.ID = f.ID
	}
	if f.Model != "" {
		s.meta.Model = f.Model
	}
	if f.Session != "" {
		s.meta.Session = f.Session
	}
	if len(f.Extra) > 0 && s.meta.Extra == nil {
		s.meta.Extra = f.Extra
	}
	if len(f.Delta) > 0 {
		s.content = true
		s.mark(&s.open, f.ChoiceIndex())
	}
	if f.Stop != StopNone {
		s.mark(&s.stopped, f.ChoiceIndex())
	}
	if f.Usage != nil && f.Total {
		s.summary = s.summary.Cover(*f.Usage)
	} else if f.Usage != nil {
		if s.restated(*f.Usage) {
			s.diags = append(s.diags, Diagnostic{
				Code: AcctTurnRestate, Class: ClassAccounting, Stage: StageResponse,
				Severity: SeverityInfo, Provider: s.meta.Provider,
				Detail: "a completed turn was announced again with identical counts",
			})
		} else {
			s.turn = s.turn.Merge(*f.Usage)
		}
	}
	if f.TurnEnd {
		s.bank()
	}
	if len(f.Diagnostics) > 0 {
		s.diags = append(s.diags, f.Diagnostics...)
	}
	if f.Err != nil {
		s.err = f.Err
	}
}

func (s *Stream) observe(f *Frame) {
	if s.onFrame != nil {
		s.onFrame(f)
	}
}

func (s *Stream) restated(u Usage) bool {
	if s.content || len(s.seen) == 0 || !s.turn.Empty() {
		return false
	}
	return sameColumns(s.seen[len(s.seen)-1], u)
}

func (s *Stream) bank() {
	if !s.turn.Empty() {
		s.seen = append(s.seen, s.turn)
		turn := s.turn
		if turn.Turns == 0 {
			turn.Turns = 1
		}
		s.total = s.total.Add(turn)
	}
	s.turn = Usage{}
	s.content = false
}

func (s *Stream) finish() {
	if s.ended {
		return
	}
	s.ended = true
	s.bank()
	s.truncated()
	for _, fn := range s.onEnd {
		fn(s)
	}
}

func (s *Stream) Tolerate(codes ...string) { s.tolerate = append(s.tolerate, codes...) }

func (s *Stream) mark(set *map[int]bool, choice int) {
	if *set == nil {
		*set = map[int]bool{}
	}
	(*set)[choice] = true
}

func (s *Stream) unfinished() bool {
	if len(s.stopped) == 0 {
		return true
	}
	for choice := range s.open {
		if !s.stopped[choice] {
			return true
		}
	}
	return false
}

func (s *Stream) truncated() {
	if !s.exhausted || s.err != nil || !s.unfinished() {
		return
	}
	s.diags = append(s.diags, Diagnostic{
		Code: WireNoFinish, Class: ClassTransport, Stage: StageResponse,
		Severity: SeverityLoss, Provider: s.meta.Provider, Model: s.meta.Model,
		Detail: "the stream ended without a finish reason, so the answer is not known to be complete",
	})
	for _, code := range s.tolerate {
		if code == WireNoFinish {
			return
		}
	}
	s.err = &Error{
		Kind: KindUpstream, Code: WireNoFinish, Provider: s.meta.Provider, Model: s.meta.Model,
		Message: "the stream ended without a finish reason, so the answer is not known to be complete",
		Sent:    true, Retryable: true,
	}
}

func (s *Stream) fail(err error) {
	s.done = true
	if s.err == nil {
		s.err = err
	}
	s.finish()
}

func (s *Stream) Usage() Usage {
	out := s.total
	if !s.turn.Empty() {
		out = out.Add(s.turn)
	}
	if s.summary.Empty() {
		return out
	}
	return out.Cover(s.summary)
}

func (s *Stream) Err() error { return s.err }

func (s *Stream) Diagnostics() Diagnostics { return s.diags }

func (s *Stream) Note(d Diagnostic) { s.diags = append(s.diags, d) }

func (s *Stream) Record(a Attempted) { s.attempts = append(s.attempts, a) }

func (s *Stream) Attempts() []Attempted { return s.attempts }

func (s *Stream) OnFrame(fn func(*Frame)) { s.onFrame = fn }

func (s *Stream) OnEnd(fn func(*Stream)) {
	if fn == nil {
		return
	}
	s.onEnd = append(s.onEnd, fn)
}

func (s *Stream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	s.done = true
	var err error
	if s.closer != nil {
		err = s.closer()
		if err != nil && s.err == nil {
			s.err = err
		}
	}
	s.finish()
	return err
}

func sameColumns(a, b Usage) bool {
	return a.Prompt == b.Prompt && a.Completion == b.Completion &&
		a.CacheRead == b.CacheRead && a.CacheWrite == b.CacheWrite &&
		a.CacheWrite5m == b.CacheWrite5m && a.Reasoning == b.Reasoning
}

func StaticStream(meta StreamMeta, frames []*Frame) *Stream {
	i := 0
	return NewStream(meta, func() (*Frame, error) {
		if i >= len(frames) {
			return nil, nil
		}
		f := frames[i]
		i++
		return f, nil
	}, nil)
}
