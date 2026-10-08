// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"encoding/json"
	"iter"
	"sync"
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

// A Stream is read by one consumer and closed by whoever is supervising it,
// which are not always the same goroutine, so the accounting behind it is
// guarded. Reading it from two goroutines at once is not supported and never
// was: what the lock makes safe is closing one that is being read. The lock covers the bookkeeping only: next, the consumer's own loop
// body, and the end callbacks all run outside it, because each of them can take
// as long as the network does and the callbacks read the Stream back.
type Stream struct {
	mu sync.Mutex

	meta    StreamMeta
	next    func() (*Frame, error)
	closer  func() error
	onFrame func(*Frame)
	hide    func(*Error) *Error
	onEnd   []func(*Stream)

	pending   []*Frame
	primed    bool
	done      bool
	closed    bool
	ended     bool
	committed bool
	content   bool
	open      map[int]bool
	stopped   map[int]bool
	unstopped map[int]bool
	unheld    bool
	// ourErr marks a failure goose wrote itself, which a redaction meant for a
	// backend's words has nothing to do with.
	ourErr    bool
	turnStop  bool
	tolerate  []string
	policy    *LossPolicy
	to        Renders
	exhausted bool

	turn    Usage
	total   Usage
	summary Usage
	settled *Usage
	seen    []Usage

	err      error
	diags    Diagnostics
	attempts []Attempted
}

func NewStream(meta StreamMeta, next func() (*Frame, error), closer func() error) *Stream {
	return &Stream{meta: meta, next: next, closer: closer}
}

func (s *Stream) Meta() StreamMeta {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.meta
}

func (s *Stream) Committed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.committed
}

func (s *Stream) Prime() error {
	s.mu.Lock()
	if s.primed || s.done {
		s.mu.Unlock()
		return s.Err()
	}
	s.primed = true
	if s.next == nil {
		s.done = true
		s.mu.Unlock()
		s.finish()
		return s.Err()
	}
	next := s.next
	s.mu.Unlock()

	// A backend opens with frames that name the answer and count it before any
	// of it arrives, and can still fail after them. Those are held with the
	// first frame of content, so a failure before it is one before anything was
	// delivered, and the router can still try elsewhere. What the held frames
	// counted is absorbed, and reported whichever way it goes.
	var held []*Frame
	for {
		f, err := next()
		if err != nil {
			s.fail(err)
			return s.Err()
		}
		s.mu.Lock()
		if f == nil {
			s.done, s.exhausted = true, true
			s.pending = held
			s.mu.Unlock()
			s.finish()
			return s.Err()
		}
		s.absorb(f)
		if f.Err != nil && len(f.Delta) == 0 {
			s.done = true
			s.mu.Unlock()
			s.finish()
			return s.Err()
		}
		held = append(held, f)
		if len(f.Delta) > 0 || len(held) == maxHeld {
			s.pending = held
			s.mu.Unlock()
			return nil
		}
		s.mu.Unlock()
	}
}

// maxHeld bounds what Prime holds before content: a backend opens with a frame
// or two, and one that never gets to the answer is not held without end.
const maxHeld = 16

func (s *Stream) Frames() iter.Seq[*Frame] {
	return func(yield func(*Frame) bool) {
		for {
			s.mu.Lock()
			next, onFrame, hide := s.next, s.onFrame, s.hide
			var f *Frame
			held := len(s.pending) > 0
			if held {
				f, s.pending = s.pending[0], s.pending[1:]
			}
			stop := !held && (s.done || next == nil)
			s.mu.Unlock()
			if stop {
				return
			}

			if !held {
				var err error
				f, err = next()
				if err != nil {
					s.fail(err)
					return
				}
				if f == nil {
					s.mu.Lock()
					s.done, s.exhausted = true, true
					s.mu.Unlock()
					s.finish()
					return
				}
			}
			s.mu.Lock()
			if !held {
				s.absorb(f)
			}
			refused := s.refuse(f.Diagnostics, f.Delta)
			if !refused {
				s.committed = true
			}
			s.mu.Unlock()
			if refused {
				s.finish()
				return
			}

			f = s.shown(hide, f)
			if onFrame != nil {
				onFrame(f)
			}
			// A consumer that stops reading has ended the stream: nothing else
			// will close the body or book what the call cost.
			if !yield(f) {
				_ = s.Close()
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
		// A candidate can say why it stopped after the turn carrying its
		// content has already been banked, which is the shape a whole-turn
		// binary emits, so a stop clears the mark rather than only adding one.
		s.mark(&s.stopped, f.ChoiceIndex())
		delete(s.unstopped, f.ChoiceIndex())
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
		// open and stopped describe the turn that is ending. Carrying them into
		// the next one would let a turn that finished vouch for a later one that
		// was cut off, so what is kept is the verdict rather than the marks.
		// A turn that carried nothing is bookkeeping, not a turn, and says
		// nothing about the one before it.
		if len(s.open) > 0 || len(s.stopped) > 0 {
			// A candidate that never said why it stopped is remembered by name:
			// a later turn that finished cleanly speaks for itself, not for the
			// one this candidate left open.
			for choice := range s.open {
				if !s.stopped[choice] {
					s.mark(&s.unstopped, choice)
				}
			}
			s.turnStop = s.turnComplete()
			s.open, s.stopped = nil, nil
		}
		s.bank()
	}
	if len(f.Diagnostics) > 0 {
		s.diags = append(s.diags, f.Diagnostics...)
	}
	if f.Err != nil {
		s.err = f.Err
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
	s.mu.Lock()
	if s.ended {
		s.mu.Unlock()
		return
	}
	s.ended = true
	s.bank()
	s.truncated()
	onEnd := make([]func(*Stream), len(s.onEnd))
	copy(onEnd, s.onEnd)
	s.mu.Unlock()

	// Outside the lock: these read the stream back for what it cost.
	for _, fn := range onEnd {
		fn(s)
	}
}

func (s *Stream) Tolerate(codes ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tolerate = append(s.tolerate, codes...)
}

// Enforce holds a stream to a loss policy on its way to a destination that
// renders what to declares: what it has recorded by then, and every frame still
// to be delivered, including any part the destination has no field for. A loss
// the policy makes fatal ends the stream with a KindLoss failure from Err,
// which is the refusal an unstreamed answer gets, and the frame carrying it is
// not delivered. Frames already delivered cannot be recalled, so a stream
// refused part way has handed over everything before the loss. Notes added
// after it are not held to it. A zero Renders takes every part as it is.
func (s *Stream) Enforce(p LossPolicy, to Renders) {
	s.mu.Lock()
	s.policy, s.to = &p, to
	var held []Part
	for _, f := range s.pending {
		held = append(held, f.Delta...)
	}
	refused := s.refuse(s.diags, held)
	s.mu.Unlock()
	if refused {
		s.finish()
	}
}

// Check holds what an encoder lost in writing the stream out to the policy the
// stream was enforced with: a loss the destination's Renders could not name
// beforehand, met only in the writing. A loss the policy will not take ends the
// stream as Enforce would and is returned, for the encoder to write as its
// failure in place of what it lost and of everything after it. A stream that
// was never enforced refuses nothing.
func (s *Stream) Check(ds Diagnostics) error {
	if len(ds) == 0 {
		return nil
	}
	s.mu.Lock()
	refused := s.refuse(ds, nil)
	s.mu.Unlock()
	if !refused {
		return nil
	}
	s.finish()
	return s.Err()
}

// refuse ends the stream on a loss the policy will not take, in what was noted
// or in parts the destination cannot write out, and runs under the mutex.
func (s *Stream) refuse(ds Diagnostics, parts []Part) bool {
	if s.policy == nil {
		return false
	}
	lost := s.to.unrendered(parts)
	if len(lost) > 0 {
		ds = append(append(Diagnostics{}, ds...), lost...)
	}
	err := s.policy.Check(ds)
	if err == nil {
		return false
	}
	// What the destination could not write out is refused before any encoder
	// sees it, so nothing else will ever note it: said here, it still reaches
	// the diagnostics header and the ledger row rather than only the message.
	for _, d := range lost {
		s.diags = append(s.diags, s.policy.Apply(d))
	}
	s.pending, s.done = nil, true
	if s.err == nil {
		// A stream is the answer to a request that went, so its refusal is of
		// something sent, whatever the policy's own error assumes. It also
		// names the backend and model the caller was already told in the
		// stream's own metadata, so the refusal does not read as anonymous.
		e := AsError(err)
		e.Sent = true
		e.Provider, e.Model = s.meta.Provider, s.meta.Model
		s.err, s.ourErr = e, true
	}
	return true
}

// The candidate index is the upstream's to choose, and every distinct one costs
// an entry for as long as the turn runs. This matches what the fold will hold
// and sits above the most any backend advertises it will serve, so a request
// goose accepted is always trackable.
const maxChoices = 256

// mark refuses past the cap rather than growing without end, and remembers that
// it did: a turn holding candidates goose never tracked is not one it can call
// complete.
func (s *Stream) mark(set *map[int]bool, choice int) {
	if *set == nil {
		*set = map[int]bool{}
	}
	if _, known := (*set)[choice]; !known && len(*set) >= maxChoices {
		s.unheld = true
		return
	}
	(*set)[choice] = true
}

func (s *Stream) turnComplete() bool {
	if s.unheld || len(s.stopped) == 0 {
		return false
	}
	for choice := range s.open {
		if !s.stopped[choice] {
			return false
		}
	}
	return true
}

func (s *Stream) unfinished() bool {
	if s.unheld || len(s.unstopped) > 0 {
		return true
	}
	if len(s.open) == 0 && len(s.stopped) == 0 {
		// Nothing has arrived since the last turn closed, so whether the answer
		// is whole is whether that turn was.
		return !s.turnStop
	}
	return !s.turnComplete()
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
	s.ourErr = true
	s.err = &Error{
		Kind: KindUpstream, Code: WireNoFinish, Provider: s.meta.Provider, Model: s.meta.Model,
		Message: "the stream ended without a finish reason, so the answer is not known to be complete",
		Sent:    true, Retryable: true,
	}
}

func (s *Stream) fail(err error) {
	s.mu.Lock()
	s.done = true
	if s.err == nil {
		s.err = err
	}
	s.mu.Unlock()
	s.finish()
}

// Settle fixes what a stream reports it cost. The copy a caller is handed is
// meant to be the row that was booked rather than a second opinion on it.
func (s *Stream) Settle(u Usage) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.settled = &u
}

func (s *Stream) Usage() Usage {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.settled != nil {
		return *s.settled
	}
	out := s.total
	if !s.turn.Empty() {
		out = out.Add(s.turn)
	}
	if s.summary.Empty() {
		return out
	}
	return out.Cover(s.summary)
}

func (s *Stream) Err() error {
	s.mu.Lock()
	err, hide, ours := s.err, s.hide, s.ourErr
	s.mu.Unlock()
	if err == nil || hide == nil || ours {
		return err
	}
	if e := AsError(err); e != nil {
		return s.hidden(hide, e)
	}
	return err
}

func (s *Stream) Diagnostics() Diagnostics {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append(Diagnostics{}, s.diags...)
}

func (s *Stream) Note(d Diagnostic) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.diags = append(s.diags, d)
}

func (s *Stream) Record(a Attempted) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.attempts = append(s.attempts, a)
}

func (s *Stream) Attempts() []Attempted {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Attempted{}, s.attempts...)
}

// Redact sets what a backend's failure may be repeated as, for a proxy serving
// somebody else. It covers the error a frame carries and the one Err reports,
// including a frame already primed before it was set. What the backend actually
// said is kept for Unredacted, so the record an operator keeps is not the
// sentence a stranger is shown. goose's own words, such as a truncation, are
// not a backend's and do not pass through it.
func (s *Stream) Redact(fn func(*Error) *Error) {
	s.mu.Lock()
	s.hide = fn
	s.mu.Unlock()
}

// hidden is what a failure may be repeated as. It never runs under the mutex,
// because a hook is somebody else's code and may read the stream back.
func (s *Stream) hidden(hide func(*Error) *Error, e *Error) *Error {
	if hide == nil || e == nil {
		return e
	}
	return hide(e)
}

// shown is the frame a consumer sees. A frame belongs to whoever made it, so a
// redacted failure travels on a copy rather than over the original.
func (s *Stream) shown(hide func(*Error) *Error, f *Frame) *Frame {
	if hide == nil || f == nil || f.Err == nil {
		return f
	}
	out := *f
	out.Err = s.hidden(hide, f.Err)
	return &out
}

// Unredacted is the failure as the backend gave it, for the record an operator
// keeps rather than the reply a caller is shown.
func (s *Stream) Unredacted() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *Stream) OnFrame(fn func(*Frame)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.onFrame = fn
}

// OnEnd adds a callback for when the stream ends. One added after it has
// already ended runs at once rather than never, so a caller that registers late
// still books what the call cost.
func (s *Stream) OnEnd(fn func(*Stream)) {
	if fn == nil {
		return
	}
	s.mu.Lock()
	if s.ended {
		s.mu.Unlock()
		fn(s)
		return
	}
	s.onEnd = append(s.onEnd, fn)
	s.mu.Unlock()
}

func (s *Stream) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed, s.done = true, true
	closer := s.closer
	s.mu.Unlock()

	var err error
	if closer != nil {
		err = closer()
		if err != nil {
			s.mu.Lock()
			if s.err == nil {
				s.err = err
			}
			s.mu.Unlock()
		}
	}
	s.finish()
	return err
}

func sameColumns(a, b Usage) bool {
	return a.Prompt == b.Prompt && a.Completion == b.Completion &&
		a.CacheRead == b.CacheRead && a.CacheWrite == b.CacheWrite &&
		a.CacheWrite5m == b.CacheWrite5m && a.CacheWrite1h == b.CacheWrite1h &&
		a.Reasoning == b.Reasoning
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
