// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
)

// pending is one route target readied for a send: what to call, what to send,
// and the bookkeeping that follows either way.
type pending struct {
	provider llm.Provider
	attempt  *llm.Attempt
	base     llm.Attempted
	diags    *llm.Diagnostics
	probe    probe
	release  func()
	book     func(llm.Attempted)
	// asked says the model is the caller's own, named by no rule or target.
	asked bool
	// started is when the attempt began, and due when it must end, which the
	// wait for a slot and the send share.
	started, due time.Time
	// waited says the slot had to be waited for, so the send was given only
	// what the queue left of the budget.
	waited bool
}

// prepare readies one target, having already booked and noted whatever a
// failure before the send owed. A target that fails here is one to walk past,
// not one to stop on: the next link may well serve.
func (c *Client) prepare(
	ctx context.Context, dec route.Decision, i int, target route.Target,
	req *llm.Request, group []llm.Tool, cc callConfig, attempts *[]llm.Attempted, skipped *llm.Diagnostics,
) (*pending, *llm.Error) {
	p, ok := c.providers[target.Provider]
	if !ok {
		return nil, llm.Errorf(llm.KindNoRoute, "provider %q is not registered", target.Provider)
	}
	diags := new(llm.Diagnostics)
	a := c.attemptFor(dec, target, i, req, group, cc, diags)
	for _, d := range cc.noted {
		diags.Note(a.Policy.Apply(d))
	}
	pr, ok := c.breaker.allow(target.Provider)
	if !ok {
		*skipped = append(*skipped, a.Policy.Apply(breakerOpen(target.Provider)))
		return nil, llm.Errorf(llm.KindOverloaded, "%s has been failing and was skipped", target.Provider)
	}
	book := func(rec llm.Attempted) {
		*attempts = append(*attempts, rec)
		c.book(ctx, rec, cc, diags)
	}
	a.Book = book
	base := llm.Attempted{
		Route: dec.Rule, Provider: target.Provider, Model: a.Model,
		Requested: req.Model, Index: i,
	}
	rejected := func(e *llm.Error) llm.Attempted {
		rec := base
		rec.Outcome, rec.Err = llm.OutcomeRejected, c.reason(e)
		rec.Started, rec.Diagnostics = c.now(), *diags
		return rec
	}

	if nerr := negotiate(a, p.Capabilities()); nerr != nil {
		c.breaker.abandoned(target.Provider, pr)
		e := llm.AsError(nerr)
		book(rejected(e))
		return nil, e
	}
	// What planning noted has already been through the caller's policy, so a
	// loss it will not take is known now. Refused here it costs nothing, and
	// the next link may carry it whole.
	refuse := func(lerr error) (*pending, *llm.Error) {
		c.breaker.abandoned(target.Provider, pr)
		e := llm.AsError(lerr)
		e.Provider, e.Model = target.Provider, a.Model
		book(rejected(e))
		return nil, e
	}
	if lerr := a.Policy.Check(*diags); lerr != nil {
		return refuse(lerr)
	}
	// A provider that can say what building its request will lose, or that
	// it will not send at all, is checked the same way, ahead of its gate: a
	// request it would refuse once the request is actually built is refused
	// now, so the call never queues for a slot it was always going to be
	// turned away from. What it found is its own to note once it builds the
	// request, so it is noted here only where it turns the attempt away, for
	// the row to say why.
	if pc, ok := p.(llm.Precheck); ok {
		found, lerr := pc.Precheck(a)
		if lerr == nil {
			lerr = a.Policy.Check(found)
		}
		if lerr != nil {
			for _, d := range found {
				diags.Note(a.Policy.Apply(d))
			}
			return refuse(lerr)
		}
	}

	started, due := c.now(), wall(a.Limits.Timeout)
	release, waited, gerr := c.acquire(ctx, p, due)
	if gerr != nil {
		c.breaker.abandoned(target.Provider, pr)
		e := llm.AsError(gerr)
		e.Provider = target.Provider
		a.Note(saturated(target.Provider, e))
		rec := rejected(e)
		if queued(e) {
			// Not turned away but kept waiting, so its row says how the wait
			// ended and how long it was.
			rec.Outcome = outcomeFor(e)
			rec.Started, rec.Elapsed = started, elapsed(started, c.now())
		}
		book(rec)
		return nil, e
	}

	return &pending{
		provider: p, attempt: a, base: base, diags: diags, probe: pr,
		release: release, book: book,
		asked:   target.FromCall || target.Model == "",
		started: started, due: due, waited: waited,
	}, nil
}

// sequence is the order a call tries its route's links in, by their places in
// it, and how many of them lead. A request that offers the model tools, and
// does not forbid it to call one, goes first to the links that carry every one
// of them, in the operator's order. A link that would have to strip some is
// reached only once all of those have failed, been skipped or refused, since an
// answer without the tools answers a different question and the next link may
// well have them. Where no link carries them, none is put behind another.
func (c *Client) sequence(dec route.Decision, req *llm.Request, group []llm.Tool, renders llm.Renders) (order []int, lead int) {
	offered := *req
	if len(offered.Tools) == 0 {
		offered.Tools = group
	}
	order = make([]int, 0, len(dec.Targets))
	defers := offerOf(&offered).any() && !declined(&offered)
	var later []int
	for i, t := range dec.Targets {
		p, ok := c.providers[t.Provider]
		if defers && !(ok && carries(p.Capabilities(), &offered, renders)) {
			later = append(later, i)
			continue
		}
		order = append(order, i)
	}
	if len(order) == 0 {
		return later, len(later)
	}
	return append(order, later...), len(order)
}

// owes says whether a link's failure is the one the caller is now owed in
// place of the last. A link refused before anything was sent, whether prepare
// turned it away or its own provider refused to send what it had built, says
// no more than that it could not have served either. That never takes the
// place of a failure that reached a backend, which is the one that may have
// been billed and the one a retry may cure. Nor does it take the place of any
// failure when the link was put behind the others, since it is reached only
// after they have all failed. A wait at its gate that ran out is the attempt
// ending, not a refusal, and stands.
func owes(last, e *llm.Error, behind bool) bool {
	if last == nil || e.Sent || queued(e) {
		return true
	}
	return !behind && !last.Sent
}

func (c *Client) Chat(ctx context.Context, req *llm.Request, opts ...CallOption) (*llm.Response, error) {
	cc := c.config(false, opts)
	dec, err := c.plan(req, cc)
	if err != nil {
		return nil, err
	}

	var attempts []llm.Attempted
	// lastResp is only ever what the link that failed with lastErr answered.
	var lastResp *llm.Response
	var lastErr *llm.Error
	var skipped llm.Diagnostics

	group := c.groupTools(dec)
	order, lead := c.sequence(dec, req, group, cc.renders)
	for n, i := range order {
		target := dec.Targets[i]
		pend, perr := c.prepare(ctx, dec, i, target, req, group, cc, &attempts, &skipped)
		if perr != nil {
			if owes(lastErr, perr, n >= lead) {
				lastResp, lastErr = nil, perr
			}
			if ctx.Err() != nil || queued(perr) || !fallbackWorthy(perr) {
				break
			}
			continue
		}
		p, a := pend.provider, pend.attempt

		started := pend.started
		actx, acancel := deadline(ctx, pend.due)
		noted := len(*pend.diags)
		resp, cerr := p.Complete(actx, a)
		// Our own wall, not the backend's failing. The caller has already waited
		// the whole attempt budget, and the next link would be given its own, so
		// walking the chain only makes them wait it twice.
		walled := ctx.Err() == nil && errors.Is(context.Cause(actx), context.DeadlineExceeded)
		acancel()
		pend.release()
		rec := pend.base
		rec.Started, rec.Elapsed, rec.Diagnostics = started, elapsed(started, c.now()), *pend.diags
		if resp != nil {
			rec.Usage = resp.Usage.Unmeasured().Unpriced(p.Capabilities().Usage.Cost)
			resp.Usage = rec.Usage
		}

		if cerr != nil {
			e := llm.AsError(cerr)
			if resp == nil && e.Sent {
				// The request went, so the vendor may well have billed it.
				// Nobody counted it, which is not the same as it having cost
				// nothing, and every row for a sent attempt says which.
				rec.Usage = rec.Usage.Unmeasured()
			}
			c.fault(pend, ctx.Err() != nil, walled, e)
			rec.Sent, rec.Outcome, rec.Err, rec.Status = e.Sent, outcomeFor(e), c.reason(e), e.Status
			rec.Detail = c.detail(e)
			pend.book(rec)
			// A link's own refusal, found only once its provider built the
			// request, is owed the same way as one prepare found first.
			if owes(lastErr, e, n >= lead) {
				lastResp, lastErr = resp, e
			}
			if walled || ctx.Err() != nil || !fallbackWorthy(e) || n == len(order)-1 {
				break
			}
			continue
		}

		if resp == nil {
			// Nothing back is not an answer. Booking it as one bills a call
			// that worked and tells the breaker this backend is healthy, while
			// the caller is handed a failure.
			c.breaker.abandoned(target.Provider, pend.probe)
			e := llm.AsError(llm.Errorf(llm.KindInternal, "provider %q returned no response and no error", target.Provider))
			rec.Sent, rec.Outcome, rec.Err = true, outcomeFor(e), c.reason(e)
			rec.Usage = rec.Usage.Unmeasured()
			pend.book(rec)
			lastResp, lastErr = nil, e
			break
		}

		c.breaker.succeeded(target.Provider)
		rec.Sent, rec.Outcome = true, llm.OutcomeOK
		pend.book(rec)
		resp.Attempts = attempts
		answered(resp, pend, noted)
		resp.Diagnostics = append(resp.Diagnostics, skipped...)
		resp.Diagnostics = append(resp.Diagnostics, *pend.diags...)
		resp.Policy = a.Policy
		if err := a.Policy.Check(slices.Concat(resp.Diagnostics, a.Renders.Unrendered(resp))); err != nil {
			// Refused once it was answered, so the request went and was billed,
			// which a caller reading only the failure must not take for free.
			e := llm.AsError(err)
			e.Provider, e.Model, e.Sent = a.Provider, a.Model, true
			return resp, e
		}
		return resp, nil
	}

	e := lastErr
	if e == nil {
		e = llm.Errorf(llm.KindNoRoute, "the route produced no usable provider")
	}
	e.Route = dec.Rule
	e.Diagnostics = append(e.Diagnostics, skipped...)
	if lastResp != nil {
		lastResp.Attempts = attempts
		lastResp.Diagnostics = append(lastResp.Diagnostics, skipped...)
		e.Usage = &lastResp.Usage
	}
	return lastResp, e
}

func (c *Client) Stream(ctx context.Context, req *llm.Request, opts ...CallOption) (*llm.Stream, error) {
	cc := c.config(true, opts)
	dec, err := c.plan(req, cc)
	if err != nil {
		return nil, err
	}

	var attempts []llm.Attempted
	var lastErr *llm.Error
	var skipped llm.Diagnostics

	group := c.groupTools(dec)
	order, lead := c.sequence(dec, req, group, cc.renders)
	for n, i := range order {
		target := dec.Targets[i]
		pend, perr := c.prepare(ctx, dec, i, target, req, group, cc, &attempts, &skipped)
		if perr != nil {
			if owes(lastErr, perr, n >= lead) {
				lastErr = perr
			}
			if ctx.Err() != nil || queued(perr) || !fallbackWorthy(perr) {
				break
			}
			continue
		}
		p, a := pend.provider, pend.attempt
		release, book, base := pend.release, pend.book, pend.base

		started := pend.started
		actx, acancel, primed := untilPrimed(ctx, pend.due)
		var s *llm.Stream
		var serr error
		if p.Capabilities().Streaming {
			s, serr = p.Stream(actx, a)
		} else {
			var resp *llm.Response
			noted := len(*pend.diags)
			resp, serr = p.Complete(actx, a)
			if resp != nil {
				answered(resp, pend, noted)
				s = wirecore.Synthesise(resp, a.Renders)
			}
		}
		if serr == nil && s != nil {
			// Priming can run to the end of a stream that brings no content,
			// and what the caller lives with is known before that.
			s.Tolerate(a.Policy.Allow...)
			serr = s.Prime()
		}
		primed()
		walled := ctx.Err() == nil && errors.Is(context.Cause(actx), context.DeadlineExceeded)
		if serr != nil {
			e := llm.AsError(serr)
			c.fault(pend, ctx.Err() != nil, walled, e)
			acancel()
			release()
			if s != nil {
				_ = s.Close()
			}
			rec := base
			rec.Sent, rec.Outcome, rec.Err, rec.Status = e.Sent, outcomeFor(e), c.reason(e), e.Status
			rec.Detail = c.detail(e)
			rec.Started, rec.Elapsed, rec.Diagnostics = started, elapsed(started, c.now()), *pend.diags
			switch {
			case s != nil:
				// Read the same way a row that succeeded is read: what a
				// backend did not count is uncounted, not nothing.
				rec.Usage = s.Usage().Unmeasured().Unpriced(p.Capabilities().Usage.Cost)
			case e.Sent:
				rec.Usage = rec.Usage.Unmeasured()
			}
			book(rec)
			if s != nil {
				u := rec.Usage
				e.Usage = &u
			}
			// A link's own refusal, found only once its provider built the
			// request, is owed the same way as one prepare found first.
			if owes(lastErr, e, n >= lead) {
				lastErr = e
			}
			if walled || ctx.Err() != nil || !fallbackWorthy(e) || n == len(order)-1 {
				break
			}
			continue
		}
		if s == nil {
			acancel()
			release()
			c.breaker.abandoned(target.Provider, pend.probe)
			// The attempt was made, so it is a row like any other. Saying
			// nothing about it would leave a call the vendor may have billed
			// with no line in the ledger at all.
			e := llm.Errorf(llm.KindInternal, "provider %q returned no stream and no error", target.Provider)
			rec := base
			rec.Sent, rec.Outcome, rec.Err = true, outcomeFor(e), c.reason(llm.AsError(e))
			rec.Started, rec.Elapsed, rec.Diagnostics = started, elapsed(started, c.now()), *pend.diags
			rec.Usage = llm.Usage{}.Unmeasured()
			book(rec)
			lastErr = e
			break
		}

		for _, rec := range attempts {
			s.Record(rec)
		}
		for _, d := range skipped {
			s.Note(d)
		}
		for _, d := range *pend.diags {
			s.Note(d)
		}
		// Enforced once it holds what the attempt noted, so a stream answers to
		// everything an unstreamed answer is checked against.
		s.Enforce(a.Policy, a.Renders)
		grace := c.abandonGrace()
		var releaseOnce, bookOnce sync.Once
		var abandonMu sync.Mutex
		var abandon *time.Timer
		var abandonOff bool
		free := func() { releaseOnce.Do(func() { release(); acancel() }) }
		stopWatch := context.AfterFunc(ctx, func() {
			free()
			abandonMu.Lock()
			if abandonOff {
				abandonMu.Unlock()
				return
			}
			abandon = time.AfterFunc(grace, func() {
				bookOnce.Do(func() {
					// Nothing is known of how the stream went, but a probe it
					// holds goes back, or the backend is never tried again.
					c.breaker.abandoned(target.Provider, pend.probe)
					rec := base
					rec.Sent, rec.Outcome = true, llm.OutcomeCancelled
					rec.Started, rec.Elapsed = started, elapsed(started, c.now())
					rec.Usage = llm.Usage{}.Unmeasured()
					rec.Err = "the stream was abandoned without being closed, so what it cost is not known"
					c.book(context.WithoutCancel(ctx), rec, cc, nil)
				})
			})
			abandonMu.Unlock()
		})
		s.OnEnd(func(done *llm.Stream) {
			stopWatch()
			// The stream ended, so the row it earned is the one to book. Left
			// armed, the abandonment timer would take that place with a row
			// saying the cost is not known, and hold this call's state for the
			// grace period after everything was already done.
			// The flag matters as much as the stop: the watch may not have armed
			// the timer yet, and arming one after this point would hold the call
			// for the whole grace after it was already done.
			abandonMu.Lock()
			abandonOff = true
			if abandon != nil {
				abandon.Stop()
			}
			abandonMu.Unlock()
			free()
			// The ledger keeps what the backend said, whatever the caller is shown.
			e := llm.AsError(done.Unredacted())
			rec := base
			rec.Sent, rec.Outcome = true, llm.OutcomeOK
			rec.Started, rec.Elapsed = started, elapsed(started, c.now())
			rec.Usage = done.Usage().Unmeasured().Unpriced(p.Capabilities().Usage.Cost)
			done.Settle(rec.Usage)
			rec.Diagnostics = done.Diagnostics()
			if e != nil {
				rec.Err, rec.Status = c.reason(e), e.Status
				rec.Detail = c.detail(e)
				if e.Kind != llm.KindLoss {
					// A loss the stream's own policy refused is goose's, not the
					// backend's: the answer was still served and billed in full, as
					// the same refusal reads on the unstreamed row, which is booked
					// before its own post-send check can touch it.
					rec.Outcome = outcomeFor(e)
				}
			}
			done.Record(rec)
			bookOnce.Do(func() {
				c.ended(pend, ctx.Err() != nil, e)
				c.book(context.WithoutCancel(ctx), rec, cc, done)
			})
		})
		return s, nil
	}

	e := lastErr
	if e == nil {
		e = llm.Errorf(llm.KindNoRoute, "the route produced no usable provider")
	}
	e.Route = dec.Rule
	e.Diagnostics = append(e.Diagnostics, skipped...)
	return nil, e
}

// answered names an answer for the link that served it and for the model the
// caller asked for, and leaves out of it what the attempt was also told while
// the provider answered. A provider hands what it noted back on its answer as
// well, so a caller of its own sees it, and the chain already holds the
// attempt's copy, which has been through the caller's policy where the
// answer's has not. noted is how much the attempt held before the call.
func answered(resp *llm.Response, pend *pending, noted int) {
	resp.Provider = pend.base.Provider
	if resp.Requested == "" {
		resp.Requested = pend.base.Requested
	}
	held := map[llm.Diagnostic]int{}
	for _, d := range (*pend.diags)[noted:] {
		held[unweighed(d)]++
	}
	if len(held) == 0 {
		return
	}
	kept := resp.Diagnostics[:0:0]
	for _, d := range resp.Diagnostics {
		if k := unweighed(d); held[k] > 0 {
			held[k]--
			continue
		}
		kept = append(kept, d)
	}
	resp.Diagnostics = kept
}

// unweighed is what a note says, leaving out what an attempt fills in when it
// is noted, whose it was and what the policy makes of it, so the attempt's
// copy and the answer's read as the one note.
func unweighed(d llm.Diagnostic) llm.Diagnostic {
	d.Severity, d.Provider, d.Model, d.Format = 0, "", "", ""
	return d
}

// ended gives the breaker its verdict on a stream once the stream is over,
// not once it was handed over: a backend can fail after its first frame of
// content as surely as before it. A stream that ran to its end, that its
// caller left or that the caller's own policy refused was the backend serving.
func (c *Client) ended(pend *pending, gone bool, e *llm.Error) {
	if e == nil || gone || e.Kind == llm.KindLoss {
		c.breaker.succeeded(pend.base.Provider)
		return
	}
	c.fault(pend, false, false, e)
}

// fault counts a failure against a backend, unless what failed was not the
// backend. A caller that has gone — hung up or out of its own time — says
// nothing about it, and nor does a failure the caller brought with it, or the
// attempt's own wall falling on a send that was given only what the wait for a
// slot left of it: the time went on calls ahead of it at the gate. The
// breaker is shared by every caller, so holding any of these against the
// backend would let one of them take it out of the chain for all the rest. The
// probe slot still goes back either way, or a backend whose one probe met such
// a call would never be tried again.
func (c *Client) fault(pend *pending, gone, walled bool, e *llm.Error) {
	if gone || walled && pend.waited || !blames(e, pend.provider.Capabilities(), pend.asked) {
		c.breaker.abandoned(pend.base.Provider, pend.probe)
		return
	}
	c.breaker.failed(pend.base.Provider, pend.probe)
}

// blames says whether a failure is the backend's own. A malformed or oversized
// request — whether the backend refused it or goose did before sending — a
// capability asked of a backend that lacks it and a loss the caller would not
// take would fail the same way on a healthy one. So would a model the caller
// named that the backend does not have; one the operator named is the
// backend's to serve. A refused or rate-limited key answers for whoever holds
// it, so on a backend with no key of its own it is the caller's, whether the
// caller's key was sent or none was. A backend that could not be reached is its
// own failure, sent or not.
func blames(e *llm.Error, caps llm.Capabilities, asked bool) bool {
	switch e.Kind {
	case llm.KindInvalidRequest, llm.KindUnsupported, llm.KindLoss, llm.KindCancelled, llm.KindNoRoute:
		return false
	}
	if asked && (e.Kind == llm.KindNotFound || e.Status == http.StatusNotFound) {
		return false
	}
	if !caps.ClientKey {
		return true
	}
	switch {
	case e.Kind == llm.KindAuth, e.Kind == llm.KindPermission, e.Kind == llm.KindRateLimit,
		e.Status == http.StatusUnauthorized, e.Status == http.StatusForbidden,
		e.Status == http.StatusTooManyRequests:
		return false
	}
	return true
}

// A target's failure is worth falling back from unless it would fail the same
// way everywhere: the request itself is malformed, or the caller has gone. Too
// large for one backend is not too large for the next, which has a ceiling of
// its own. A timeout is worth it only while the caller's own context is still
// alive, which the callers check for themselves. A capability one backend lacks
// is its own: what none of them has was refused while planning.
func fallbackWorthy(e *llm.Error) bool {
	switch e.Kind {
	case llm.KindInvalidRequest:
		return e.Status == http.StatusRequestEntityTooLarge
	case llm.KindCancelled:
		return false
	}
	return true
}

// queued says an attempt waited at its backend's gate and ended there, out of
// time or with its caller gone, rather than being turned away at once. Its
// budget is spent as surely as one sent would have spent it, so the chain goes
// no further.
func queued(e *llm.Error) bool {
	return e.Kind == llm.KindTimeout || e.Kind == llm.KindCancelled
}

func saturated(provider string, e *llm.Error) llm.Diagnostic {
	d := llm.Diagnostic{
		Code: llm.PolicySaturated, Class: llm.ClassPolicy, Stage: llm.StagePlan,
		Severity: llm.SeverityInfo, Provider: provider,
		Detail: "this backend is at its concurrency limit, so the attempt was shed before anything was sent",
	}
	if queued(e) {
		d.Detail = "this backend was at its concurrency limit, and the attempt ended while it waited for a slot, before anything was sent"
	}
	return d
}
