// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"errors"
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
	release  func()
	book     func(llm.Attempted)
}

// prepare readies one target, having already booked and noted whatever a
// failure before the send owed. A target that fails here is one to walk past,
// not one to stop on: the next link may well serve.
func (c *Client) prepare(
	ctx context.Context, dec route.Decision, i int, target route.Target,
	req *llm.Request, cc callConfig, attempts *[]llm.Attempted, skipped *llm.Diagnostics,
) (*pending, *llm.Error) {
	p, ok := c.providers[target.Provider]
	if !ok {
		return nil, llm.Errorf(llm.KindNoRoute, "provider %q is not registered", target.Provider)
	}
	diags := new(llm.Diagnostics)
	a := c.attemptFor(dec, target, i, req, cc, diags)
	if !c.breaker.allow(target.Provider) {
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
		c.breaker.abandoned(target.Provider)
		e := llm.AsError(nerr)
		book(rejected(e))
		return nil, e
	}

	release, gerr := c.acquire(ctx, p)
	if gerr != nil {
		c.breaker.abandoned(target.Provider)
		e := llm.AsError(gerr)
		e.Provider = target.Provider
		a.Note(saturated(target.Provider))
		book(rejected(e))
		return nil, e
	}

	return &pending{
		provider: p, attempt: a, base: base, diags: diags,
		release: release, book: book,
	}, nil
}

func (c *Client) Chat(ctx context.Context, req *llm.Request, opts ...CallOption) (*llm.Response, error) {
	cc := c.config(false, opts)
	dec, err := c.plan(req, cc)
	if err != nil {
		return nil, err
	}

	var attempts []llm.Attempted
	var lastResp *llm.Response
	var lastErr error
	var skipped llm.Diagnostics

	for i, target := range dec.Targets {
		pend, perr := c.prepare(ctx, dec, i, target, req, cc, &attempts, &skipped)
		if perr != nil {
			lastErr = perr
			if ctx.Err() != nil {
				break
			}
			continue
		}
		p, a := pend.provider, pend.attempt

		started := c.now()
		actx, acancel := deadline(ctx, a.Limits.Timeout)
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
			c.fault(target.Provider, ctx.Err() != nil)
			rec.Sent, rec.Outcome, rec.Err, rec.Status = e.Sent, outcomeFor(e), c.reason(e), e.Status
			rec.Detail = c.detail(e)
			pend.book(rec)
			lastResp, lastErr = resp, e
			if walled || ctx.Err() != nil || !fallbackWorthy(e) || i == len(dec.Targets)-1 {
				break
			}
			continue
		}

		if resp == nil {
			// Nothing back is not an answer. Booking it as one bills a call
			// that worked and tells the breaker this backend is healthy, while
			// the caller is handed a failure.
			c.breaker.abandoned(target.Provider)
			e := llm.AsError(llm.Errorf(llm.KindInternal, "provider %q returned no response and no error", target.Provider))
			rec.Sent, rec.Outcome, rec.Err = true, outcomeFor(e), c.reason(e)
			pend.book(rec)
			lastErr = e
			break
		}

		c.breaker.succeeded(target.Provider)
		rec.Sent, rec.Outcome = true, llm.OutcomeOK
		pend.book(rec)
		resp.Attempts = attempts
		resp.Provider = target.Provider
		if resp.Requested == "" {
			resp.Requested = req.Model
		}
		resp.Diagnostics = append(resp.Diagnostics, skipped...)
		resp.Diagnostics = append(resp.Diagnostics, *pend.diags...)
		if err := a.Policy.Check(resp.Diagnostics); err != nil {
			return resp, err
		}
		return resp, nil
	}

	if lastErr == nil {
		lastErr = llm.Errorf(llm.KindNoRoute, "the route produced no usable provider")
	}
	e := llm.AsError(lastErr)
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
	var lastErr error
	var skipped llm.Diagnostics

	for i, target := range dec.Targets {
		pend, perr := c.prepare(ctx, dec, i, target, req, cc, &attempts, &skipped)
		if perr != nil {
			lastErr = perr
			if ctx.Err() != nil {
				break
			}
			continue
		}
		p, a := pend.provider, pend.attempt
		release, book, base := pend.release, pend.book, pend.base

		started := c.now()
		actx, acancel, primed := untilPrimed(ctx, a.Limits.Timeout)
		var s *llm.Stream
		var serr error
		if p.Capabilities().Streaming {
			s, serr = p.Stream(actx, a)
		} else {
			var resp *llm.Response
			resp, serr = p.Complete(actx, a)
			if resp != nil {
				s = wirecore.Synthesise(resp)
			}
		}
		if serr == nil && s != nil {
			serr = s.Prime()
		}
		primed()
		walled := ctx.Err() == nil && errors.Is(context.Cause(actx), context.DeadlineExceeded)
		if serr != nil {
			c.fault(target.Provider, ctx.Err() != nil)
			acancel()
			release()
			if s != nil {
				_ = s.Close()
			}
			e := llm.AsError(serr)
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
			lastErr = e
			if walled || ctx.Err() != nil || !fallbackWorthy(e) || i == len(dec.Targets)-1 {
				break
			}
			continue
		}
		if s == nil {
			acancel()
			release()
			c.breaker.abandoned(target.Provider)
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

		s.Tolerate(a.Policy.Allow...)
		c.breaker.succeeded(target.Provider)
		for _, rec := range attempts {
			s.Record(rec)
		}
		for _, d := range skipped {
			s.Note(d)
		}
		for _, d := range *pend.diags {
			s.Note(d)
		}
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
				rec.Outcome, rec.Err, rec.Status = outcomeFor(e), c.reason(e), e.Status
				rec.Detail = c.detail(e)
			}
			done.Record(rec)
			bookOnce.Do(func() { c.book(context.WithoutCancel(ctx), rec, cc, done) })
		})
		return s, nil
	}

	if lastErr == nil {
		lastErr = llm.Errorf(llm.KindNoRoute, "the route produced no usable provider")
	}
	e := llm.AsError(lastErr)
	e.Route = dec.Rule
	e.Diagnostics = append(e.Diagnostics, skipped...)
	return nil, e
}

// A target's failure is worth falling back from unless it would fail the same
// way everywhere: the request itself is malformed, or the caller has gone. A
// timeout is worth it only while the caller's own context is still alive, which
// the callers check for themselves.
// fault counts a failure against a backend, unless what failed was not the
// backend. A caller that has gone — hung up or out of its own time — says
// nothing about it, and holding that against it would take a healthy backend
// out of the chain. The probe slot still goes back either way, or a backend
// whose one probe met a departing caller would never be tried again.
func (c *Client) fault(provider string, gone bool) {
	if gone {
		c.breaker.abandoned(provider)
		return
	}
	c.breaker.failed(provider)
}

func fallbackWorthy(e *llm.Error) bool {
	switch e.Kind {
	case llm.KindInvalidRequest, llm.KindUnsupported, llm.KindCancelled:
		return false
	}
	return true
}

func saturated(provider string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.PolicySaturated, Class: llm.ClassPolicy, Stage: llm.StagePlan,
		Severity: llm.SeverityInfo, Provider: provider,
		Detail: "this backend is at its concurrency limit, so the attempt was shed before anything was sent",
	}
}
