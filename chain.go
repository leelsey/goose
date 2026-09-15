// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"sync"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

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
		p, ok := c.providers[target.Provider]
		if !ok {
			lastErr = llm.Errorf(llm.KindNoRoute, "provider %q is not registered", target.Provider)
			continue
		}
		var diags llm.Diagnostics
		a := c.attemptFor(dec, target, i, req, cc, &diags)
		if !c.breaker.allow(target.Provider) {
			skipped = append(skipped, a.Policy.Apply(breakerOpen(target.Provider)))
			lastErr = llm.Errorf(llm.KindOverloaded, "%s has been failing and was skipped", target.Provider)
			continue
		}
		book := func(rec llm.Attempted) {
			attempts = append(attempts, rec)
			c.book(ctx, rec, cc, &diags)
		}
		a.Book = book
		base := llm.Attempted{
			Route: dec.Rule, Provider: target.Provider, Model: a.Model,
			Requested: req.Model, Index: i,
		}
		rejected := func(e *llm.Error) llm.Attempted {
			rec := base
			rec.Outcome, rec.Err = llm.OutcomeRejected, e.Message
			rec.Started, rec.Diagnostics = c.now(), diags
			return rec
		}

		if nerr := negotiate(a, p.Capabilities()); nerr != nil {
			c.breaker.abandoned(target.Provider)
			e := llm.AsError(nerr)
			book(rejected(e))
			lastErr = e
			if i == len(dec.Targets)-1 {
				break
			}
			continue
		}

		release, gerr := c.acquire(ctx, p)
		if gerr != nil {
			c.breaker.abandoned(target.Provider)
			e := llm.AsError(gerr)
			e.Provider = target.Provider
			a.Note(saturated(target.Provider))
			book(rejected(e))
			lastErr = e
			continue
		}

		started := c.now()
		actx, acancel := deadline(ctx, a.Limits.Timeout)
		resp, cerr := p.Complete(actx, a)
		acancel()
		release()
		rec := base
		rec.Started, rec.Elapsed, rec.Diagnostics = started, elapsed(started, c.now()), diags
		if resp != nil {
			rec.Usage = resp.Usage.Unpriced(p.Capabilities().Usage.Cost)
			resp.Usage = rec.Usage
		}

		if cerr != nil {
			e := llm.AsError(cerr)
			c.breaker.failed(target.Provider)
			rec.Sent, rec.Outcome, rec.Err, rec.Status = e.Sent, outcomeFor(e), e.Message, e.Status
			rec.Detail = e.Body
			book(rec)
			lastResp, lastErr = resp, e
			if !e.Retryable || i == len(dec.Targets)-1 {
				break
			}
			continue
		}

		c.breaker.succeeded(target.Provider)
		rec.Sent, rec.Outcome = true, llm.OutcomeOK
		book(rec)

		if resp == nil {
			lastErr = llm.Errorf(llm.KindInternal, "provider %q returned no response and no error", target.Provider)
			break
		}
		resp.Attempts = attempts
		resp.Provider = target.Provider
		if resp.Requested == "" {
			resp.Requested = req.Model
		}
		resp.Diagnostics = append(resp.Diagnostics, skipped...)
		resp.Diagnostics = append(resp.Diagnostics, diags...)
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
		p, ok := c.providers[target.Provider]
		if !ok {
			lastErr = llm.Errorf(llm.KindNoRoute, "provider %q is not registered", target.Provider)
			continue
		}
		var diags llm.Diagnostics
		a := c.attemptFor(dec, target, i, req, cc, &diags)
		if !c.breaker.allow(target.Provider) {
			skipped = append(skipped, a.Policy.Apply(breakerOpen(target.Provider)))
			lastErr = llm.Errorf(llm.KindOverloaded, "%s has been failing and was skipped", target.Provider)
			continue
		}
		book := func(rec llm.Attempted) {
			attempts = append(attempts, rec)
			c.book(ctx, rec, cc, &diags)
		}
		a.Book = book
		base := llm.Attempted{
			Route: dec.Rule, Provider: target.Provider, Model: a.Model,
			Requested: req.Model, Index: i,
		}
		rejected := func(e *llm.Error) llm.Attempted {
			rec := base
			rec.Outcome, rec.Err = llm.OutcomeRejected, e.Message
			rec.Started, rec.Diagnostics = c.now(), diags
			return rec
		}

		if nerr := negotiate(a, p.Capabilities()); nerr != nil {
			c.breaker.abandoned(target.Provider)
			e := llm.AsError(nerr)
			book(rejected(e))
			lastErr = e
			if i == len(dec.Targets)-1 {
				break
			}
			continue
		}

		release, gerr := c.acquire(ctx, p)
		if gerr != nil {
			c.breaker.abandoned(target.Provider)
			e := llm.AsError(gerr)
			a.Note(saturated(target.Provider))
			book(rejected(e))
			lastErr = e
			continue
		}

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
		if serr != nil {
			c.breaker.failed(target.Provider)
			acancel()
			release()
			if s != nil {
				_ = s.Close()
			}
			e := llm.AsError(serr)
			rec := base
			rec.Sent, rec.Outcome, rec.Err, rec.Status = e.Sent, outcomeFor(e), e.Message, e.Status
			rec.Detail = e.Body
			rec.Started, rec.Elapsed, rec.Diagnostics = started, elapsed(started, c.now()), diags
			if s != nil {
				rec.Usage = s.Usage()
			}
			book(rec)
			if s != nil {
				u := rec.Usage
				e.Usage = &u
			}
			lastErr = e
			if !e.Retryable || i == len(dec.Targets)-1 {
				break
			}
			continue
		}
		if s == nil {
			acancel()
			release()
			lastErr = llm.Errorf(llm.KindInternal, "provider %q returned no stream and no error", target.Provider)
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
		for _, d := range diags {
			s.Note(d)
		}
		grace := c.abandonGrace()
		var releaseOnce, bookOnce sync.Once
		free := func() { releaseOnce.Do(func() { release(); acancel() }) }
		stopWatch := context.AfterFunc(ctx, func() {
			free()
			time.AfterFunc(grace, func() {
				bookOnce.Do(func() {
					rec := base
					rec.Sent, rec.Outcome = true, llm.OutcomeCancelled
					rec.Started, rec.Elapsed = started, elapsed(started, c.now())
					rec.Usage = llm.Usage{}.Unmeasured()
					rec.Err = "the stream was abandoned without being closed, so what it cost is not known"
					c.book(context.WithoutCancel(ctx), rec, cc, nil)
				})
			})
		})
		s.OnEnd(func(done *llm.Stream) {
			stopWatch()
			free()
			e := llm.AsError(done.Err())
			rec := base
			rec.Sent, rec.Outcome = true, llm.OutcomeOK
			rec.Started, rec.Elapsed = started, elapsed(started, c.now())
			rec.Usage = done.Usage().Unmeasured().Unpriced(p.Capabilities().Usage.Cost)
			rec.Diagnostics = done.Diagnostics()
			if e != nil {
				rec.Outcome, rec.Err, rec.Status = outcomeFor(e), e.Message, e.Status
				rec.Detail = e.Body
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

func saturated(provider string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.PolicySaturated, Class: llm.ClassPolicy, Stage: llm.StagePlan,
		Severity: llm.SeverityInfo, Provider: provider,
		Detail: "this backend is at its concurrency limit, so the attempt was shed before anything was sent",
	}
}
