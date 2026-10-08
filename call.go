// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"maps"
	"net/http"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

type callConfig struct {
	format     string
	renders    llm.Renders
	path       string
	header     http.Header
	session    string
	provider   string
	credential llm.Credential
	policy     *llm.LossPolicy
	limits     *llm.Limits
	noted      llm.Diagnostics
	stream     bool
}

type CallOption func(*callConfig)

func WithFormat(name string) CallOption {
	return func(c *callConfig) { c.format = name }
}

func WithRenders(r llm.Renders) CallOption {
	return func(c *callConfig) { c.renders = r }
}

func WithPath(p string) CallOption {
	return func(c *callConfig) { c.path = p }
}

func WithHeader(h http.Header) CallOption {
	return func(c *callConfig) { c.header = h }
}

func WithSession(id string) CallOption {
	return func(c *callConfig) { c.session = id }
}

func WithSessionProvider(id string) CallOption {
	return func(c *callConfig) { c.provider = id }
}

func WithCredential(cr llm.Credential) CallOption {
	return func(c *callConfig) { c.credential = cr }
}

func WithCallPolicy(p llm.LossPolicy) CallOption {
	return func(c *callConfig) { c.policy = &p }
}

func WithCallLimits(l llm.Limits) CallOption {
	return func(c *callConfig) { c.limits = &l }
}

// WithDiagnostics hands the call what was noted of the request before it
// reached the client, such as a proxy's decoding of its caller's wire. They
// are held to the call's policy before any link is tried, and travel on every
// attempt's row and on the answer.
func WithDiagnostics(ds llm.Diagnostics) CallOption {
	return func(c *callConfig) { c.noted = append(c.noted, ds...) }
}

func (c *Client) config(stream bool, opts []CallOption) callConfig {
	cc := callConfig{stream: stream, header: http.Header{}}
	for _, o := range opts {
		if o != nil {
			o(&cc)
		}
	}
	return cc
}

func (c *Client) plan(req *llm.Request, cc callConfig) (route.Decision, error) {
	if req == nil {
		return route.Decision{}, llm.Errorf(llm.KindInvalidRequest, "nil request")
	}
	if err := validate(req); err != nil {
		return route.Decision{}, err
	}
	if err := renderable(cc.renders, req); err != nil {
		return route.Decision{}, err
	}
	call := &route.Call{
		Model: req.Model, Format: cc.format, Path: cc.path, Header: cc.header,
		Stream: cc.stream, Session: cc.session, Provider: cc.provider,
	}
	dec, err := c.table.Route(call)
	if err != nil {
		return dec, err
	}
	// What was noted before the call is about the request itself, so a loss
	// there is one no link could avoid, and it is refused before any is tried.
	if err := c.policyFor(dec, cc).Check(cc.noted); err != nil {
		e := llm.AsError(err)
		e.Route = dec.Rule
		return dec, e
	}
	return dec, nil
}

func renderable(renders llm.Renders, req *llm.Request) error {
	if req.Candidates > 1 && req.Candidates > renders.Candidates && renders.Declared() {
		return &llm.Error{
			Kind: llm.KindInvalidRequest, Code: llm.CapCandidatesOne,
			Message: "this format carries one completion; asking for more would deliver a fraction of the answer",
		}
	}
	if req.Thinking != nil && req.Thinking.Enabled && renders.Declared() && !renders.Thinking {
		return &llm.Error{
			Kind: llm.KindInvalidRequest, Code: llm.DropThinking,
			Message: "this format has no field for reasoning, so thinking would be billed and not delivered",
		}
	}
	return nil
}

func (c *Client) policyFor(dec route.Decision, cc callConfig) llm.LossPolicy {
	switch {
	case cc.policy != nil:
		return *cc.policy
	case dec.PolicySet:
		return dec.Policy
	default:
		return c.policy
	}
}

// A narrower limit is a refinement of the wider one, not a replacement for it:
// a route that pins a frame size must not thereby lose the client's deadline.
func (c *Client) limitsFor(dec route.Decision, cc callConfig) llm.Limits {
	out := c.limits
	if dec.Limits != (llm.Limits{}) {
		out = fillLimits(dec.Limits, out)
	}
	if cc.limits != nil {
		out = fillLimits(*cc.limits, out)
	}
	return out
}

func fillLimits(over, under llm.Limits) llm.Limits {
	if over.MaxRequestBytes == 0 {
		over.MaxRequestBytes = under.MaxRequestBytes
	}
	if over.MaxResponseBytes == 0 {
		over.MaxResponseBytes = under.MaxResponseBytes
	}
	if over.MaxFrameBytes == 0 {
		over.MaxFrameBytes = under.MaxFrameBytes
	}
	if over.MaxBufferBytes == 0 {
		over.MaxBufferBytes = under.MaxBufferBytes
	}
	if over.Timeout == 0 {
		over.Timeout = under.Timeout
	}
	return over
}

func (c *Client) attemptFor(dec route.Decision, target route.Target, index int, req *llm.Request, group []llm.Tool, cc callConfig, rec llm.Recorder) *llm.Attempt {
	a := &llm.Attempt{
		Request:    req.Clone(),
		Provider:   target.Provider,
		Model:      target.Model,
		Stream:     cc.stream,
		Format:     cc.format,
		Route:      dec.Rule,
		Index:      index,
		Credential: cc.credential,
		Renders:    cc.renders,
		Policy:     c.policyFor(dec, cc),
		Limits:     c.limitsFor(dec, cc),
		Recorder:   rec,
		Redact:     c.redact,
	}
	if a.Model == "" {
		a.Model = req.Model
	}
	a.Request.Model = a.Model
	switch {
	case cc.session == "":
	case cc.provider == "" || cc.provider == target.Provider:
		a.Session = cc.session
	default:
		a.Note(llm.Diagnostic{
			Code: llm.CapSessionElsewhere, Class: llm.ClassCapability, Stage: llm.StagePlan,
			Severity: llm.SeverityLoss, Provider: target.Provider, Field: "session",
			Detail: "the conversation handle was issued by " + cc.provider + ", so this attempt starts a new one",
		})
	}
	a.Tools = mergeTools(a.Request.Tools, group, rec, target.Provider)
	a.Request.Tools = a.Tools
	return a
}

// groupTools is what the route's tool groups add to a call. It is read once
// for the whole chain, so which links the call goes to first is decided on the
// same tools every link is then offered.
func (c *Client) groupTools(dec route.Decision) []llm.Tool {
	var group []llm.Tool
	for _, name := range dec.Tools {
		group = append(group, c.tools[name]...)
		if src := c.sources[name]; src != nil {
			group = append(group, src()...)
		}
	}
	return group
}

func mergeTools(own, group []llm.Tool, rec llm.Recorder, provider string) []llm.Tool {
	if len(group) == 0 {
		return own
	}
	byCaller := make(map[string]bool, len(own))
	for _, t := range own {
		byCaller[t.Name] = true
	}
	named := make(map[string]bool, len(own))
	maps.Copy(named, byCaller)
	out := append([]llm.Tool(nil), own...)
	for _, t := range group {
		if named[t.Name] {
			if rec != nil {
				detail := "the route registers this tool twice, and the first registration is the one that stands"
				if byCaller[t.Name] {
					detail = "the route offers a tool the caller named too, and the caller's is the one it wrote the call against"
				}
				rec.Note(llm.Diagnostic{
					Code: llm.DropToolDefinition, Class: llm.ClassCapability, Stage: llm.StagePlan,
					Severity: llm.SeverityInfo, Provider: provider, Field: t.Name,
					Detail: detail,
				})
			}
			continue
		}
		named[t.Name] = true
		out = append(out, t)
	}
	return out
}

// acquire waits for a slot at the provider's gate. The wait is part of the
// attempt, so it is held to the attempt's deadline as well as the caller's. It
// says whether the slot had to be waited for: a free one is handed over at
// once, and a busy one frees only when the call holding it ends, which no
// backend does inside a millisecond.
func (c *Client) acquire(ctx context.Context, p llm.Provider, due time.Time) (release func(), waited bool, err error) {
	gated, ok := p.(llm.Gated)
	if !ok {
		return func() {}, false, nil
	}
	ctx, cancel := deadline(ctx, due)
	defer cancel()
	asked := time.Now()
	release, err = gated.Acquire(ctx)
	return release, time.Since(asked) >= time.Millisecond, err
}

func (c *Client) book(ctx context.Context, rec llm.Attempted, cc callConfig, rec2 llm.Recorder) {
	if len(c.sinks) == 0 {
		return
	}
	ctx = context.WithoutCancel(ctx)
	if err := c.sinks.Record(ctx, usage.FromAttempt(rec, c.now(), cc.format, cc.stream)); err != nil && rec2 != nil {
		rec2.Note(llm.Diagnostic{
			Code: llm.AcctRowDropped, Class: llm.ClassAccounting, Stage: llm.StageResponse,
			Severity: llm.SeverityLoss, Provider: rec.Provider, Model: rec.Model,
			Detail: "the ledger refused this row, so the call is on no record but this one: " + err.Error(),
		})
	}
}

// OutcomeFor is the ledger outcome a failure is booked under. A call that
// never reached its own send, such as passthrough forwarding one straight to
// a backend, books what came back the same way a routed one does: a caller
// gone or a wait that ran out is cancelled, not blamed on the backend as an
// upstream error.
func OutcomeFor(e *llm.Error) llm.Outcome { return outcomeFor(e) }

func outcomeFor(e *llm.Error) llm.Outcome {
	switch {
	case e == nil:
		return llm.OutcomeOK
	case e.Kind == llm.KindCancelled, e.Kind == llm.KindTimeout:
		return llm.OutcomeCancelled
	case e.Kind == llm.KindSaturated, e.Kind == llm.KindLoss && !e.Sent:
		return llm.OutcomeRejected
	default:
		return llm.OutcomeError
	}
}

func elapsed(since time.Time, now time.Time) int64 {
	if since.IsZero() {
		return 0
	}
	return now.Sub(since).Milliseconds()
}

// wall is when an attempt given d must end, or zero where it has no limit. It
// is fixed before the attempt waits for a slot, so the wait and the send share
// the one budget rather than each being given the whole of it.
func wall(d time.Duration) time.Time {
	if d <= 0 {
		return time.Time{}
	}
	return time.Now().Add(d)
}

func deadline(ctx context.Context, due time.Time) (context.Context, context.CancelFunc) {
	if due.IsZero() {
		return context.WithCancel(ctx)
	}
	return context.WithDeadline(ctx, due)
}

// untilPrimed walls an attempt until its stream is primed, after which the body
// takes as long as it takes. The wall cancels rather than expires, so it names
// its own cause: a deadline we set is not the caller hanging up, and a client
// still on the line is owed 504 rather than 499.
func untilPrimed(ctx context.Context, due time.Time) (context.Context, context.CancelFunc, func()) {
	ctx, cancel := context.WithCancelCause(ctx)
	stop := func() { cancel(context.Canceled) }
	if due.IsZero() {
		return ctx, stop, func() {}
	}
	timer := time.AfterFunc(time.Until(due), func() { cancel(context.DeadlineExceeded) })
	return ctx, stop, func() { timer.Stop() }
}

// No backend names a model this long, and the name is matched against every
// route pattern before anything else happens, so an unbounded one is work a
// caller can ask for without sending a question at all.
const maxModelName = 512

func validate(req *llm.Request) error {
	if len(req.Messages) == 0 && len(req.System) == 0 {
		return llm.Errorf(llm.KindInvalidRequest, "the request carries no messages")
	}
	if len(req.Model) > maxModelName {
		return llm.Errorf(llm.KindInvalidRequest,
			"the model name is %d characters; no backend names one longer than %d", len(req.Model), maxModelName)
	}
	for i, t := range req.Tools {
		if t.Name == "" {
			return llm.Errorf(llm.KindInvalidRequest, "tool %d has no name", i)
		}
	}
	for _, m := range req.Messages {
		for _, p := range m.Content {
			if r, ok := p.(*llm.ToolResult); ok && r.ID == "" && r.Name == "" {
				return llm.Errorf(llm.KindInvalidRequest, "a tool result names neither a call nor a tool")
			}
		}
	}
	return nil
}
