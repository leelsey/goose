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
	return c.table.Route(call)
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

func (c *Client) attemptFor(dec route.Decision, target route.Target, index int, req *llm.Request, cc callConfig, rec llm.Recorder) *llm.Attempt {
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
	var group []llm.Tool
	for _, name := range dec.Tools {
		group = append(group, c.tools[name]...)
		if src := c.sources[name]; src != nil {
			group = append(group, src()...)
		}
	}
	a.Tools = mergeTools(a.Request.Tools, group, rec, target.Provider)
	a.Request.Tools = a.Tools
	return a
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

func (c *Client) acquire(ctx context.Context, p llm.Provider) (func(), error) {
	gated, ok := p.(llm.Gated)
	if !ok {
		return func() {}, nil
	}
	return gated.Acquire(ctx)
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

func outcomeFor(e *llm.Error) llm.Outcome {
	switch {
	case e == nil:
		return llm.OutcomeOK
	case e.Kind == llm.KindCancelled, e.Kind == llm.KindTimeout:
		return llm.OutcomeCancelled
	case e.Kind == llm.KindSaturated:
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

func deadline(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc) {
	if d <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, d)
}

// untilPrimed walls an attempt until its stream is primed, after which the body
// takes as long as it takes. The wall cancels rather than expires, so it names
// its own cause: a deadline we set is not the caller hanging up, and a client
// still on the line is owed 504 rather than 499.
func untilPrimed(ctx context.Context, d time.Duration) (context.Context, context.CancelFunc, func()) {
	ctx, cancel := context.WithCancelCause(ctx)
	stop := func() { cancel(context.Canceled) }
	if d <= 0 {
		return ctx, stop, func() {}
	}
	timer := time.AfterFunc(d, func() { cancel(context.DeadlineExceeded) })
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
