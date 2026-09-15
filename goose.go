// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT
// by Leelsey (le@elsey.me.uk) - https://leelsey.me.uk

// Package goose routes an LLM call to the upstream that serves it.
package goose

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

type Client struct {
	providers     map[string]llm.Provider
	order         []string
	table         *route.Table
	sinks         usage.Multi
	tools         map[string][]llm.Tool
	sources       map[string]func() []llm.Tool
	env           llm.Env
	policy        llm.LossPolicy
	limits        llm.Limits
	breakerPolicy *Breaker
	abandon       time.Duration
	breaker       *breakers
}

const DefaultAbandonGrace = 2 * time.Second

func (c *Client) abandonGrace() time.Duration {
	if c.abandon > 0 {
		return c.abandon
	}
	return DefaultAbandonGrace
}

type Option func(*Client) error

func WithProvider(p llm.Provider) Option {
	return func(c *Client) error {
		if p == nil {
			return llm.Errorf(llm.KindInvalidRequest, "nil provider")
		}
		id := p.ID()
		if id == "" {
			return llm.Errorf(llm.KindInvalidRequest, "provider has no id")
		}
		if _, exists := c.providers[id]; exists {
			return llm.Errorf(llm.KindInvalidRequest, "provider %q is registered twice", id)
		}
		c.providers[id] = p
		c.order = append(c.order, id)
		return nil
	}
}

func WithRoutes(rules ...route.Rule) Option {
	return func(c *Client) error {
		c.table.Rules = append(c.table.Rules, rules...)
		return nil
	}
}

func WithSelector(fn func(*route.Call) []route.Target) Option {
	return func(c *Client) error {
		c.table.Selector = fn
		return nil
	}
}

func WithDefaultTools(groups ...string) Option {
	return func(c *Client) error {
		c.table.Defaults.Tools = append(c.table.Defaults.Tools, groups...)
		return nil
	}
}

func WithUsage(sinks ...usage.Sink) Option {
	return func(c *Client) error {
		c.sinks = append(c.sinks, sinks...)
		return nil
	}
}

func WithTools(group string, tools []llm.Tool) Option {
	return func(c *Client) error {
		c.tools[group] = tools
		return nil
	}
}

func WithToolSource(group string, fn func() []llm.Tool) Option {
	return func(c *Client) error {
		if fn == nil {
			return llm.Errorf(llm.KindInvalidRequest, "tool source %q is nil", group)
		}
		c.sources[group] = fn
		return nil
	}
}

func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) error { c.env.HTTP = h; return nil }
}

func WithLogger(l *slog.Logger) Option {
	return func(c *Client) error { c.env.Logger = l; return nil }
}

func WithClock(now func() time.Time) Option {
	return func(c *Client) error { c.env.Now = now; return nil }
}

func WithLossPolicy(p llm.LossPolicy) Option {
	return func(c *Client) error {
		c.policy = p
		c.table.Defaults.Policy = p
		return nil
	}
}

func WithBreaker(b Breaker) Option {
	return func(c *Client) error {
		if b.Failures < 0 {
			return llm.Errorf(llm.KindInvalidRequest, "breaker failures is %d; a count cannot be negative", b.Failures)
		}
		if b.Failures == 0 {
			b.Failures = DefaultBreakerFailures
		}
		c.breakerPolicy = &b
		return nil
	}
}

func WithLimits(l llm.Limits) Option {
	return func(c *Client) error {
		c.limits = l
		c.table.Defaults.Limits = l
		return nil
	}
}

func New(opts ...Option) (*Client, error) {
	c := &Client{
		providers: map[string]llm.Provider{},
		table:     &route.Table{},
		tools:     map[string][]llm.Tool{},
		sources:   map[string]func() []llm.Tool{},
	}
	for _, o := range opts {
		if o == nil {
			continue
		}
		if err := o(c); err != nil {
			return nil, err
		}
	}
	if len(c.providers) == 0 {
		return nil, llm.Errorf(llm.KindInvalidRequest, "no providers registered")
	}
	if len(c.table.Rules) == 0 && c.table.Selector == nil {
		if len(c.providers) > 1 {
			return nil, llm.Errorf(llm.KindInvalidRequest,
				"%d providers are registered but no routes; add WithRoutes or WithSelector", len(c.providers))
		}
		c.table.Rules = []route.Rule{{
			Name:  "default",
			Match: route.Always(),
			Chain: []route.Target{{Provider: c.order[0]}},
		}}
	}
	for i, r := range c.table.Rules {
		for _, t := range r.Chain {
			if _, ok := c.providers[t.Provider]; !ok {
				return nil, llm.Errorf(llm.KindInvalidRequest, "route %d names unregistered provider %q", i, t.Provider)
			}
		}
	}
	if c.breakerPolicy != nil {
		c.breaker = newBreakers(*c.breakerPolicy, c.env.Now)
	}
	for _, id := range c.order {
		if b, ok := c.providers[id].(llm.Binder); ok {
			if err := b.Bind(c.env); err != nil {
				return nil, err
			}
		}
	}
	return c, nil
}

func (c *Client) Provider(id string) (llm.Provider, bool) {
	p, ok := c.providers[id]
	return p, ok
}

func (c *Client) Providers() []string { return append([]string(nil), c.order...) }

func (c *Client) Env() llm.Env { return c.env }

func (c *Client) Record(ctx context.Context, rec llm.Attempted, format string) {
	c.book(ctx, rec, callConfig{format: format}, nil)
}

func listFailure(err error) string {
	if e := llm.AsError(err); e != nil {
		return e.Message
	}
	return err.Error()
}

func (c *Client) Models(ctx context.Context) ([]llm.Model, llm.Diagnostics) {
	var out []llm.Model
	var diags llm.Diagnostics
	seen := map[string]bool{}
	for _, id := range c.order {
		lister, ok := c.providers[id].(llm.Lister)
		if !ok {
			diags = append(diags, llm.Diagnostic{
				Code: llm.CapNoModelList, Class: llm.ClassCapability, Stage: llm.StageResponse,
				Severity: llm.SeverityInfo, Provider: id,
				Detail: "this backend has no model list, so nothing of what it serves is in this one",
			})
			continue
		}
		models, err := lister.Models(ctx)
		if err != nil {
			diags = append(diags, llm.Diagnostic{
				Code: llm.CapNoModelList, Class: llm.ClassCapability, Stage: llm.StageResponse,
				Severity: llm.SeverityLoss, Provider: id,
				Detail: "the model list could not be read: " + listFailure(err),
			})
			continue
		}
		for _, m := range models {
			if seen[m.ID] {
				continue
			}
			seen[m.ID] = true
			out = append(out, m)
		}
	}
	return out, diags
}

type Health struct {
	Provider string `json:"provider"`
	Ready    bool   `json:"ready"`
	Known    bool   `json:"known"`
	Err      string `json:"error,omitempty"`
}

func (c *Client) Ready(ctx context.Context) ([]Health, bool) {
	out := make([]Health, 0, len(c.order))
	ok := true
	for _, id := range c.order {
		h := Health{Provider: id, Ready: true}
		prober, can := c.providers[id].(llm.Prober)
		if !can {
			h.Known = false
			out = append(out, h)
			continue
		}
		h.Known = true
		if err := prober.Probe(ctx); err != nil {
			h.Ready, h.Err, ok = false, llm.AsError(err).Message, false
		}
		out = append(out, h)
	}
	return out, ok
}

func (c *Client) Close(ctx context.Context) error {
	var first error
	for _, id := range c.order {
		closer, ok := c.providers[id].(llm.Closer)
		if !ok {
			continue
		}
		if err := closer.Close(ctx); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func (c *Client) now() time.Time {
	if c.env.Now != nil {
		return c.env.Now()
	}
	return time.Now()
}
