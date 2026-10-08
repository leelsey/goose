// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package openrouter talks to OpenRouter through the OpenAI shape.
package openrouter

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

// Provider serves the OpenAI shape from OpenRouter, whose catalogue spans
// every vendor behind it and so refuses a model id it does not have rather
// than one that is merely malformed.
type Provider struct {
	*openai.Provider
}

type Options struct {
	ID        string
	BaseURL   string
	APIKey    string
	Model     string
	Referer   string
	Title     string
	Providers []string
	// AllowFallbacks left false asks the gateway not to reach for a second
	// provider, which is the opposite of its own default; the CLI sets it.
	AllowFallbacks bool
	MaxRetries     int
	Timeout        time.Duration
	Header         http.Header
	AllowClientKey bool
}

func New(o Options) (*Provider, error) {
	if o.ID == "" {
		o.ID = "openrouter"
	}
	if o.BaseURL == "" {
		o.BaseURL = defaultBaseURL
	}
	h := http.Header{}
	for k, vs := range o.Header {
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	if o.Referer != "" {
		h.Set("HTTP-Referer", o.Referer)
	}
	if o.Title != "" {
		h.Set("X-Title", o.Title)
	}
	body := map[string]any{}
	if pref := o.Preferences(); pref != nil {
		body["provider"] = pref
	}
	if len(body) == 0 {
		body = nil
	}
	inner, err := openai.New(openai.Options{
		ID: o.ID, BaseURL: o.BaseURL, APIKey: o.APIKey, Model: o.Model,
		MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: h,
		AllowClientKey: o.AllowClientKey, Body: body,
		Dialect: openai.Dialect{StreamUsage: true, Extensions: true, PerModelCeiling: true},
	})
	if err != nil {
		return nil, err
	}
	return &Provider{Provider: inner}, nil
}

func (o Options) Preferences() map[string]any {
	if len(o.Providers) == 0 && o.AllowFallbacks {
		return nil
	}
	pref := map[string]any{"allow_fallbacks": o.AllowFallbacks}
	if len(o.Providers) > 0 {
		pref["order"] = o.Providers
	}
	return pref
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	resp, err := p.Provider.Complete(ctx, a)
	if err != nil {
		refine(llm.AsError(err))
	}
	return resp, err
}

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	s, err := p.Provider.Stream(ctx, a)
	if err != nil {
		refine(llm.AsError(err))
	}
	return s, err
}

// refine reads what the status does not say. OpenRouter refuses a model id it
// does not have with the 400 a malformed request gets, and only a malformed
// request fails the same on every backend.
func refine(e *llm.Error) {
	if e.Kind == llm.KindInvalidRequest && strings.Contains(strings.ToLower(e.Message), "is not a valid model id") {
		e.Kind = llm.KindNotFound
	}
}
