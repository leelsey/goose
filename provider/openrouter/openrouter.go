// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package openrouter talks to OpenRouter through the OpenAI shape.
package openrouter

import (
	"net/http"
	"time"

	"github.com/leelsey/goose/provider/openai"
)

const defaultBaseURL = "https://openrouter.ai/api/v1"

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

func New(o Options) (*openai.Provider, error) {
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
	return openai.New(openai.Options{
		ID: o.ID, BaseURL: o.BaseURL, APIKey: o.APIKey, Model: o.Model,
		MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: h,
		AllowClientKey: o.AllowClientKey, Body: body,
		Dialect: openai.Dialect{StreamUsage: true, Extensions: true, PerModelCeiling: true},
	})
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
