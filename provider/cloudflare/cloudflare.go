// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package cloudflare talks to Workers AI through its OpenAI endpoint.
package cloudflare

import (
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
)

type Options struct {
	ID         string
	AccountID  string
	APIKey     string
	Model      string
	BaseURL    string
	MaxRetries int
	Timeout    time.Duration
	Header     http.Header
}

func New(o Options) (*openai.Provider, error) {
	if o.ID == "" {
		o.ID = "cloudflare"
	}
	base := strings.TrimRight(o.BaseURL, "/")
	if base == "" {
		if o.AccountID == "" {
			return nil, llm.Errorf(llm.KindInvalidRequest, "cloudflare: account_id or base_url is required")
		}
		base = "https://api.cloudflare.com/client/v4/accounts/" + url.PathEscape(o.AccountID) + "/ai/v1"
	}
	return openai.New(openai.Options{
		ID: o.ID, BaseURL: base, APIKey: o.APIKey, Model: o.Model,
		MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: o.Header,
		Dialect: openai.Dialect{StreamUsage: true},
	})
}
