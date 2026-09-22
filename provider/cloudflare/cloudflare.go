// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package cloudflare talks to Workers AI through its OpenAI endpoint.
package cloudflare

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
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

// Provider serves the OpenAI shapes from Workers AI, whose catalogue sits
// outside them on Cloudflare's own account endpoint.
type Provider struct {
	*openai.Provider
	search string
	key    string
	header http.Header
	call   *httpcore.Caller
}

func New(o Options) (*Provider, error) {
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
	inner, err := openai.New(openai.Options{
		ID: o.ID, BaseURL: base, APIKey: o.APIKey, Model: o.Model,
		MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: o.Header,
		Dialect: openai.Dialect{StreamUsage: true},
	})
	if err != nil {
		return nil, err
	}
	p := &Provider{Provider: inner, key: o.APIKey, header: o.Header}
	// Only a base that still names the OpenAI surface tells us where the
	// account sits; anything else is a gateway we cannot reason about.
	if account, ok := strings.CutSuffix(base, "/ai/v1"); ok {
		p.search = account + "/ai/models/search"
	}
	p.call = &httpcore.Caller{Provider: o.ID, MaxRetries: o.MaxRetries, HTTP: httpcore.NewClient(o.Timeout)}
	return p, nil
}

func (p *Provider) Bind(env llm.Env) error {
	if p.Provider == nil || p.call == nil {
		return llm.Errorf(llm.KindInternal, "cloudflare: this provider was not built by New")
	}
	if env.HTTP != nil {
		p.call.HTTP = env.HTTP
	}
	return p.Provider.Bind(env)
}

func (p *Provider) catalogue() http.Header {
	h := http.Header{}
	for k, vs := range p.header {
		for _, v := range vs {
			h.Add(k, v)
		}
	}
	if p.key != "" {
		h.Set("Authorization", "Bearer "+p.key)
	}
	return h
}

// This host answers 200 and puts the refusal in the body, so the envelope is
// read rather than the status.
type envelope struct {
	Success bool `json:"success"`
	Errors  []struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"errors"`
	Result []struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	} `json:"result"`
	Info struct {
		Page      int `json:"page"`
		PerPage   int `json:"per_page"`
		Count     int `json:"count"`
		TotalPage int `json:"total_pages"`
		Total     int `json:"total_count"`
	} `json:"result_info"`
}

func (e *envelope) err(provider string) error {
	if e.Success {
		return nil
	}
	message := "the account refused the request"
	if len(e.Errors) > 0 && e.Errors[0].Message != "" {
		message = e.Errors[0].Message
	}
	return &llm.Error{
		Kind: llm.KindUpstream, Provider: provider, Sent: true, Status: http.StatusOK,
		Message: message,
	}
}

func (p *Provider) page(ctx context.Context, n int) (*envelope, error) {
	resp, err := p.call.Do(ctx, httpcore.Request{
		Method: http.MethodGet,
		URL:    p.search + "?per_page=50&page=" + strconv.Itoa(n),
		Header: p.catalogue(),
	})
	if err != nil {
		return nil, err
	}
	var body envelope
	err = json.NewDecoder(resp.Body).Decode(&body)
	resp.Body.Close()
	if err != nil {
		return nil, httpcore.MalformedList(p.ID(), err)
	}
	if err := body.err(p.ID()); err != nil {
		return nil, err
	}
	return &body, nil
}

func (p *Provider) Models(ctx context.Context) ([]llm.Model, error) {
	if p.search == "" {
		return p.Provider.Models(ctx)
	}
	var out []llm.Model
	for n := 1; n <= maxPages; n++ {
		body, err := p.page(ctx, n)
		if err != nil {
			return nil, err
		}
		for _, m := range body.Result {
			if m.Name == "" {
				continue
			}
			out = append(out, llm.Model{ID: m.Name, Display: m.Description, Owner: p.ID()})
		}
		// The page size is the host's to honour, so where the list ends is what
		// it says rather than what was asked for.
		if len(body.Result) == 0 || !body.more(len(out)) {
			break
		}
	}
	return out, nil
}

// Twenty pages of fifty is past anything Workers AI serves, and a host that
// keeps answering must not keep us asking.
const maxPages = 20

func (e *envelope) more(read int) bool {
	switch {
	case e.Info.TotalPage > 0:
		return e.Info.Page < e.Info.TotalPage
	case e.Info.Total > 0:
		return read < e.Info.Total
	default:
		return len(e.Result) >= 50
	}
}

func (p *Provider) Probe(ctx context.Context) error {
	if p.search == "" {
		return p.Provider.Probe(ctx)
	}
	_, err := p.page(ctx, 1)
	return err
}
