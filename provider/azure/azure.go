// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package azure talks to Azure AI Foundry, which serves the OpenAI shapes.
package azure

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
)

type Options struct {
	ID         string
	Resource   string
	BaseURL    string
	Deployment string
	Model      string
	APIKey     string
	APIVersion string
	Token      func(ctx context.Context) (string, error)
	MaxRetries int
	Timeout    time.Duration
	Header     http.Header
}

func New(o Options) (*openai.Provider, error) {
	if o.ID == "" {
		o.ID = "azure"
	}
	base := strings.TrimRight(o.BaseURL, "/")
	if base == "" {
		if o.Resource == "" {
			return nil, llm.Errorf(llm.KindInvalidRequest, "azure: resource or base_url is required")
		}
		base = "https://" + o.Resource + ".openai.azure.com"
	}

	opts := openai.Options{
		ID: o.ID, APIKey: o.APIKey, Model: firstNonEmpty(o.Model, o.Deployment),
		MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: o.Header,
		AuthHeader: "api-key",
		Dialect: openai.Dialect{
			StreamUsage: true, Extensions: false, PerModelCeiling: true, Responses: true,
		},
	}

	if o.APIVersion == "" || o.APIVersion == "v1" {
		opts.BaseURL = base + "/openai/v1"
	} else {
		if o.Deployment == "" {
			return nil, llm.Errorf(llm.KindInvalidRequest, "azure: a dated api_version needs a deployment")
		}
		opts.BaseURL = base + "/openai"
		opts.ChatPath = "/deployments/" + url.PathEscape(o.Deployment) + "/chat/completions"
		opts.ResponsesPath = "/deployments/" + url.PathEscape(o.Deployment) + "/responses"
		opts.Query = "api-version=" + url.QueryEscape(o.APIVersion)
	}

	if o.Token != nil {
		opts.Sign = func(ctx context.Context, r *http.Request, _ []byte) error {
			token, err := o.Token(ctx)
			if err != nil {
				return err
			}
			r.Header.Del("api-key")
			r.Header.Set("Authorization", "Bearer "+token)
			return nil
		}
	}
	return openai.New(opts)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
