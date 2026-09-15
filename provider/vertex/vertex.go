// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package vertex talks to Google Vertex AI.
package vertex

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/openai"
)

type Options struct {
	ID         string
	Project    string
	Region     string
	Model      string
	Publisher  string
	Token      TokenSource
	BaseURL    string
	MaxRetries int
	Timeout    time.Duration
	Header     http.Header
}

const defaultRegion = "us-central1"

func Publisher(model string) string {
	m := strings.ToLower(lastSegment(model))
	switch {
	case strings.HasPrefix(m, "claude"):
		return "anthropic"
	case strings.HasPrefix(m, "gemini"), strings.HasPrefix(m, "text-"), strings.HasPrefix(m, "imagen"):
		return "google"
	case strings.HasPrefix(m, "llama"):
		return "meta"
	case strings.HasPrefix(m, "mistral"), strings.HasPrefix(m, "codestral"), strings.HasPrefix(m, "ministral"):
		return "mistral"
	case strings.HasPrefix(m, "jamba"):
		return "ai21"
	default:
		return "google"
	}
}

func lastSegment(model string) string {
	if i := strings.LastIndexAny(model, "/"); i >= 0 {
		return model[i+1:]
	}
	return model
}

func New(o Options) (llm.Provider, error) {
	if o.Project == "" {
		return nil, llm.Errorf(llm.KindInvalidRequest, "vertex: project is required")
	}
	if o.Token == nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "vertex: a token source is required")
	}
	if o.Region == "" {
		o.Region = defaultRegion
	}
	if o.BaseURL == "" {
		o.BaseURL = "https://" + o.Region + "-aiplatform.googleapis.com"
	}
	publisher := o.Publisher
	if publisher == "" {
		publisher = Publisher(o.Model)
	}
	id := o.ID
	if id == "" {
		id = "vertex-" + publisher
	}

	sign := func(ctx context.Context, r *http.Request, _ []byte) error {
		token, err := o.Token.Token(ctx)
		if err != nil {
			return err
		}
		r.Header.Set("Authorization", "Bearer "+token)
		return nil
	}
	prefix := "/v1/projects/" + url.PathEscape(o.Project) +
		"/locations/" + url.PathEscape(o.Region) +
		"/publishers/" + url.PathEscape(publisher)

	switch publisher {
	case "google":
		return google.New(google.Options{
			ID: id, BaseURL: o.BaseURL, Prefix: prefix, Model: o.Model,
			MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: o.Header, Sign: sign,
		})

	case "anthropic":
		return anthropic.New(anthropic.Options{
			ID: id, BaseURL: o.BaseURL, Model: o.Model,
			MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: o.Header, Sign: sign,
			NoConnectors: true,
			Path: func(a *llm.Attempt, stream bool) string {
				action := "rawPredict"
				if stream {
					action = "streamRawPredict"
				}
				return prefix + "/models/" + url.PathEscape(modelOf(a, o.Model)) + ":" + action
			},
			RewriteBody: func(body map[string]any, _ bool) {
				delete(body, "model")
				body["anthropic_version"] = "vertex-2023-10-16"
			},
		})

	default:
		return openai.New(openai.Options{
			ID: id, BaseURL: o.BaseURL + "/v1/projects/" + url.PathEscape(o.Project) +
				"/locations/" + url.PathEscape(o.Region) + "/endpoints/openapi",
			Model: o.Model, MaxRetries: o.MaxRetries, Timeout: o.Timeout, Header: o.Header,
			Sign:    sign,
			Dialect: openai.Dialect{StreamUsage: true},
		})
	}
}

func modelOf(a *llm.Attempt, fallback string) string {
	switch {
	case a == nil:
		return fallback
	case a.Model != "":
		return a.Model
	case fallback != "":
		return fallback
	case a.Request != nil:
		return a.Request.Model
	default:
		return ""
	}
}
