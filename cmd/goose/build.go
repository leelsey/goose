// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/mcp"
	"github.com/leelsey/goose/provider/agentcli"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/azure"
	"github.com/leelsey/goose/provider/bedrock"
	"github.com/leelsey/goose/provider/cloudflare"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/ollama"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/provider/openrouter"
	"github.com/leelsey/goose/provider/vertex"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

func build(cfg *Config, logger *slog.Logger, sinks []usage.Sink, tools *mcp.Manager) (*goose.Client, error) {
	opts := []goose.Option{goose.WithLogger(logger)}
	if tools != nil {
		opts = append(opts, goose.WithToolSource("mcp", tools.Proxied))
	}
	if len(sinks) > 0 {
		opts = append(opts, goose.WithUsage(sinks...))
	}
	if cfg.Server.BreakerFailures > 0 {
		opts = append(opts, goose.WithBreaker(goose.Breaker{
			Failures: cfg.Server.BreakerFailures,
			Cooldown: parseDuration(cfg.Server.BreakerCooldown),
		}))
	}
	if cfg.Server.Strict {
		opts = append(opts, goose.WithLossPolicy(llm.LossPolicy{Strict: true}))
	}

	for _, p := range cfg.Providers {
		built, err := newProvider(p)
		if err != nil {
			return nil, err
		}
		opts = append(opts, goose.WithProvider(built))
	}
	pinned := map[string]string{}
	for _, p := range cfg.Providers {
		id := p.ID
		if id == "" {
			id = p.Kind
		}
		if p.Model != "" {
			pinned[id] = p.Model
		}
	}
	routes := cfg.Routes
	if len(routes) == 0 && len(cfg.Providers) == 1 {
		id := cfg.Providers[0].ID
		if id == "" {
			id = cfg.Providers[0].Kind
		}
		routes = []Route{{Name: "default", Chain: []string{id}}}
	}
	for _, r := range routes {
		rule, err := newRule(r, pinned)
		if err != nil {
			return nil, err
		}
		opts = append(opts, goose.WithRoutes(rule))
	}
	return goose.New(opts...)
}

func headerOf(m map[string]string) http.Header {
	if len(m) == 0 {
		return nil
	}
	h := http.Header{}
	for k, v := range m {
		h.Set(k, v)
	}
	return h
}

func newProvider(p Provider) (llm.Provider, error) {
	id := p.ID
	if id == "" {
		id = p.Kind
	}
	timeout := parseTimeout(p.Timeout)
	switch p.Kind {
	case "anthropic":
		return anthropic.New(anthropic.Options{
			ID: id, BaseURL: p.BaseURL, APIKey: p.APIKey, Model: p.Model,
			Version: p.Version, Betas: p.Betas, MaxRetries: p.MaxRetries,
			Timeout: timeout, AllowClientKey: p.ClientKey, Header: headerOf(p.Headers),
			KeyScheme: llm.SchemeAnthropic,
		})
	case "openai":
		return openai.New(openai.Options{
			ID: id, BaseURL: p.BaseURL, APIKey: p.APIKey, Model: p.Model,
			Dialect: openai.OpenAIDialect(), MaxRetries: p.MaxRetries,
			Timeout: timeout, AllowClientKey: p.ClientKey, Header: headerOf(p.Headers),
			KeyScheme:    llm.SchemeOpenAI,
			Organisation: p.Options["organisation"], Project: p.Options["project"],
		})
	case "compatible":
		dialect := openai.CompatibleDialect()
		for _, feature := range strings.Split(p.Dialect, ",") {
			switch strings.TrimSpace(feature) {
			case "stream_usage":
				dialect.StreamUsage = true
			case "extensions":
				dialect.Extensions = true
			case "per_model_ceiling":
				dialect.PerModelCeiling = true
			case "responses":
				dialect.Responses = true
			case "":
			default:
				return nil, fmt.Errorf("provider %q: unknown dialect feature %q", id, feature)
			}
		}
		if p.BaseURL == "" {
			return nil, fmt.Errorf("provider %q: a compatible backend needs base_url", id)
		}
		return openai.New(openai.Options{
			ID: id, BaseURL: p.BaseURL, APIKey: p.APIKey, Model: p.Model,
			Dialect: dialect, MaxRetries: p.MaxRetries, Timeout: timeout,
			AllowClientKey: p.ClientKey, Header: headerOf(p.Headers),
			AuthHeader: p.Options["auth_header"], AuthPrefix: p.Options["auth_prefix"],
		})
	case "azure":
		return azure.New(azure.Options{
			ID: id, Resource: p.Options["resource"], BaseURL: p.BaseURL,
			Deployment: p.Options["deployment"], Model: p.Model, APIKey: p.APIKey,
			APIVersion: p.Options["api_version"], MaxRetries: p.MaxRetries, Timeout: timeout,
			Header: headerOf(p.Headers),
		})
	case "cloudflare":
		return cloudflare.New(cloudflare.Options{
			ID: id, AccountID: p.Options["account_id"], BaseURL: p.BaseURL,
			APIKey: p.APIKey, Model: p.Model, MaxRetries: p.MaxRetries, Timeout: timeout,
			Header: headerOf(p.Headers),
		})
	case "openrouter":
		return openrouter.New(openrouter.Options{
			ID: id, BaseURL: p.BaseURL, APIKey: p.APIKey, Model: p.Model,
			Referer: p.Options["referer"], Title: p.Options["title"],
			Providers:      splitList(p.Options["providers"]),
			AllowFallbacks: p.Options["allow_fallbacks"] != "false",
			MaxRetries:     p.MaxRetries, Timeout: timeout, AllowClientKey: p.ClientKey,
			Header: headerOf(p.Headers),
		})
	case "bedrock":
		region := firstNonEmpty(p.Options["region"], os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"))
		return bedrock.New(bedrock.Options{
			ID: id, Region: region, Model: p.Model, BaseURL: p.BaseURL,
			Credentials: awsCredentials(p, os.Getenv),
			MaxRetries:  p.MaxRetries, Timeout: timeout, Header: headerOf(p.Headers),
		})
	case "vertex":
		token, keyProject, err := googleToken(p, os.Getenv, os.ReadFile)
		if err != nil {
			return nil, err
		}
		return vertex.New(vertex.Options{
			ID: id, Project: firstNonEmpty(p.Options["project"], os.Getenv("GOOGLE_CLOUD_PROJECT"), os.Getenv("GCP_PROJECT"), keyProject),
			Region:    firstNonEmpty(p.Options["region"], os.Getenv("GOOGLE_CLOUD_LOCATION")),
			Model:     p.Model,
			Publisher: p.Options["publisher"],
			Token:     token, BaseURL: p.BaseURL,
			MaxRetries: p.MaxRetries, Timeout: timeout, Header: headerOf(p.Headers),
		})
	case "claude", "codex", "antigravity", "copilot":
		allow := append(agentcli.DefaultEnvAllowlist(), splitList(p.Options["env_allow"])...)
		extra := map[string]string{}
		if p.APIKey != "" {
			if name := p.Options["api_key_env_name"]; name != "" {
				extra[name] = p.APIKey
			}
		}
		return agentcli.New(agentcli.Options{
			ID: id, Kind: agentcli.Kind(p.Kind), Binary: p.BaseURL, Model: p.Model,
			Dir: p.Options["dir"], Env: spawn.ScrubEnv(os.Environ(), allow, extra),
			SystemMode:    agentcli.SystemMode(p.Options["system_mode"]),
			Tools:         p.Options["tools"],
			MaxTurns:      atoi(p.Options["max_turns"]),
			Sandbox:       p.Options["sandbox"],
			MaxConcurrent: atoi(p.Options["max_concurrent"]),
			Timeout:       timeout,
			TempDir:       p.Options["temp_dir"],
		})
	case "google":
		return google.New(google.Options{
			ID: id, BaseURL: p.BaseURL, Prefix: p.Prefix, APIKey: p.APIKey, Model: p.Model,
			MaxRetries: p.MaxRetries, Timeout: timeout, AllowClientKey: p.ClientKey,
			Header: headerOf(p.Headers), KeyScheme: llm.SchemeGoogle,
		})
	case "ollama":
		return ollama.New(ollama.Options{
			ID: id, BaseURL: p.BaseURL, APIKey: p.APIKey, Model: p.Model,
			KeepAlive: p.Options["keep_alive"], MaxRetries: p.MaxRetries, Timeout: timeout,
			Header: headerOf(p.Headers),
		})
	default:
		return nil, fmt.Errorf("provider %q: unknown kind %q", id, p.Kind)
	}
}

func atoi(s string) int {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0
	}
	return n
}

func splitList(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if v := strings.TrimSpace(p); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func newRule(r Route, pinned map[string]string) (route.Rule, error) {
	var matchers []route.Matcher
	if len(r.Models) > 0 {
		var globs []route.Matcher
		for _, m := range r.Models {
			globs = append(globs, route.Model(m))
		}
		matchers = append(matchers, route.Any(globs...))
	}
	if len(r.Formats) > 0 {
		matchers = append(matchers, route.Format(r.Formats...))
	}
	for _, h := range r.Header {
		key, value, _ := strings.Cut(h, "=")
		matchers = append(matchers, route.Header(strings.TrimSpace(key), strings.TrimSpace(value)))
	}
	match := route.Always()
	if len(matchers) > 0 {
		match = route.All(matchers...)
	}
	rule := route.Rule{
		Name: r.Name, Match: match, Model: r.Model, Sticky: r.Sticky, Tools: r.Tools,
		Policy: llm.LossPolicy{Strict: r.Strict, Allow: r.Allow, Deny: r.Deny},
	}
	for _, id := range r.Chain {
		provider, model, _ := strings.Cut(id, ":")
		provider, model = strings.TrimSpace(provider), strings.TrimSpace(model)
		if model == "" && r.Model == "" {
			model = pinned[provider]
		}
		rule.Chain = append(rule.Chain, route.Target{Provider: provider, Model: model})
	}
	return rule, nil
}
