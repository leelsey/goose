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
	"time"

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

// Without these an attempt has no deadline and an upstream body no ceiling, so
// one stalled backend holds a request, its connection and its concurrency slot
// until the client gives up.
func limitsOf(s Server) llm.Limits {
	timeout := defAttempt
	if d, err := time.ParseDuration(s.AttemptTimeout); err == nil && d > 0 {
		timeout = d
	}
	response := int64(defResponse)
	if s.MaxResponseMB > 0 {
		response = int64(s.MaxResponseMB) << 20
	}
	return llm.Limits{Timeout: timeout, MaxResponseBytes: response, MaxFrameBytes: s.MaxFrameMB << 20}
}

// In a config file zero is how an operator asks for no gate at all, which the
// provider spells as a negative; leaving the key out is what asks for its
// default. The two cannot be told apart once the string has been through atoi.
func agentGate(v string) int {
	if v == "" {
		return 0
	}
	if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n <= 0 {
		return -1
	} else if err == nil {
		return n
	}
	// Unreadable: checkOptions refuses this before it reaches here, so taking
	// the default is belt and braces rather than a silent removal of the gate.
	return 0
}

func build(cfg *Config, logger *slog.Logger, sinks []usage.Sink, tools *mcp.Manager) (*goose.Client, error) {
	opts := []goose.Option{goose.WithLogger(logger)}
	if tools != nil {
		opts = append(opts, goose.WithToolSource("mcp", tools.Proxied))
	} else if len(cfg.MCP) > 0 {
		// A check starts no tool server, but a route naming the group is still
		// checked against it.
		opts = append(opts, goose.WithToolSource("mcp", func() []llm.Tool { return nil }))
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
	opts = append(opts, goose.WithLimits(limitsOf(cfg.Server)))
	if cfg.Server.RedactUpstream {
		opts = append(opts, goose.WithRedactedErrors())
	}

	for _, p := range cfg.Providers {
		built, err := newProvider(p)
		if err != nil {
			id := p.ID
			if id == "" {
				id = p.Kind
			}
			// A library's own error carries the kind an API caller would
			// branch on, which tells an operator nothing about the file. One
			// wrapped in what this file adds keeps the whole of it.
			msg := err.Error()
			if e, ok := err.(*llm.Error); ok {
				msg = e.Message
			}
			return nil, fmt.Errorf("provider %q: %s", id, msg)
		}
		opts = append(opts, goose.WithProvider(built))
	}
	rules, err := rulesOf(cfg)
	if err != nil {
		return nil, err
	}
	opts = append(opts, goose.WithRoutes(rules...))
	return goose.New(opts...)
}

func rulesOf(cfg *Config) ([]route.Rule, error) {
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
	var rules []route.Rule
	for _, r := range routes {
		rule, err := newRule(r, pinned, cfg.Server.Strict)
		if err != nil {
			return nil, err
		}
		rules = append(rules, rule)
	}
	return rules, nil
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
			KeyScheme: llm.SchemeAnthropic, NoClientBetas: p.ClientBeta != nil && !*p.ClientBeta,
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
				return nil, fmt.Errorf("unknown dialect feature %q", feature)
			}
		}
		if p.BaseURL == "" {
			return nil, fmt.Errorf("a compatible backend needs base_url")
		}
		return openai.New(openai.Options{
			ID: id, BaseURL: p.BaseURL, APIKey: p.APIKey, Model: p.Model,
			Dialect: dialect, MaxRetries: p.MaxRetries, Timeout: timeout,
			AllowClientKey: p.ClientKey, Header: headerOf(p.Headers),
			AuthHeader: p.Options["auth_header"], AuthPrefix: p.Options["auth_prefix"],
		})
	case "azure":
		if p.Options["resource"] == "" && p.BaseURL == "" {
			return nil, fmt.Errorf("an azure backend needs options.resource or base_url")
		}
		return azure.New(azure.Options{
			ID: id, Resource: p.Options["resource"], BaseURL: p.BaseURL,
			Deployment: p.Options["deployment"], Model: p.Model, APIKey: p.APIKey,
			APIVersion: p.Options["api_version"], MaxRetries: p.MaxRetries, Timeout: timeout,
			Header: headerOf(p.Headers),
		})
	case "cloudflare":
		if p.Options["account_id"] == "" && p.BaseURL == "" {
			return nil, fmt.Errorf("a cloudflare backend needs options.account_id or base_url")
		}
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
		if region == "" {
			return nil, fmt.Errorf("a bedrock backend needs a region: set options.region or AWS_REGION")
		}
		if !regionName(region) {
			return nil, fmt.Errorf("region %q, from options.region, AWS_REGION or AWS_DEFAULT_REGION, is not a region's name", region)
		}
		return bedrock.New(bedrock.Options{
			ID: id, Region: region, Model: p.Model, BaseURL: p.BaseURL,
			Credentials: awsCredentials(p, os.Getenv),
			MaxRetries:  p.MaxRetries, Timeout: timeout, Header: headerOf(p.Headers),
		})
	case "vertex":
		token, keyProject, err := googleToken(p, os.Getenv, os.ReadFile)
		if err != nil {
			return nil, fmt.Errorf("options.credentials_file or GOOGLE_APPLICATION_CREDENTIALS: %w", err)
		}
		project := firstNonEmpty(p.Options["project"], os.Getenv("GOOGLE_CLOUD_PROJECT"), os.Getenv("GCP_PROJECT"), keyProject)
		if project == "" {
			return nil, fmt.Errorf("a vertex backend needs a project: set options.project or GOOGLE_CLOUD_PROJECT")
		}
		region := firstNonEmpty(p.Options["region"], os.Getenv("GOOGLE_CLOUD_LOCATION"))
		if region != "" && !regionName(region) {
			return nil, fmt.Errorf("region %q, from options.region or GOOGLE_CLOUD_LOCATION, is not a region's name", region)
		}
		return vertex.New(vertex.Options{
			ID: id, Project: project, Region: region,
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
			MaxConcurrent: agentGate(p.Options["max_concurrent"]),
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
		return nil, fmt.Errorf("unknown kind %q", p.Kind)
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

// newRule builds a route's rule. strict is the server's: a route that leaves
// its own strict unsaid keeps it, though it names codes to allow or deny,
// since a rule that names any policy at all is not given the server's.
func newRule(r Route, pinned map[string]string, strict bool) (route.Rule, error) {
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
	if r.Strict != nil {
		strict = *r.Strict
	}
	rule := route.Rule{
		Name: r.Name, Match: match, Model: r.Model, Sticky: r.Sticky, Tools: r.Tools,
		Policy:    llm.LossPolicy{Strict: strict, Allow: r.Allow, Deny: r.Deny},
		PolicySet: r.Strict != nil || len(r.Allow) > 0 || len(r.Deny) > 0,
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
