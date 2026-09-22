// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/BurntSushi/toml"

	"github.com/leelsey/goose/llm"
)

type Config struct {
	Server    Server     `toml:"server"`
	Formats   []string   `toml:"formats"`
	Providers []Provider `toml:"provider"`
	Routes    []Route    `toml:"route"`
	MCP       []MCP      `toml:"mcp"`
	Usage     Usage      `toml:"usage"`
}

type MCP struct {
	Name      string            `toml:"name"`
	Transport string            `toml:"transport"`
	Command   string            `toml:"command"`
	Args      []string          `toml:"args"`
	Dir       string            `toml:"dir"`
	EnvAllow  []string          `toml:"env_allow"`
	URL       string            `toml:"url"`
	Headers   map[string]string `toml:"headers"`
	Timeout   string            `toml:"timeout"`
}

type Server struct {
	Host            string   `toml:"host"`
	Port            int      `toml:"port"`
	APIKey          string   `toml:"api_key"`
	APIKeyEnv       string   `toml:"api_key_env"`
	CORS            bool     `toml:"cors"`
	CORSOrigin      string   `toml:"cors_origin"`
	MaxBodyMB       int      `toml:"max_body_mb"`
	TLSCert         string   `toml:"tls_cert"`
	TLSKey          string   `toml:"tls_key"`
	MCPTools        bool     `toml:"mcp_tools"`
	MaxConcurrent   int      `toml:"max_concurrent"`
	MaxPerMinute    int      `toml:"max_per_minute"`
	AttemptTimeout  string   `toml:"attempt_timeout"`
	MaxResponseMB   int      `toml:"max_response_mb"`
	ForwardedHeader string   `toml:"forwarded_header"`
	BreakerFailures int      `toml:"breaker_failures"`
	BreakerCooldown string   `toml:"breaker_cooldown"`
	Metrics         bool     `toml:"metrics"`
	RedactUpstream  bool     `toml:"redact_upstream_errors"`
	Formats         []string `toml:"formats"`
	Passthrough     string   `toml:"passthrough"`
	Strict          bool     `toml:"strict"`
}

type Provider struct {
	ID         string            `toml:"id"`
	Kind       string            `toml:"kind"`
	BaseURL    string            `toml:"base_url"`
	Prefix     string            `toml:"prefix"`
	Model      string            `toml:"model"`
	APIKey     string            `toml:"api_key"`
	APIKeyEnv  string            `toml:"api_key_env"`
	Version    string            `toml:"version"`
	Betas      []string          `toml:"betas"`
	Timeout    string            `toml:"timeout"`
	MaxRetries int               `toml:"max_retries"`
	Dialect    string            `toml:"dialect"`
	ClientKey  bool              `toml:"allow_client_key"`
	Headers    map[string]string `toml:"headers"`
	Options    map[string]string `toml:"options"`
}

type Route struct {
	Name    string   `toml:"name"`
	Models  []string `toml:"models"`
	Formats []string `toml:"formats"`
	Header  []string `toml:"header"`
	Model   string   `toml:"model"`
	Chain   []string `toml:"chain"`
	Tools   []string `toml:"tools"`
	Sticky  bool     `toml:"sticky"`
	Strict  *bool    `toml:"strict"`
	Allow   []string `toml:"allow"`
	Deny    []string `toml:"deny"`
}

type Usage struct {
	Log     string `toml:"log"`
	Summary bool   `toml:"summary"`
}

const (
	dotFile  = ".goose.toml"
	dirFile  = "goose.toml"
	envKey   = "GOOSE_PROXY_KEY"
	defPort  = 11234
	defHost  = "127.0.0.1"
	defTimeo = 5 * time.Minute

	// An attempt that has not answered in this long is not going to, and an
	// upstream that keeps sending past this is not answering a question anyone
	// asked. Both are generous: a streamed answer stops this clock once it has
	// been primed, so only a whole non-streamed answer has to fit inside it,
	// and that includes a reasoning model working through a large ceiling.
	// Raise it with server.attempt_timeout where that is not enough.
	defAttempt  = 15 * time.Minute
	defResponse = 64 << 20
)

func defaults() *Config {
	return &Config{
		Server:  Server{Host: defHost, Port: defPort},
		Formats: []string{"openai", "openai-responses", "anthropic", "google", "ollama"},
	}
}

func load(explicit string) (*Config, string, error) {
	cfg := defaults()
	for _, path := range candidates(explicit) {
		if path == "" {
			continue
		}
		if _, err := os.Stat(path); err != nil {
			if explicit != "" && path == explicit {
				return nil, "", fmt.Errorf("config %s: %w", path, err)
			}
			continue
		}
		md, err := toml.DecodeFile(path, cfg)
		if err != nil {
			return nil, "", fmt.Errorf("config %s: %w", path, err)
		}
		if left := md.Undecoded(); len(left) > 0 {
			names := make([]string, 0, len(left))
			for _, k := range left {
				names = append(names, k.String())
			}
			return nil, "", fmt.Errorf("config %s: unknown key(s): %s", path, strings.Join(names, ", "))
		}
		if len(cfg.Server.Formats) > 0 {
			cfg.Formats = cfg.Server.Formats
		}
		return cfg, path, nil
	}
	return cfg, "(defaults)", nil
}

func candidates(explicit string) []string {
	out := []string{explicit, dotFile, dirFile}
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, filepath.Join(home, ".config", "goose", "config.toml"))
	}
	return out
}

func (c *Config) resolveSecrets(getenv func(string) string) {
	if c.Server.APIKey == "" {
		if c.Server.APIKeyEnv != "" {
			c.Server.APIKey = getenv(c.Server.APIKeyEnv)
		} else {
			c.Server.APIKey = getenv(envKey)
		}
	}
	for i := range c.Providers {
		p := &c.Providers[i]
		if p.APIKey != "" {
			continue
		}
		if p.APIKeyEnv != "" {
			p.APIKey = getenv(p.APIKeyEnv)
			continue
		}
		for _, name := range defaultKeyEnv(p.Kind) {
			if v := getenv(name); v != "" {
				p.APIKey = v
				break
			}
		}
	}
}

func defaultKeyEnv(kind string) []string {
	switch kind {
	case "anthropic":
		return []string{"ANTHROPIC_API_KEY"}
	case "openai":
		return []string{"OPENAI_API_KEY"}
	case "google":
		return []string{"GEMINI_API_KEY", "GOOGLE_API_KEY"}
	case "azure":
		return []string{"AZURE_OPENAI_API_KEY"}
	case "cloudflare":
		return []string{"CLOUDFLARE_API_TOKEN"}
	case "openrouter":
		return []string{"OPENROUTER_API_KEY"}
	default:
		return nil
	}
}

func (c *Config) validate() error {
	if c.Server.Port < 1 || c.Server.Port > 65535 {
		return fmt.Errorf("server.port %d is out of range", c.Server.Port)
	}
	for _, knob := range []struct {
		name  string
		value int
	}{
		{"server.max_body_mb", c.Server.MaxBodyMB},
		{"server.max_concurrent", c.Server.MaxConcurrent},
		{"server.max_per_minute", c.Server.MaxPerMinute},
		{"server.max_response_mb", c.Server.MaxResponseMB},
		{"server.breaker_failures", c.Server.BreakerFailures},
	} {
		if knob.value < 0 {
			return fmt.Errorf("%s is %d; a limit cannot be negative", knob.name, knob.value)
		}
	}
	for _, d := range c.durations() {
		if d.value == "" {
			continue
		}
		if _, err := time.ParseDuration(d.value); err != nil {
			return fmt.Errorf("%s: %q is not a duration (try 30s, 2m, 1h)", d.name, d.value)
		}
	}
	if (c.Server.TLSCert == "") != (c.Server.TLSKey == "") {
		return fmt.Errorf("server.tls_cert and server.tls_key are set together or not at all")
	}
	for _, f := range []struct{ name, path string }{
		{"server.tls_cert", c.Server.TLSCert},
		{"server.tls_key", c.Server.TLSKey},
	} {
		if f.path == "" {
			continue
		}
		if _, err := os.Stat(f.path); err != nil {
			return fmt.Errorf("%s: %w", f.name, err)
		}
	}
	if len(c.Providers) == 0 {
		return fmt.Errorf("no providers configured")
	}
	seen := map[string]bool{}
	for i, p := range c.Providers {
		id := p.ID
		if id == "" {
			id = p.Kind
		}
		if id == "" {
			return fmt.Errorf("provider %d has no id and no kind", i)
		}
		if seen[id] {
			return fmt.Errorf("provider %q is defined twice", id)
		}
		seen[id] = true
		if !knownKind(p.Kind) {
			return fmt.Errorf("provider %q: unknown kind %q (known: %s)", id, p.Kind, strings.Join(kinds, ", "))
		}
		if err := checkOptions(id, p); err != nil {
			return err
		}
	}
	for i, r := range c.Routes {
		if len(r.Chain) == 0 {
			return fmt.Errorf("route %d (%s) has an empty chain", i, r.Name)
		}
		for _, f := range r.Formats {
			if !knownFormat(f) {
				return fmt.Errorf("route %d (%s): unknown format %q (known: %s)", i, r.Name, f, strings.Join(formats, ", "))
			}
		}
		for _, h := range r.Header {
			if key, _, _ := strings.Cut(h, "="); strings.TrimSpace(key) == "" {
				return fmt.Errorf("route %d (%s): header %q names no header", i, r.Name, h)
			}
		}
		for _, set := range []struct {
			name  string
			codes []string
		}{{"allow", r.Allow}, {"deny", r.Deny}} {
			for _, code := range set.codes {
				if !llm.KnownCode(code) {
					return fmt.Errorf("route %d (%s): %s names unknown diagnostic %q", i, r.Name, set.name, code)
				}
			}
		}
		for _, entry := range r.Chain {
			id, _, _ := strings.Cut(entry, ":")
			if !seen[strings.TrimSpace(id)] {
				return fmt.Errorf("route %d (%s) names unknown provider %q", i, r.Name, id)
			}
		}
	}
	if p := c.Server.Passthrough; p != "" && !seen[p] {
		return fmt.Errorf("server.passthrough names unknown provider %q", p)
	}
	if len(c.Providers) > 1 && len(c.Routes) == 0 {
		return fmt.Errorf("%d providers are configured but no routes; add [[route]] blocks", len(c.Providers))
	}
	seenMCP := map[string]bool{}
	for i, m := range c.MCP {
		if m.Name == "" {
			return fmt.Errorf("mcp %d has no name", i)
		}
		if seenMCP[m.Name] {
			return fmt.Errorf("mcp server %q is defined twice", m.Name)
		}
		seenMCP[m.Name] = true
		if m.Command == "" && m.URL == "" {
			return fmt.Errorf("mcp server %q needs a command or a url", m.Name)
		}
		for name := range m.Headers {
			if strings.TrimSpace(name) == "" {
				return fmt.Errorf("mcp server %q: a header with no name is a setting nothing carries", m.Name)
			}
		}
		if m.Transport != "" && !slices.Contains(transports, m.Transport) {
			return fmt.Errorf("mcp server %q: unknown transport %q (known: %s)", m.Name, m.Transport, strings.Join(transports, ", "))
		}
		if m.Transport == "stdio" && m.Command == "" {
			return fmt.Errorf("mcp server %q: a stdio server needs a command", m.Name)
		}
		if (m.Transport == "http" || m.Transport == "sse") && m.URL == "" {
			return fmt.Errorf("mcp server %q: a %s server needs a url", m.Name, m.Transport)
		}
	}
	for _, r := range c.Routes {
		for _, group := range r.Tools {
			if group != "mcp" {
				return fmt.Errorf("route %q names unknown tool group %q", r.Name, group)
			}
			if len(c.MCP) == 0 {
				return fmt.Errorf("route %q asks for tools but no [[mcp]] server is configured", r.Name)
			}
		}
	}
	for _, f := range c.Formats {
		if !knownFormat(f) {
			return fmt.Errorf("unknown format %q (known: %s)", f, strings.Join(formats, ", "))
		}
	}
	if len(c.Formats) == 0 {
		return fmt.Errorf("no formats enabled")
	}
	return nil
}

var kinds = []string{
	"anthropic", "openai", "google", "ollama", "compatible",
	"azure", "cloudflare", "openrouter", "bedrock", "vertex",
	"claude", "codex", "antigravity", "copilot",
}

func knownKind(k string) bool {
	for _, x := range kinds {
		if x == k {
			return true
		}
	}
	return false
}

func optionsFor(kind string) []string {
	switch kind {
	case "anthropic", "google":
		return []string{}
	case "openai":
		return []string{"organisation", "project"}
	case "compatible":
		return []string{"auth_header", "auth_prefix"}
	case "azure":
		return []string{"api_version", "deployment", "resource"}
	case "cloudflare":
		return []string{"account_id"}
	case "openrouter":
		return []string{"allow_fallbacks", "providers", "referer", "title"}
	case "ollama":
		return []string{"keep_alive"}
	case "bedrock":
		return []string{"access_key_id", "credentials_file", "profile", "region", "secret_access_key", "session_token"}
	case "vertex":
		return []string{"access_token", "credentials_file", "project", "publisher", "region"}
	case "claude", "codex", "antigravity", "copilot":
		return []string{"api_key_env_name", "dir", "env_allow", "max_concurrent", "max_turns", "sandbox", "system_mode", "temp_dir", "tools"}
	}
	return nil
}

func checkOptions(id string, p Provider) error {
	known := optionsFor(p.Kind)
	names := make([]string, 0, len(p.Options))
	for name := range p.Options {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if !slices.Contains(known, name) {
			if len(known) == 0 {
				return fmt.Errorf("provider %q: kind %s reads no options, but %q is set", id, p.Kind, name)
			}
			return fmt.Errorf("provider %q: unknown option %q for kind %s (known: %s)", id, name, p.Kind, strings.Join(known, ", "))
		}
	}
	for name := range p.Headers {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("provider %q: a header with no name is a setting nothing carries", id)
		}
	}
	for _, name := range []string{"max_turns", "max_concurrent"} {
		v := p.Options[name]
		if v == "" {
			continue
		}
		if n, err := strconv.Atoi(v); err != nil || n < 0 {
			return fmt.Errorf("provider %q: option %s is %q; it must be a whole number", id, name, v)
		}
	}
	if v := p.Options["allow_fallbacks"]; v != "" && v != "true" && v != "false" {
		return fmt.Errorf("provider %q: option allow_fallbacks is %q; it must be true or false", id, v)
	}
	for _, only := range []struct {
		name  string
		set   bool
		kinds []string
	}{
		{"dialect", p.Dialect != "", []string{"compatible"}},
		{"version", p.Version != "", []string{"anthropic"}},
		{"betas", len(p.Betas) > 0, []string{"anthropic"}},
		{"prefix", p.Prefix != "", []string{"google"}},
		{"allow_client_key", p.ClientKey, []string{"anthropic", "openai", "compatible", "google", "openrouter"}},
		{"headers", len(p.Headers) > 0, []string{
			"anthropic", "openai", "compatible", "google", "ollama",
			"azure", "cloudflare", "openrouter", "bedrock", "vertex",
		}},
	} {
		if only.set && !slices.Contains(only.kinds, p.Kind) {
			return fmt.Errorf("provider %q: %s is read by kind %s, not %s", id, only.name, strings.Join(only.kinds, ", "), p.Kind)
		}
	}
	if p.Kind == "vertex" && p.Model == "" && p.Options["publisher"] == "" {
		return fmt.Errorf("provider %q: vertex serves one publisher per entry, so it needs model or option publisher", id)
	}
	return nil
}

var transports = []string{"stdio", "http", "sse"}

var formats = []string{"openai", "openai-responses", "anthropic", "google", "ollama", "ollama-generate"}

func knownFormat(f string) bool {
	for _, x := range formats {
		if x == f {
			return true
		}
	}
	return false
}

func parseTimeout(s string) time.Duration {
	if s == "" {
		return defTimeo
	}
	d, err := time.ParseDuration(s)
	if err != nil || d <= 0 {
		return defTimeo
	}
	return d
}

func parseDuration(s string) time.Duration {
	d, err := time.ParseDuration(s)
	if err != nil || d < 0 {
		return 0
	}
	return d
}

type namedDuration struct{ name, value string }

func (c *Config) durations() []namedDuration {
	out := []namedDuration{
		{"server.breaker_cooldown", c.Server.BreakerCooldown},
		{"server.attempt_timeout", c.Server.AttemptTimeout},
	}
	for _, p := range c.Providers {
		id := p.ID
		if id == "" {
			id = p.Kind
		}
		out = append(out, namedDuration{"provider " + id + " timeout", p.Timeout})
	}
	for _, m := range c.MCP {
		out = append(out, namedDuration{"mcp " + m.Name + " timeout", m.Timeout})
	}
	return out
}
