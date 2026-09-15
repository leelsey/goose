// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT
// by Leelsey (le@elsey.me.uk) - https://leelsey.me.uk

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/mcp"
	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openaichat"
	"github.com/leelsey/goose/wire/openairesp"
)

var version = "dev"

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "goose: "+err.Error())
		os.Exit(1)
	}
}

type flags struct {
	config   string
	listen   string
	key      string
	provider string
	model    string
	baseURL  string
	usageLog string
	verbose  bool
	version  bool
	check    bool
	stdio    bool
}

func run(args []string, stdout, stderr *os.File) error {
	var f flags
	if len(args) > 0 && args[0] == "mcp" {
		f.stdio, args = true, args[1:]
	}
	fs := flag.NewFlagSet("goose", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, usageText) }
	fs.StringVar(&f.config, "c", "", "configuration file")
	fs.StringVar(&f.listen, "l", "", "listen address (host:port)")
	fs.StringVar(&f.key, "k", "", "proxy API key")
	fs.StringVar(&f.provider, "p", "", "single provider kind (anthropic, openai, google, ollama, compatible)")
	fs.StringVar(&f.model, "m", "", "model the provider serves")
	fs.StringVar(&f.baseURL, "u", "", "provider base URL")
	fs.StringVar(&f.usageLog, "usage-log", "", "append one JSON object per upstream call to this file")
	fs.BoolVar(&f.verbose, "v", false, "log every request")
	fs.BoolVar(&f.version, "version", false, "print the version and exit")
	fs.BoolVar(&f.check, "check", false, "validate the configuration and exit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if f.version {
		fmt.Fprintf(stdout, "goose %s\n", version)
		return nil
	}

	cfg, origin, err := load(f.config)
	if err != nil {
		return err
	}
	if err := apply(cfg, f); err != nil {
		return err
	}
	cfg.resolveSecrets(os.Getenv)
	if err := cfg.validate(); err != nil {
		return err
	}

	level := slog.LevelInfo
	if f.verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(stderr, &slog.HandlerOptions{Level: level}))

	sinks, closeSinks, mem, err := buildSinks(cfg)
	if err != nil {
		return err
	}
	defer closeSinks()

	var metrics *httpx.Metrics
	if cfg.Server.Metrics {
		metrics = httpx.NewMetrics()
		sinks = append(sinks, metrics)
	}

	addr := net.JoinHostPort(cfg.Server.Host, strconv.Itoa(cfg.Server.Port))
	if !f.stdio {
		if cfg.Server.APIKey == "" && !httpx.Loopback(cfg.Server.Host) {
			return fmt.Errorf("refusing to serve %s without an API key; set server.api_key or %s", addr, envKey)
		}
		if servesTools(cfg) && cfg.Server.APIKey == "" && !httpx.Loopback(cfg.Server.Host) {
			return fmt.Errorf("refusing to serve tool servers on %s without an API key", addr)
		}
	}

	var tools *mcp.Manager
	if !f.check {
		tools, err = connectTools(context.Background(), cfg)
		if err != nil {
			return err
		}
		if tools == nil && (f.stdio || cfg.Server.MCPTools) {
			tools = newToolManager()
		}
	}
	if tools != nil {
		defer tools.Close()
		tools.Watch(context.Background(), 0, func(err error) {
			logger.Warn("tool list refresh failed", "error", err.Error())
		})
	}

	client, err := build(cfg, logger, sinks, tools)
	if err != nil {
		return err
	}
	defer client.Close(context.Background())

	if tools != nil && (f.stdio || cfg.Server.MCPTools) {
		if err := tools.Serve("goose", gooseTools(client)...); err != nil {
			return err
		}
	}
	if f.stdio && !f.check {
		return serveStdio(tools, stdout, stderr, cfg, origin, mem)
	}

	codecs, err := buildCodecs(cfg.Formats)
	if err != nil {
		return err
	}
	srv, err := httpx.New(client, httpx.Options{
		Codecs:          codecs,
		APIKey:          cfg.Server.APIKey,
		CORS:            cfg.Server.CORS,
		CORSOrigin:      cfg.Server.CORSOrigin,
		MaxBody:         int64(cfg.Server.MaxBodyMB) << 20,
		Logger:          logger,
		LogRequests:     f.verbose,
		Passthrough:     cfg.Server.Passthrough,
		MCP:             mcpHandler(tools),
		MaxConcurrent:   cfg.Server.MaxConcurrent,
		MaxPerMinute:    cfg.Server.MaxPerMinute,
		ForwardedHeader: cfg.Server.ForwardedHeader,
		Metrics:         metrics,
		RedactUpstream:  cfg.Server.RedactUpstream,
	})
	if err != nil {
		return err
	}

	if f.check {
		fmt.Fprintf(stdout, "goose %s: configuration from %s is valid\n", version, origin)
		for _, r := range srv.Routes() {
			fmt.Fprintln(stdout, "  "+r)
		}
		for _, m := range cfg.MCP {
			fmt.Fprintf(stdout, "  POST /mcp -> %s (not started)\n", m.Name)
		}
		if cfg.Server.MCPTools || f.stdio {
			fmt.Fprintln(stdout, "  POST /mcp -> goose's own tools")
		}
		return nil
	}

	banner(stdout, cfg, origin, addr, srv.Routes(), tools)

	server := &http.Server{
		Addr:              addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 30 * time.Second,
		WriteTimeout:      60 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errs := make(chan error, 1)
	go func() {
		err := listen(server, cfg)
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			errs <- err
			return
		}
		errs <- nil
	}()

	select {
	case err := <-errs:
		return err
	case <-ctx.Done():
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		_ = server.Close()
	}
	if mem != nil && cfg.Usage.Summary {
		summary(stdout, mem)
	}
	return nil
}

func listen(server *http.Server, cfg *Config) error {
	if cfg.Server.TLSCert != "" {
		server.TLSConfig = &tls.Config{MinVersion: tls.VersionTLS12}
		return server.ListenAndServeTLS(cfg.Server.TLSCert, cfg.Server.TLSKey)
	}
	return server.ListenAndServe()
}

func servesTools(cfg *Config) bool {
	return len(cfg.MCP) > 0 || cfg.Server.MCPTools
}

func mcpHandler(m *mcp.Manager) http.Handler {
	if m == nil {
		return nil
	}
	return m.Handler()
}

func apply(cfg *Config, f flags) error {
	if f.listen != "" {
		host, port, err := net.SplitHostPort(f.listen)
		if err != nil {
			return fmt.Errorf("listen %q: %w", f.listen, err)
		}
		n, err := strconv.Atoi(port)
		if err != nil {
			return fmt.Errorf("listen %q: %w", f.listen, err)
		}
		cfg.Server.Host, cfg.Server.Port = host, n
	}
	if f.key != "" {
		cfg.Server.APIKey = f.key
	}
	if f.usageLog != "" {
		cfg.Usage.Log = f.usageLog
	}
	if f.provider == "" && (f.model != "" || f.baseURL != "") {
		return fmt.Errorf("-m and -u name the model and base URL for -p; give a provider or set them in the file")
	}
	if f.provider != "" {
		cfg.Providers = []Provider{{
			ID: f.provider, Kind: f.provider, Model: f.model, BaseURL: f.baseURL,
		}}
		cfg.Routes = nil
	}
	return nil
}

func buildSinks(cfg *Config) ([]usage.Sink, func(), *usage.Memory, error) {
	var sinks []usage.Sink
	closers := []func(){}
	var mem *usage.Memory
	if cfg.Usage.Summary {
		mem = usage.NewMemory()
		sinks = append(sinks, mem)
	}
	if cfg.Usage.Log != "" {
		f, err := os.OpenFile(cfg.Usage.Log, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
		if err != nil {
			return nil, func() {}, nil, fmt.Errorf("usage log: %w", err)
		}
		sinks = append(sinks, usage.NewJSONL(f))
		closers = append(closers, func() { _ = f.Close() })
	}
	return sinks, func() {
		for _, c := range closers {
			c()
		}
	}, mem, nil
}

func buildCodecs(names []string) ([]wire.Codec, error) {
	var out []wire.Codec
	add := func(codecs ...wire.Codec) {
		for _, c := range codecs {
			if !hasCodec(out, c.Name()) {
				out = append(out, c)
			}
		}
	}
	for _, name := range names {
		switch name {
		case "openai":
			add(openaichat.New(openaichat.Options{}))
		case "openai-responses":
			add(openairesp.New(openairesp.Options{}))
		case "anthropic":
			add(anthropicmsg.New(anthropicmsg.Options{}))
		case "google":
			add(geminiapi.New(geminiapi.Options{}))
		case "ollama":
			add(ollamaapi.NewChat(ollamaapi.Options{}), ollamaapi.NewGenerate(ollamaapi.Options{}))
		case "ollama-generate":
			add(ollamaapi.NewGenerate(ollamaapi.Options{}))
		default:
			return nil, fmt.Errorf("unknown format %q", name)
		}
	}
	return out, nil
}

func hasCodec(list []wire.Codec, name string) bool {
	for _, c := range list {
		if c.Name() == name {
			return true
		}
	}
	return false
}

func banner(out *os.File, cfg *Config, origin, addr string, routes []string, tools *mcp.Manager) {
	scheme := "http"
	if cfg.Server.TLSCert != "" {
		scheme = "https"
	}
	fmt.Fprintf(out, "goose %s  config %s\n", version, origin)
	fmt.Fprintf(out, "listening on %s://%s\n", scheme, addr)
	for _, p := range cfg.Providers {
		id := p.ID
		if id == "" {
			id = p.Kind
		}
		key := "no key"
		if p.APIKey != "" {
			key = "key set"
		}
		fmt.Fprintf(out, "  provider %-12s %-11s %-24s %s\n", id, p.Kind, dashIfEmpty(p.Model), key)
	}
	for _, r := range cfg.Routes {
		fmt.Fprintf(out, "  route    %-12s %-11s -> %s\n", dashIfEmpty(r.Name), strings.Join(r.Models, ","), strings.Join(r.Chain, " then "))
	}
	if tools != nil {
		for _, t := range tools.Servers() {
			count := fmt.Sprintf("%d tools", t.Tools)
			if n := len(t.Refused); n > 0 {
				count += fmt.Sprintf(" (%d refused: %s)", n, strings.Join(t.Refused, ", "))
			}
			fmt.Fprintf(out, "  tools    %-12s %-24s %-11s %-11s %s\n",
				t.Name, dashIfEmpty(t.Info.Name), dashIfEmpty(t.Info.Version),
				dashIfEmpty(t.Info.ProtocolVersion), count)
		}
	}
	for _, r := range routes {
		fmt.Fprintf(out, "  serves   %s\n", r)
	}
	if cfg.Server.APIKey == "" {
		fmt.Fprintln(out, "  warning  no API key: anyone who can reach this port can spend your credit")
	}
	if cfg.Server.TLSCert == "" && !httpx.Loopback(cfg.Server.Host) {
		fmt.Fprintln(out, "  warning  no TLS: keys and prompts cross the network in the clear")
	}
	fmt.Fprintln(out)
}

func dashIfEmpty(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func summary(out *os.File, mem *usage.Memory) {
	total := mem.Total()
	attempts, unmeasured := mem.Stats()
	fmt.Fprintf(out, "\nusage: %d upstream attempts, %d unmeasured\n", attempts, unmeasured)
	fmt.Fprintf(out, "       prompt %d (cache read %d, write %d), completion %d (reasoning %d)\n",
		total.Prompt, total.CacheRead, total.CacheWrite, total.Completion, total.Reasoning)
	if total.Cost != nil {
		fmt.Fprintf(out, "       cost %.4f %s\n", total.Cost.Amount, total.Cost.Currency)
	}
}

const usageText = `goose - LLM proxy and provider bridge

usage: goose [flags]
       goose mcp [flags]   serve the tool protocol on stdin/stdout

  -c <file>        configuration file (default .goose.toml, goose.toml,
                   ~/.config/goose/config.toml)
  -l <host:port>   listen address (default 127.0.0.1:11234)
  -k <key>         proxy API key (or set GOOSE_PROXY_KEY)
  -p <kind>        serve one provider: anthropic, openai, google, ollama, compatible
  -m <model>       model that provider serves
  -u <url>         provider base URL
  -usage-log <f>   append one JSON object per upstream call to this file
  -check           validate the configuration and exit
  -v               log every request
  -version         print the version and exit
`
