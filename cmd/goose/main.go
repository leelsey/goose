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
	"io/fs"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
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
	// Parsing stops at the first word that is not a flag, so anything after it
	// would otherwise be dropped without a word, flags and all.
	if rest := fs.Args(); len(rest) > 0 {
		if slices.Contains(rest, "mcp") && !f.stdio {
			return fmt.Errorf("mcp comes before any flag: goose mcp [flags]")
		}
		return fmt.Errorf("unexpected argument %q", rest[0])
	}
	if f.version {
		fmt.Fprintf(stdout, "goose %s\n", version)
		return nil
	}

	level := slog.LevelInfo
	if f.verbose {
		level = slog.LevelDebug
	}
	logger := slog.New(slog.NewTextHandler(stderr, &slog.HandlerOptions{Level: level}))

	cfg, origin, err := load(f.config)
	if err != nil {
		return err
	}
	skipped := unread(f.config)
	for _, name := range skipped {
		logger.Warn("a configuration file in the working directory is read only when named", "skipped", name, "use", "-c "+name)
	}
	if err := apply(cfg, f); err != nil {
		return err
	}
	if origin == noFile && len(cfg.Providers) == 0 {
		return nothingRead(skipped)
	}
	if err := cfg.validate(!f.stdio); err != nil {
		return err
	}
	notes, err := cfg.resolveSecrets(os.Getenv, !f.stdio)
	if err != nil {
		return err
	}
	for _, note := range notes {
		logger.Warn(note)
	}

	sinks, closeSinks, mem, err := buildSinks(cfg, f.check)
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

	calls := &inflight{}
	if tools != nil && (f.stdio || cfg.Server.MCPTools) {
		own := gooseTools(client, cfg, logger)
		// The listener logs every request under -v, a call to /mcp among
		// them; goose mcp has no listener, so its own tools say it instead.
		if f.stdio && f.verbose {
			own = logged(own, logger)
		}
		if err := tools.Serve("goose", calls.tools(own)...); err != nil {
			return err
		}
	}
	// goose mcp binds no port and speaks no vendor wire, so what the listener
	// alone reads is neither checked nor listed for it.
	if f.stdio && f.check {
		fmt.Fprintf(stdout, "goose %s: configuration from %s is valid\n", version, origin)
		for _, m := range cfg.MCP {
			fmt.Fprintf(stdout, "  tools -> %s (not started)\n", m.Name)
		}
		fmt.Fprintln(stdout, "  tools -> goose's own")
		return nil
	}
	if f.stdio {
		err := serveStdio(tools, stdout, stderr, origin)
		calls.settle(settleFor, logger)
		if mem != nil && cfg.Usage.Summary {
			summary(stderr, mem)
		}
		return err
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
		Host:            cfg.Server.Host,
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
		if cfg.Server.MCPTools {
			fmt.Fprintln(stdout, "  POST /mcp -> goose's own tools")
		}
		return nil
	}

	banner(stdout, cfg, origin, addr, srv.Routes(), tools)

	server := &http.Server{
		Addr:              addr,
		Handler:           calls.handler(srv.Handler()),
		ReadHeaderTimeout: 30 * time.Second,
		// A request body arrives in one go; only the answer is slow. Without
		// this a client dribbling its body holds a connection, and a
		// concurrency slot behind it, for as long as it cares to.
		ReadTimeout:  5 * time.Minute,
		WriteTimeout: 60 * time.Minute,
		IdleTimeout:  2 * time.Minute,
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

	// The first signal asks for the open requests to finish, and a stream can
	// take the whole budget doing so; one more is how an operator at a
	// terminal says not to wait for them.
	again := make(chan os.Signal, 1)
	signal.Notify(again, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(again)
	stop()
	shutdown(server, calls, 10*time.Second, logger, again)
	if mem != nil && cfg.Usage.Summary {
		summary(stdout, mem)
	}
	return nil
}

// nothingRead is the error for a start with no file and no -p. Said bare, "no
// providers configured" sends the operator looking for a mistake in a file
// that was never opened.
func nothingRead(skipped []string) error {
	if len(skipped) > 0 {
		return fmt.Errorf("no providers configured: %s in the working directory is read only when named; give -c %s", skipped[0], skipped[0])
	}
	where := "~/.config/goose/config.toml"
	if home := homeConfig(); home != "" {
		where = home
	}
	return fmt.Errorf("no providers configured: no file was named with -c and %s does not exist; give -c <file>, or -p <kind> to serve one provider", where)
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
			ID: f.provider, Kind: f.provider, Model: f.model, BaseURL: f.baseURL, typed: true,
		}}
		// Both name the file's providers, which are no longer there.
		cfg.Routes = nil
		cfg.Server.Passthrough = ""
	}
	return nil
}

// buildSinks opens what a run will write usage to. A check serves nothing, so
// it is validated without opening — and so without needing write access to —
// the ledger; the directory it sits in belongs to whoever runs the service,
// which need not be whoever runs -check.
func buildSinks(cfg *Config, check bool) ([]usage.Sink, func(), *usage.Memory, error) {
	if cfg.Usage.Log != "" && check {
		if err := ledgerPlace(cfg.Usage.Log); err != nil {
			return nil, func() {}, nil, fmt.Errorf("usage log: %w", err)
		}
	}
	var sinks []usage.Sink
	closers := []func(){}
	var mem *usage.Memory
	if cfg.Usage.Summary {
		mem = usage.NewMemory()
		sinks = append(sinks, mem)
	}
	if cfg.Usage.Log != "" && !check {
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

// ledgerPlace refuses, without writing anything, a ledger that no account
// could open: one whose directory is missing, or that is itself a directory.
// What cannot be looked at, for want of permission, is left for the run to
// find out, since it may belong to an account that can.
func ledgerPlace(path string) error {
	dir := filepath.Dir(path)
	if info, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("%s: the directory does not exist", dir)
	} else if err == nil && !info.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		return fmt.Errorf("%s is a directory", path)
	}
	return nil
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
			if t.Truncated {
				count += " (and more than goose will hold)"
			}
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

  -c <file>        configuration file (default ~/.config/goose/config.toml; a
                   goose.toml or .goose.toml in the working directory is read
                   only when named here)
  -l <host:port>   listen address (default 127.0.0.1:11234)
  -k <key>         proxy API key; prefer GOOSE_PROXY_KEY, since anyone on this
                   host can read an argument list
  -p <kind>        serve one provider: anthropic, openai, google, ollama, compatible
  -m <model>       model that provider serves
  -u <url>         provider base URL
  -usage-log <f>   append one JSON object per upstream call to this file
  -check           validate the configuration and exit
  -v               log every request
  -version         print the version and exit
`
