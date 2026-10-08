// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package httpx serves a goose client over HTTP: the proxy, and optional.
package httpx

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

const (
	DefaultMaxBody    = 32 << 20
	DefaultProbeTTL   = 5 * time.Second
	UsageHeader       = "X-Goose-Usage"
	SessionHeader     = "X-Goose-Session"
	ProviderHeader    = "X-Goose-Provider"
	DiagnosticsHeader = "X-Goose-Diagnostics"
)

type Options struct {
	Codecs     []wire.Codec
	APIKey     string
	CORS       bool
	CORSOrigin string
	// Host is the name or address the listener is bound to, with or without
	// its port. On a loopback one with no APIKey and no ForwardedHeader, a
	// request addressed to a name that does not mean this machine is refused,
	// since that is what a page sends once it has pointed a name of its own at
	// 127.0.0.1.
	Host string

	MaxBody         int64
	Logger          *slog.Logger
	LogRequests     bool
	Passthrough     string
	MCP             http.Handler
	ProbeTTL        time.Duration
	MaxConcurrent   int
	MaxPerMinute    int
	ForwardedHeader string
	RedactUpstream  bool
	Metrics         *Metrics
}

type Server struct {
	client  *goose.Client
	opts    Options
	mux     *http.ServeMux
	handler http.Handler
	routes  []string

	inflight *gate
	rate     *limiter
	byRoute  map[string]wire.Codec

	probeMu   sync.Mutex
	probeAt   time.Time
	probeHeld []goose.Health
	probeOK   bool

	turnedMu sync.Mutex
	turned   map[string]bool
}

func New(c *goose.Client, o Options) (*Server, error) {
	if c == nil {
		return nil, llm.Errorf(llm.KindInvalidRequest, "nil client")
	}
	if len(o.Codecs) == 0 {
		return nil, llm.Errorf(llm.KindInvalidRequest, "no codecs registered")
	}
	if o.MaxBody <= 0 {
		o.MaxBody = DefaultMaxBody
	}
	if o.ProbeTTL <= 0 {
		o.ProbeTTL = DefaultProbeTTL
	}
	if o.CORS {
		if o.CORSOrigin == "" {
			o.CORSOrigin = "*"
		}
		if o.CORSOrigin == "*" && o.APIKey == "" {
			return nil, llm.Errorf(llm.KindInvalidRequest, "a wildcard CORS origin needs an API key")
		}
		if o.CORSOrigin == "null" && o.APIKey == "" {
			return nil, llm.Errorf(llm.KindInvalidRequest,
				`the CORS origin "null" is sent by a sandboxed frame on any site, so it needs an API key`)
		}
	}
	if o.Passthrough != "" {
		p, known := c.Provider(o.Passthrough)
		if !known {
			return nil, llm.Errorf(llm.KindInvalidRequest, "passthrough names no registered provider %q", o.Passthrough)
		}
		if _, can := p.(llm.Passer); !can {
			return nil, llm.Errorf(llm.KindInvalidRequest, "passthrough provider %q cannot forward opaque requests", o.Passthrough)
		}
	}
	s := &Server{
		client: c, opts: o, mux: http.NewServeMux(),
		inflight: newGate(o.MaxConcurrent),
		rate:     newLimiter(o.MaxPerMinute, c.Env().Now),
		byRoute:  map[string]wire.Codec{},
		turned:   map[string]bool{},
	}
	seen := map[string]bool{}
	for _, codec := range o.Codecs {
		for _, r := range codec.Routes() {
			pattern := r.Method + " " + r.Pattern
			if seen[pattern] {
				return nil, llm.Errorf(llm.KindInvalidRequest, "two codecs claim %s", pattern)
			}
			seen[pattern] = true
			s.byRoute[pattern] = codec
			s.mux.HandleFunc(pattern, s.handle(codec))
			s.routes = append(s.routes, pattern+" -> "+codec.Name())
		}
	}
	for _, codec := range o.Codecs {
		lister, can := codec.(wire.Lister)
		if !can {
			continue
		}
		for _, r := range lister.ModelRoutes() {
			pattern := r.Method + " " + r.Pattern
			if seen[pattern] {
				continue
			}
			seen[pattern] = true
			s.byRoute[pattern] = codec
			s.mux.HandleFunc(pattern, s.wireModels(lister))
			s.routes = append(s.routes, pattern+" -> "+codec.Name()+" models")
		}
	}
	if o.MCP != nil {
		s.mux.Handle("/mcp", http.StripPrefix("/mcp", o.MCP))
		s.mux.Handle("/mcp/", http.StripPrefix("/mcp", o.MCP))
		s.routes = append(s.routes, "POST /mcp -> tool servers")
	}
	if o.Metrics != nil {
		s.mux.Handle("GET /metrics", o.Metrics.Handler())
		s.routes = append(s.routes, "GET /metrics -> counters")
	}
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.HandleFunc("GET /ready", s.ready)
	s.mux.HandleFunc("GET /v1/models", s.models)
	s.mux.HandleFunc("GET /models", s.models)
	s.routes = append(s.routes,
		"GET /v1/models -> openai and anthropic",
		"GET /health, GET /ready -> liveness and readiness")
	if o.Passthrough != "" {
		s.mux.HandleFunc("/", s.passthrough)
	}
	s.handler = s.wrap()
	return s, nil
}

func (s *Server) Routes() []string { return append([]string(nil), s.routes...) }

func (s *Server) Handler() http.Handler { return s.handler }

func (s *Server) wrap() http.Handler {
	// Outermost first: the log records every refusal, including the ones the
	// limiter and the key check make themselves; a page the browser check
	// turns away spends no allowance, since on loopback it shares an address
	// with the operator's own client; the rate limiter counts unauthenticated
	// traffic too, because that is most of what it exists to shed; and the
	// concurrency gate sits inside the key check, because a slot is for work
	// that is actually going to happen.
	var h http.Handler = s.unserved()
	h = s.limitBody(h)
	if s.opts.Metrics != nil {
		h = s.observe(h)
	}
	if s.inflight != nil {
		h = s.gate(h)
	}
	if s.opts.APIKey != "" {
		h = s.authenticate(h)
	}
	if s.rate != nil {
		h = s.rateLimit(h)
	}
	h = s.browser(h)
	if s.opts.LogRequests && s.opts.Logger != nil {
		h = s.logRequests(h)
	}
	if s.opts.CORS {
		h = s.cors(h)
	}
	return h
}

func (s *Server) unserved() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		handler, pattern := s.mux.Handler(r)
		if pattern != "" {
			handler.ServeHTTP(w, r)
			return
		}
		probe := &verdict{}
		handler.ServeHTTP(probe, r)
		status := probe.status
		if status == 0 {
			status = http.StatusNotFound
		}
		if allow := probe.Header().Get("Allow"); allow != "" {
			w.Header().Set("Allow", allow)
		}
		e := llm.Errorf(llm.KindNotFound, "this proxy serves no %s %s", r.Method, r.URL.Path)
		if status == http.StatusMethodNotAllowed {
			e = llm.Errorf(llm.KindInvalidRequest, "%s is not a method this proxy serves on %s", r.Method, r.URL.Path)
		}
		e.Status = status
		s.refuse(w, r, e)
	})
}

type verdict struct {
	header http.Header
	status int
}

func (v *verdict) Header() http.Header {
	if v.header == nil {
		v.header = http.Header{}
	}
	return v.header
}

func (v *verdict) Write(p []byte) (int, error) { return len(p), nil }

func (v *verdict) WriteHeader(status int) {
	if v.status == 0 {
		v.status = status
	}
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.handler.ServeHTTP(w, r) }

func (s *Server) log() *slog.Logger {
	if s.opts.Logger != nil {
		return s.opts.Logger
	}
	return slog.New(slog.DiscardHandler)
}

func (s *Server) limitBody(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, s.opts.MaxBody)
		next.ServeHTTP(w, r)
	})
}

func (s *Server) authenticate(next http.Handler) http.Handler {
	key := []byte(s.opts.APIKey)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch s.served(r) {
		case "GET /health", "GET /ready":
			next.ServeHTTP(w, r)
			return
		}
		if !holdsKey(r, key) {
			s.refuse(w, r, llm.Errorf(llm.KindAuth, "invalid credentials"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func presented(r *http.Request) string {
	all := presentedAll(r)
	if len(all) == 0 {
		return ""
	}
	return all[0]
}

func presentedAll(r *http.Request) []string {
	var out []string
	for _, v := range []string{r.Header.Get("x-api-key"), r.Header.Get("x-goog-api-key")} {
		if v != "" {
			out = append(out, v)
		}
	}
	if v := r.Header.Get("Authorization"); strings.HasPrefix(v, "Bearer ") {
		out = append(out, strings.TrimPrefix(v, "Bearer "))
	}
	if v := r.URL.Query().Get("key"); v != "" {
		out = append(out, v)
	}
	return out
}

// handedOn is the credential a caller means for the backend: the one it
// presented, unless that is the proxy's own key, in which case whichever other
// one on the request is not. Which header carries which is the client's to
// choose.
func (s *Server) handedOn(r *http.Request, key string) string {
	if s.opts.APIKey == "" || key != s.opts.APIKey {
		return key
	}
	for _, candidate := range presentedAll(r) {
		if candidate != s.opts.APIKey {
			return candidate
		}
	}
	return ""
}

// travels is the routed path's rule for a caller's own key where there is one
// backend: a key whose header says which vendor minted it goes to that vendor
// and no other, and one whose header says nothing goes to the only backend.
func travels(scheme, backend llm.CredentialScheme) bool {
	return scheme == "" || backend == "" || scheme == backend
}

// A credential in the query is one like any in a header, and those are not
// forwarded as sent: the proxy's own would reach a vendor's URL and its logs,
// and a caller's would reach the backend whatever the operator allowed. The one
// the proxy reads as a key is handed on by the client-key rule instead.
var queryCredentials = []string{"key", "access_token"}

func withoutCredentials(raw string) string {
	if raw == "" {
		return raw
	}
	pairs := strings.Split(raw, "&")
	kept := pairs[:0]
	for _, pair := range pairs {
		name, _, _ := strings.Cut(pair, "=")
		if unescaped, err := url.QueryUnescape(name); err == nil {
			name = unescaped
		}
		if !slices.Contains(queryCredentials, name) {
			kept = append(kept, pair)
		}
	}
	return strings.Join(kept, "&")
}

func holdsKey(r *http.Request, key []byte) bool {
	ok := false
	for _, candidate := range presentedAll(r) {
		if subtle.ConstantTimeCompare([]byte(candidate), key) == 1 {
			ok = true
		}
	}
	return ok
}

// corsResponseHeaders are the ones cors() itself sets on every response.
var corsResponseHeaders = []string{
	"Access-Control-Allow-Origin",
	"Access-Control-Allow-Headers",
	"Access-Control-Allow-Methods",
	"Access-Control-Expose-Headers",
}

func (s *Server) cors(next http.Handler) http.Handler {
	origin := s.opts.CORSOrigin
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		if origin != "*" {
			w.Header().Add("Vary", "Origin")
		}
		w.Header().Set("Access-Control-Allow-Headers", allowedHeaders(r.Header.Get("Access-Control-Request-Headers")))
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Expose-Headers", strings.Join([]string{UsageHeader, SessionHeader, ProviderHeader, DiagnosticsHeader, "Retry-After"}, ", "))
		if r.Method == http.MethodOptions {
			w.Header().Add("Vary", "Access-Control-Request-Headers")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) trusted(r *http.Request) bool {
	if s.opts.APIKey == "" {
		return true
	}
	return holdsKey(r, []byte(s.opts.APIKey))
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	body := map[string]any{"status": "ok"}
	if s.trusted(r) {
		body["providers"] = s.client.Providers()
	}
	writeJSON(w, http.StatusOK, body)
}

func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	health, ok := s.probe(r.Context())
	status := http.StatusOK
	state := "ready"
	if !ok {
		status = http.StatusServiceUnavailable
		state = "degraded"
	}
	body := map[string]any{"status": state}
	if s.trusted(r) {
		body["providers"] = health
	}
	writeJSON(w, status, body)
}

func (s *Server) probe(ctx context.Context) ([]goose.Health, bool) {
	now := s.client.Env().Clock()
	s.probeMu.Lock()
	defer s.probeMu.Unlock()
	if !s.probeAt.IsZero() && now.Sub(s.probeAt) < s.opts.ProbeTTL {
		return s.probeHeld, s.probeOK
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	health, ok := s.client.Ready(ctx, s.opts.RedactUpstream)
	s.probeAt, s.probeHeld, s.probeOK = now, health, ok
	return health, ok
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	models, diags := s.client.Models(r.Context(), s.opts.RedactUpstream)
	noteDiagnostics(w, diags)
	data := make([]map[string]any, 0, len(models))
	for _, m := range models {
		entry := map[string]any{"id": m.ID, "object": "model", "type": "model"}
		if !m.Created.IsZero() {
			entry["created"] = m.Created.Unix()
			entry["created_at"] = m.Created.UTC().Format(time.RFC3339)
		}
		if m.Owner != "" {
			entry["owned_by"] = m.Owner
		}
		if m.Display != "" {
			entry["display_name"] = m.Display
		}
		data = append(data, entry)
	}
	body := map[string]any{"object": "list", "data": data, "has_more": false}
	if len(data) > 0 {
		body["first_id"] = models[0].ID
		body["last_id"] = models[len(models)-1].ID
	}
	writeJSON(w, http.StatusOK, body)
}

func noteDiagnostics(w http.ResponseWriter, diags llm.Diagnostics) {
	if codes := diags.Codes(); len(codes) > 0 {
		w.Header().Set(DiagnosticsHeader, strings.Join(codes, ","))
	}
}

func (s *Server) wireModels(lister wire.Lister) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		models, diags := s.client.Models(r.Context(), s.opts.RedactUpstream)
		noteDiagnostics(w, diags)
		out, err := lister.EncodeModels(models)
		if err != nil {
			writeJSON(w, http.StatusInternalServerError,
				map[string]any{"error": map[string]any{"message": "the model list could not be encoded"}})
			return
		}
		for k, vs := range out.Header {
			for _, v := range vs {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(out.Status)
		_, _ = w.Write(out.Body)
	}
}

func dotSegment(p string) bool {
	for _, seg := range strings.Split(p, "/") {
		if seg == "." || seg == ".." {
			return true
		}
	}
	return false
}

// passedMethods are the ones a vendor's API is called with. Anything else is
// not sent on the operator's credentials: TRACE has a server echo back the
// request it was sent, those credentials included, and CONNECT asks for a
// tunnel.
var passedMethods = []string{
	http.MethodGet, http.MethodHead, http.MethodPost, http.MethodPut,
	http.MethodPatch, http.MethodDelete, http.MethodOptions,
}

func tooLarge(err *http.MaxBytesError) *llm.Error {
	return &llm.Error{
		Kind: llm.KindInvalidRequest, Status: http.StatusRequestEntityTooLarge,
		Message: "request body: " + err.Error(),
	}
}

func (s *Server) passthrough(w http.ResponseWriter, r *http.Request) {
	p, ok := s.client.Provider(s.opts.Passthrough)
	if !ok {
		s.refuse(w, r, llm.Errorf(llm.KindNotFound, "the passthrough provider is not registered"))
		return
	}
	passer, ok := p.(llm.Passer)
	if !ok {
		writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": "the passthrough provider cannot forward opaque requests", "type": "unsupported"},
		})
		return
	}
	if !slices.Contains(passedMethods, r.Method) {
		w.Header().Set("Allow", strings.Join(passedMethods, ", "))
		e := llm.Errorf(llm.KindInvalidRequest, "%s is not a method this proxy forwards", r.Method)
		e.Status = http.StatusMethodNotAllowed
		s.refuse(w, r, e)
		return
	}
	// A query the proxy cannot read is one it cannot say is free of credentials,
	// so it goes no further; the mistake is the caller's, not the backend's.
	if _, err := url.ParseQuery(r.URL.RawQuery); err != nil {
		s.refuse(w, r, llm.Errorf(llm.KindInvalidRequest, "the query string could not be read: %v", err))
		return
	}
	// A Pass implementation joins this path onto base_url as a string and
	// reparses it, so a "." or ".." element could climb back out of it; that
	// reads the same once decoded whether it arrived literally or as
	// "%2E%2E" or "%2F", so the decoded path is what is checked.
	if dotSegment(r.URL.Path) {
		s.refuse(w, r, llm.Errorf(llm.KindInvalidRequest, `the request path may not hold a "." or ".." element`))
		return
	}
	// A body the listener would cut off partway is refused before any of it is
	// sent, as the routed path refuses it.
	if r.ContentLength > s.opts.MaxBody {
		s.refuse(w, r, tooLarge(&http.MaxBytesError{Limit: s.opts.MaxBody}))
		return
	}
	key := s.handedOn(r, presented(r))
	if !travels(schemeOf(r, key), p.Capabilities().KeyScheme) {
		key = ""
		w.Header().Set(DiagnosticsHeader, llm.PolicyKeyWithheld)
	}
	started := s.client.Env().Clock()
	resp, err := passer.Pass(r.Context(), &llm.PassRequest{
		// The raw, still-escaped path: the decoded r.URL.Path turns an
		// encoded "%2F" into a literal separator and an encoded "%3F" into a
		// literal '?', either of which a string-joined, reparsed URL would
		// then read as real path or query structure rather than opaque text.
		Method: r.Method, Path: r.URL.EscapedPath(), Query: withoutCredentials(r.URL.RawQuery),
		Header: r.Header, Body: r.Body, Key: key,
	})
	// Nobody counts what a forwarded call used, which is not the same as it
	// having used nothing; one that never left used nothing.
	rec := llm.Attempted{
		Provider: s.opts.Passthrough, Sent: true, Outcome: llm.OutcomeOK,
		Started: started, Elapsed: s.client.Env().Clock().Sub(started).Milliseconds(),
	}
	if err != nil {
		e := llm.AsError(err)
		// Booked the way the routed path books the same failure: a caller
		// gone or a wait that ran out is cancelled, not an upstream error the
		// backend is blamed for.
		rec.Sent, rec.Outcome, rec.Err, rec.Status = e.Sent, goose.OutcomeFor(e), e.Message, e.Status
		if e.Sent {
			rec.Usage = rec.Usage.Unmeasured()
		}
		// A body that gave no length and ran past the limit was cut off by the
		// listener while it was being forwarded. The mistake is the caller's,
		// so it is answered and booked as goose's refusal, not as the backend
		// failing.
		var over *http.MaxBytesError
		if errors.As(err, &over) {
			e = tooLarge(over)
			rec.Outcome, rec.Err, rec.Status = llm.OutcomeRejected, e.Message, e.Status
		}
		s.client.Record(context.WithoutCancel(r.Context()), rec, "passthrough")
		if over != nil {
			s.refuse(w, r, e)
			return
		}
		e = s.hide(e)
		writeJSON(w, http.StatusBadGateway, map[string]any{
			"error": map[string]any{"message": e.Message, "type": string(e.Kind)},
		})
		return
	}
	rec.Status, rec.Usage = resp.Status, rec.Usage.Unmeasured()
	if resp.Status >= 400 {
		rec.Outcome = llm.OutcomeError
	}
	s.client.Record(context.WithoutCancel(r.Context()), rec, "passthrough")
	defer resp.Body.Close()
	var drop []string
	if s.opts.CORS {
		// cors() already set these on w.Header() as the one answer this proxy
		// gives; an upstream that answers CORS itself, such as Ollama or
		// Anthropic's own browser access, must not add a second value beside
		// them.
		drop = corsResponseHeaders
	}
	httpcore.ForwardHeader(w.Header(), resp.Header, drop...)
	w.WriteHeader(resp.Status)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
		_, _ = io.Copy(flushWriter{w: w, f: f}, resp.Body)
		return
	}
	_, _ = io.Copy(w, resp.Body)
}

type flushWriter struct {
	w io.Writer
	f http.Flusher
}

func (fw flushWriter) Write(p []byte) (int, error) {
	n, err := fw.w.Write(p)
	fw.f.Flush()
	return n, err
}

func (s *Server) refuse(w http.ResponseWriter, r *http.Request, e *llm.Error) {
	status := e.Status
	if status < 400 {
		status = wire.StatusFor(e.Kind)
	}
	wire.SetRetryAfter(w.Header(), e)
	generic := func() {
		writeJSON(w, status, map[string]any{
			"error": map[string]any{"message": e.Message, "type": string(e.Kind)},
		})
	}
	codec := s.codecFor(r)
	if codec == nil {
		generic()
		return
	}
	out := codec.EncodeError(e, wire.Options{Format: codec.Name()})
	if out == nil {
		generic()
		return
	}
	overlay(w.Header(), out.Header)
	if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(out.Status)
	_, _ = w.Write(out.Body)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"response serialisation failed"}}`))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

func Loopback(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
