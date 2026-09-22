// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"time"
)

type Provider interface {
	ID() string
	Capabilities() Capabilities
	Complete(ctx context.Context, a *Attempt) (*Response, error)
	Stream(ctx context.Context, a *Attempt) (*Stream, error)
}

type Closer interface {
	Close(ctx context.Context) error
}

type Binder interface {
	Bind(Env) error
}

type Gated interface {
	Acquire(ctx context.Context) (release func(), err error)
}

type Passer interface {
	Pass(ctx context.Context, r *PassRequest) (*PassResponse, error)
}

type Lister interface {
	Models(ctx context.Context) ([]Model, error)
}

type Prober interface {
	Probe(ctx context.Context) error
}

type Model struct {
	ID      string
	Display string
	Owner   string
	Created time.Time
}

type PassRequest struct {
	Method string
	Path   string
	Query  string
	Header http.Header
	Body   io.Reader
	Key    string
}

type PassResponse struct {
	Status int
	Header http.Header
	Body   io.ReadCloser
}

type Env struct {
	HTTP   *http.Client
	Logger *slog.Logger
	Now    func() time.Time
}

func (e Env) Client() *http.Client {
	if e.HTTP != nil {
		return e.HTTP
	}
	return http.DefaultClient
}

func (e Env) Clock() time.Time {
	if e.Now != nil {
		return e.Now()
	}
	return time.Now()
}

func (e Env) Log() *slog.Logger {
	if e.Logger != nil {
		return e.Logger
	}
	return discard
}

var discard = slog.New(slog.DiscardHandler)

type Attempt struct {
	Request  *Request
	Provider string
	Model    string
	Stream   bool

	Format  string
	Renders Renders
	Route   string
	Index   int
	Session string

	Credential Credential
	Tools      []Tool

	Policy   LossPolicy
	Recorder Recorder
	Limits   Limits

	// Redact says the ledger is to be kept free of the upstream's own words.
	// It travels with the attempt because rows are booked from under the
	// provider as well as above it.
	Redact bool

	Book func(Attempted)
}

func (a *Attempt) BookExtra(rec Attempted) {
	if a == nil || a.Book == nil {
		return
	}
	if rec.Provider == "" {
		rec.Provider = a.Provider
	}
	if rec.Model == "" {
		rec.Model = a.Model
	}
	if rec.Route == "" {
		rec.Route = a.Route
	}
	a.Book(rec)
}

func (a *Attempt) Note(d Diagnostic) {
	if a == nil || a.Recorder == nil {
		return
	}
	if d.Provider == "" {
		d.Provider = a.Provider
	}
	if d.Model == "" {
		d.Model = a.Model
	}
	if d.Format == "" {
		d.Format = a.Format
	}
	a.Recorder.Note(a.Policy.Apply(d))
}

type CredentialScheme string

const (
	SchemeAnthropic CredentialScheme = "anthropic"
	SchemeOpenAI    CredentialScheme = "openai"
	SchemeGoogle    CredentialScheme = "google"
)

type Credential struct {
	Key     string
	Scheme  CredentialScheme
	Headers map[string]string
}

func (c Credential) Empty() bool { return c.Key == "" && len(c.Headers) == 0 }

type Limits struct {
	MaxRequestBytes  int64
	MaxResponseBytes int64
	MaxFrameBytes    int
	MaxBufferBytes   int
	Timeout          time.Duration
}

type CeilingField string

const (
	CeilingNone          CeilingField = ""
	CeilingMaxTokens     CeilingField = "max_tokens"
	CeilingMaxCompletion CeilingField = "max_completion_tokens"
	CeilingPerModel      CeilingField = "per_model"
)

type UsageShape struct {
	StreamUsage      bool
	StreamUsageOptIn bool
	Cost             bool
	MultiTurn        bool
}

type Capabilities struct {
	Streaming bool
	Usage     UsageShape

	SystemPrompt  bool
	Tools         bool
	ParallelTools bool
	ToolChoice    bool
	ToolResults   bool
	ToolErrors    bool

	ImageIn bool

	Thinking           bool
	ThinkingSignatures bool
	RedactedThinking   bool
	Effort             bool

	ResponseFormat bool
	Sessions       bool
	MCPServers     bool
	Extensions     []string

	Candidates       int
	CacheBreakpoints int

	Ceiling CeilingField

	MaxInputBytes int64

	KeyScheme CredentialScheme
}

type Renders struct {
	Thinking   bool
	ToolCalls  bool
	Images     bool
	Candidates int
}

func (r Renders) Declared() bool {
	return r.Thinking || r.ToolCalls || r.Images || r.Candidates > 0
}

func (c Capabilities) Takes(wire string) bool {
	if wire == "" {
		return false
	}
	for _, w := range c.Extensions {
		if w == wire {
			return true
		}
	}
	return false
}

func (c Capabilities) MaxCandidates() int {
	if c.Candidates < 1 {
		return 1
	}
	return c.Candidates
}
