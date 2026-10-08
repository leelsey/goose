// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"slices"
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

// Precheck is implemented by a provider that can find some of what building a
// request will lose cheaply, without spending anything to learn it: no
// process is run and no request is sent. It returns an error for a request
// the provider will not send whatever the policy. A caller checks what it
// finds against the attempt's policy, and takes its error as the attempt's
// refusal, before a Gated provider is asked to wait for a slot, so a request
// the provider was always going to refuse never queues behind one that could
// actually have used it. What it finds is still the provider's to note once
// it builds the request, so a caller notes it only where it refuses the
// attempt. Complete and Stream are free to check the same policy again once
// the request is actually built, since Precheck need not find every loss
// building the whole of it would.
type Precheck interface {
	Precheck(a *Attempt) (Diagnostics, error)
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

	// refused is what has been noted of the attempt that its policy will not
	// take, for Check.
	refused Diagnostics
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
	if a == nil {
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
	d = a.Policy.Apply(d)
	if d.Severity == SeverityFatal {
		a.refused = append(a.refused, d)
	}
	if a.Recorder != nil {
		a.Recorder.Note(d)
	}
}

// Check refuses the attempt when anything noted of it so far is a loss its
// policy will not take. A provider calls it once its request is built and
// before sending it: building is where it learns what the backend has no room
// for, and a loss refused there costs nothing, so the next link may still carry
// the request whole.
func (a *Attempt) Check() error {
	if a == nil || len(a.refused) == 0 {
		return nil
	}
	err := a.Policy.Check(a.refused)
	if err == nil {
		return nil
	}
	e := AsError(err)
	e.Provider, e.Model, e.Sent = a.Provider, a.Model, false
	return e
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
	// Signer is the family whose signatures this backend verifies. A signature
	// another family minted is stripped before the call; empty leaves every
	// signature to the backend.
	Signer           Family
	RedactedThinking bool
	Effort           bool

	ResponseFormat bool
	Sessions       bool
	MCPServers     bool
	Extensions     []string

	Candidates       int
	CacheBreakpoints int

	Ceiling CeilingField

	MaxInputBytes int64

	KeyScheme CredentialScheme
	// ClientKey says a caller's own credential is sent in place of the
	// operator's, so a refusal of the key is a refusal of the caller's.
	ClientKey bool
}

type Renders struct {
	Thinking   bool
	ToolCalls  bool
	Images     bool
	Candidates int
	// Raw names the vendors whose blocks, carried verbatim as Raw, the
	// destination writes out whole. Any other vendor's it drops.
	Raw []string
}

func (r Renders) Declared() bool {
	return r.Thinking || r.ToolCalls || r.Images || r.Candidates > 0
}

// Unrendered is a loss for each part of an answer that the destination has no
// field for. Its encoder drops them; known here, they are held to the policy
// before anything is delivered rather than after.
func (r Renders) Unrendered(resp *Response) Diagnostics {
	if resp == nil {
		return nil
	}
	var out Diagnostics
	for _, c := range resp.Choices {
		out = append(out, r.unrendered(c.Message.Content)...)
	}
	return out
}

// unrendered weighs parts as they arrive: a block carried verbatim from a wire
// the destination does not speak, and reasoning, an image or a tool call where
// it has nowhere to put one. A destination that declared nothing takes every
// part as it is, which is what a library caller does.
func (r Renders) unrendered(parts []Part) Diagnostics {
	if !r.Declared() {
		return nil
	}
	var out Diagnostics
	for _, p := range parts {
		switch v := p.(type) {
		case *Raw:
			if !slices.Contains(r.Raw, v.Vendor) {
				out = append(out, Diagnostic{
					Code: DropUnknownBlock, Class: ClassContent, Stage: StageEncode,
					Severity: SeverityLoss, Field: v.Type,
					Detail: "content block from " + v.Vendor + " has no shape on this wire",
				})
			}
		case *Thinking:
			if !r.Thinking && v.Text != "" {
				out = append(out, Diagnostic{
					Code: DropThinking, Class: ClassContent, Stage: StageEncode,
					Severity: SeverityLoss,
					Detail:   "this format has no field for reasoning, so it would not be delivered",
				})
			}
		case *Image:
			if !r.Images {
				out = append(out, Diagnostic{
					Code: DropImage, Class: ClassContent, Stage: StageEncode,
					Severity: SeverityLoss,
					Detail:   "this format has no field for an image in an answer, so it would not be delivered",
				})
			}
		case *ToolUse:
			if !r.ToolCalls {
				out = append(out, Diagnostic{
					Code: DropToolCall, Class: ClassContent, Stage: StageEncode,
					Severity: SeverityLoss, Field: v.Name,
					Detail: "this format has no field for a tool call, so it would not be delivered",
				})
			}
		}
	}
	return out
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
