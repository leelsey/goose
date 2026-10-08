// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"errors"
	"fmt"
	"time"
)

type ErrorKind string

const (
	KindInvalidRequest ErrorKind = "invalid_request"
	KindUnsupported    ErrorKind = "unsupported"
	KindAuth           ErrorKind = "authentication"
	KindPermission     ErrorKind = "permission"
	KindNotFound       ErrorKind = "not_found"
	KindRateLimit      ErrorKind = "rate_limit"
	KindOverloaded     ErrorKind = "overloaded"
	KindUpstream       ErrorKind = "upstream"
	KindTransport      ErrorKind = "transport"
	KindTimeout        ErrorKind = "timeout"
	KindCancelled      ErrorKind = "cancelled"
	KindSaturated      ErrorKind = "saturated"
	KindLoss           ErrorKind = "loss"
	KindInternal       ErrorKind = "internal"
	KindNoRoute        ErrorKind = "no_route"
)

// Reason is the ledger's short account of a failure. A message parsed out of
// an upstream's body is the upstream's own sentence, and for a content filter
// that sentence is usually the passage it objected to, so redaction covers it
// too; goose's own words are kept, since nobody's request is in them.
func (e *Error) Reason(redact bool) string {
	if e == nil {
		return ""
	}
	if redact && e.Body != "" {
		return string(e.Kind)
	}
	return e.Message
}

// Detail is what the ledger is told beyond the reason, and is the upstream's
// body in full.
func (e *Error) Detail(redact bool) string {
	if e == nil || redact {
		return ""
	}
	return e.Body
}

type Error struct {
	Kind     ErrorKind
	Code     string
	Message  string
	Provider string
	Model    string
	Route    string

	Status     int
	RetryAfter time.Duration
	// Body is what the upstream sent to say it failed: a refusal's body, a 200
	// carrying an error, or a stream's error frame. It being set is what marks
	// Message as the upstream's words rather than goose's, so an adapter that
	// reads a failure out of what a backend sent sets it whatever the status.
	Body string

	Sent      bool
	Retryable bool

	Diagnostics []Diagnostic
	Usage       *Usage

	Err error
}

func (e *Error) Error() string {
	switch {
	case e == nil:
		return "<nil>"
	case e.Provider != "" && e.Status != 0:
		return fmt.Sprintf("%s: %s: %d: %s", e.Provider, e.Kind, e.Status, e.Message)
	case e.Provider != "":
		return fmt.Sprintf("%s: %s: %s", e.Provider, e.Kind, e.Message)
	default:
		return fmt.Sprintf("%s: %s", e.Kind, e.Message)
	}
}

func (e *Error) Unwrap() error { return e.Err }

func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Kind == e.Kind
}

func Errorf(kind ErrorKind, format string, args ...any) *Error {
	return &Error{Kind: kind, Message: fmt.Sprintf(format, args...)}
}

func AsError(err error) *Error {
	if err == nil {
		return nil
	}
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	return &Error{Kind: KindInternal, Message: err.Error(), Err: err}
}

func RetryableStatus(status int) bool {
	switch {
	case status == 408, status == 409, status == 425, status == 429:
		return true
	case status >= 500:
		return true
	default:
		return false
	}
}

func KindForStatus(status int) ErrorKind {
	switch {
	case status == 400, status == 413, status == 422:
		return KindInvalidRequest
	case status == 401:
		return KindAuth
	case status == 403:
		return KindPermission
	case status == 404:
		return KindNotFound
	case status == 408:
		return KindTimeout
	case status == 429:
		return KindRateLimit
	case status == 503, status == 529:
		return KindOverloaded
	case status >= 500:
		return KindUpstream
	default:
		return KindUpstream
	}
}
