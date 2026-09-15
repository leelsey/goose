// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package wire is the seam between a client's wire format and goose's model.
package wire

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/leelsey/goose/llm"
)

const StatusClientClosed = 499

func StatusFor(k llm.ErrorKind) int {
	switch k {
	case llm.KindInvalidRequest, llm.KindLoss:
		return http.StatusBadRequest
	case llm.KindUnsupported:
		return http.StatusUnprocessableEntity
	case llm.KindAuth:
		return http.StatusUnauthorized
	case llm.KindPermission:
		return http.StatusForbidden
	case llm.KindNotFound, llm.KindNoRoute:
		return http.StatusNotFound
	case llm.KindRateLimit:
		return http.StatusTooManyRequests
	case llm.KindOverloaded, llm.KindSaturated:
		return http.StatusServiceUnavailable
	case llm.KindTimeout:
		return http.StatusGatewayTimeout
	case llm.KindCancelled:
		return StatusClientClosed
	case llm.KindInternal:
		return http.StatusInternalServerError
	default:
		return http.StatusBadGateway
	}
}

func SetRetryAfter(h http.Header, e *llm.Error) {
	if e == nil || e.RetryAfter <= 0 {
		return
	}
	secs := int(e.RetryAfter.Round(time.Second) / time.Second)
	if secs < 1 {
		secs = 1
	}
	h.Set("Retry-After", strconv.Itoa(secs))
}

type Inbound struct {
	Method string
	Path   string
	Query  url.Values
	Header http.Header
	Body   []byte
}

func (in *Inbound) Get(key string) string {
	if in == nil || in.Header == nil {
		return ""
	}
	return in.Header.Get(key)
}

type Outbound struct {
	Status int
	Header http.Header
	Body   []byte
}

type Options struct {
	Stream       bool
	StreamUsage  bool
	Session      string
	Key          string
	Format       string
	RequestModel string
	Header       http.Header
}

type Route struct {
	Method  string
	Pattern string
}

type Renders = llm.Renders

type Writer interface {
	Open(status int, header http.Header) error
	Write(p []byte) (int, error)
	Flush() error
}

type Codec interface {
	Name() string
	Routes() []Route
	Renders() Renders
	Decode(in *Inbound) (*llm.Request, Options, llm.Diagnostics, error)
	EncodeResponse(resp *llm.Response, opts Options) (*Outbound, llm.Diagnostics, error)
	EncodeStream(ctx context.Context, s *llm.Stream, opts Options, w Writer) (llm.Diagnostics, error)
	EncodeError(err *llm.Error, opts Options) *Outbound
}

type Lister interface {
	ModelRoutes() []Route
	EncodeModels(models []llm.Model) (*Outbound, error)
}

func WithExtra(body []byte, extra map[string]json.RawMessage) ([]byte, error) {
	if len(extra) == 0 {
		return body, nil
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return body, err
	}
	for k, v := range extra {
		if _, taken := obj[k]; taken {
			continue
		}
		obj[k] = v
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return body, err
	}
	return out, nil
}
