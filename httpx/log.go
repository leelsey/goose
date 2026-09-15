// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"net/http"
	"time"
)

type observer struct {
	http.ResponseWriter
	status  int
	written int64
}

func (o *observer) WriteHeader(status int) {
	if o.status == 0 {
		o.status = status
	}
	o.ResponseWriter.WriteHeader(status)
}

func (o *observer) Write(p []byte) (int, error) {
	if o.status == 0 {
		o.status = http.StatusOK
	}
	n, err := o.ResponseWriter.Write(p)
	o.written += int64(n)
	return n, err
}

func (o *observer) Flush() {
	if f, ok := o.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (o *observer) Unwrap() http.ResponseWriter { return o.ResponseWriter }

func (s *Server) logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		o := &observer{ResponseWriter: w}
		next.ServeHTTP(o, r)

		attrs := []any{
			"method", r.Method,
			"path", r.URL.Path,
			"status", o.status,
			"bytes", o.written,
			"ms", time.Since(started).Milliseconds(),
		}
		if provider := o.Header().Get(ProviderHeader); provider != "" {
			attrs = append(attrs, "provider", provider)
		}
		if u := o.Header().Get(UsageHeader); u != "" {
			attrs = append(attrs, "usage", u)
		}
		if d := o.Header().Get(DiagnosticsHeader); d != "" {
			attrs = append(attrs, "lost", d)
		}
		s.log().Info("request", attrs...)
	})
}
