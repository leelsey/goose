// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/llm"
)

type gate struct{ slots chan struct{} }

func newGate(n int) *gate {
	if n <= 0 {
		return nil
	}
	return &gate{slots: make(chan struct{}, n)}
}

func (g *gate) enter() bool {
	select {
	case g.slots <- struct{}{}:
		return true
	default:
		return false
	}
}

func (g *gate) leave() { <-g.slots }

type bucket struct {
	tokens float64
	seen   time.Time
}

type limiter struct {
	perMinute float64
	burst     float64
	now       func() time.Time

	mu      sync.Mutex
	callers map[string]*bucket
	swept   time.Time
}

func newLimiter(perMinute int, now func() time.Time) *limiter {
	if perMinute <= 0 {
		return nil
	}
	if now == nil {
		now = time.Now
	}
	return &limiter{
		perMinute: float64(perMinute),
		burst:     float64(perMinute),
		now:       now,
		callers:   map[string]*bucket{},
	}
}

func (l *limiter) allow(key string) bool {
	now := l.now()
	l.mu.Lock()
	defer l.mu.Unlock()
	l.sweep(now)

	b, ok := l.callers[key]
	if !ok {
		b = &bucket{tokens: l.burst, seen: now}
		l.callers[key] = b
	}
	b.tokens += now.Sub(b.seen).Minutes() * l.perMinute
	if b.tokens > l.burst {
		b.tokens = l.burst
	}
	b.seen = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

func (l *limiter) sweep(now time.Time) {
	if now.Sub(l.swept) < time.Minute {
		return
	}
	l.swept = now
	for key, b := range l.callers {
		if now.Sub(b.seen) > 2*time.Minute {
			delete(l.callers, key)
		}
	}
}

func caller(r *http.Request, forwarded string) string {
	if key := presented(r); key != "" {
		return "key:" + key
	}
	if forwarded != "" {
		if client := firstForwarded(r.Header.Get(forwarded)); client != "" {
			return "addr:" + client
		}
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	return "addr:" + host
}

func firstForwarded(value string) string {
	first, _, _ := strings.Cut(value, ",")
	first = strings.TrimSpace(first)
	if first == "" {
		return ""
	}
	if rest, ok := strings.CutPrefix(strings.ToLower(first), "for="); ok {
		first = strings.Trim(rest, `"`)
	}
	if host, _, err := net.SplitHostPort(first); err == nil {
		return host
	}
	return strings.Trim(first, "[]")
}

func (s *Server) limit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions || operational(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		if s.rate != nil && !s.rate.allow(caller(r, s.opts.ForwardedHeader)) {
			s.opts.Metrics.throttle()
			w.Header().Set("Retry-After", "1")
			s.refuse(w, r, llm.Errorf(llm.KindRateLimit, "too many requests"))
			return
		}
		if s.inflight != nil {
			if !s.inflight.enter() {
				s.opts.Metrics.refused()
				w.Header().Set("Retry-After", "1")
				s.refuse(w, r, llm.Errorf(llm.KindSaturated, "the proxy is at capacity"))
				return
			}
			defer s.inflight.leave()
		}
		next.ServeHTTP(w, r)
	})
}

func operational(path string) bool {
	switch path {
	case "/health", "/ready", "/metrics":
		return true
	}
	return false
}
