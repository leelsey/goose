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

// A caller has some say in its own key, so the number of them a flood can mint
// is bounded: past the cap everyone new shares one bucket until the sweep makes
// room. Sharing is the safe direction — it throttles harder, never less.
const (
	maxBuckets = 4096

	// One bucket for everyone who proved the proxy's own key, since that is one
	// principal however many of them there are.
	provenBucket = "proven"
)

func (l *limiter) allow(key string) bool {
	now := l.now()
	l.mu.Lock()
	defer l.mu.Unlock()
	l.sweep(now)

	b, ok := l.callers[key]
	if !ok && len(l.callers) >= maxBuckets && !reserved(key) {
		// The overflow bucket is per class, so a flood that fills the table
		// cannot spend an allowance belonging to callers on the other side of
		// the key check.
		key = "overflow:" + bucketClass(key)
		b, ok = l.callers[key]
	}
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

// A key the proxy chooses rather than reads is never crowded out: there is one
// of it, so it cannot be what fills the table.
func reserved(key string) bool { return key == provenBucket }

func bucketClass(key string) string {
	if strings.HasPrefix(key, "key:") {
		return "key"
	}
	return "addr"
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
	return addr(r, forwarded)
}

func addr(r *http.Request, forwarded string) string {
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

// A bucket is only as good as what it is keyed on, and a caller has a say in
// most of what it sends. The key check passes on any credential presented while
// presented() reads the first, so reading the bucket off the request lets a
// caller holding the real key mint a fresh allowance with every request by
// sending a nonce beside it.
//
// So the key is what was actually proven. Where the proxy has a key of its own,
// everyone who holds it is one principal as far as this proxy can tell, and
// they share one allowance; a deployment that gates tenants with a shared key
// while they carry their own upstream keys shares it too. Where the proxy has
// no key, nothing is proven and the credential a caller carries upstream is the
// only thing telling two of them apart. An unproven caller gets neither, and
// shares the bucket of the connection it arrived on.
func (s *Server) bucket(r *http.Request) string {
	switch {
	case s.opts.APIKey == "":
		return caller(r, s.opts.ForwardedHeader)
	case s.trusted(r):
		return provenBucket
	default:
		return addr(r, "")
	}
}

// The nearest hop appended the last entry, so that is the address the proxy in
// front actually saw. The leftmost is whatever the client wrote, which a caller
// wanting its own allowance can write afresh every request — precise where the
// proxy replaces the header, and no limit at all where the proxy appends.
// Behind a chain this is the nearest proxy rather than the client, so callers
// behind one hop share an allowance; a deployment wanting them apart needs a
// front end that replaces the header rather than adding to it.
func firstForwarded(value string) string {
	first := value
	if i := strings.LastIndex(value, ","); i >= 0 {
		first = value[i+1:]
	}
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

// rateLimit counts a caller's requests, and so belongs outside the key check:
// the flooding it exists to shed is mostly unauthenticated.
func (s *Server) rateLimit(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions || operational(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		if !s.rate.allow(s.bucket(r)) {
			s.opts.Metrics.throttle()
			w.Header().Set("Retry-After", "1")
			s.refuse(w, r, llm.Errorf(llm.KindRateLimit, "too many requests"))
			return
		}
		next.ServeHTTP(w, r)
	})
}

// gate holds a slot for as long as the request runs, and so belongs inside the
// key check: a slot is for work the proxy is going to do, and a request with no
// credentials is not going to reach any.
func (s *Server) gate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions || operational(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		if !s.inflight.enter() {
			s.opts.Metrics.refused()
			w.Header().Set("Retry-After", "1")
			s.refuse(w, r, llm.Errorf(llm.KindSaturated, "the proxy is at capacity"))
			return
		}
		defer s.inflight.leave()
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
