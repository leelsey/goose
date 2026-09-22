// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
)

type Metrics struct {
	mu sync.Mutex

	requests   map[requestKey]int64
	latency    map[string]duration
	attempts   map[attemptKey]int64
	tokens     map[tokenKey]int64
	unmeasured map[string]int64
	inflight   int64
	shed       int64
	throttled  int64
}

type requestKey struct{ format, status string }

type attemptKey struct{ provider, model, outcome string }

type tokenKey struct{ provider, model, kind string }

type duration struct {
	count int64
	total float64
}

func NewMetrics() *Metrics {
	return &Metrics{
		requests:   map[requestKey]int64{},
		latency:    map[string]duration{},
		attempts:   map[attemptKey]int64{},
		tokens:     map[tokenKey]int64{},
		unmeasured: map[string]int64{},
	}
}

func (m *Metrics) Record(_ context.Context, r usage.Record) error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	model := m.label(r.Model)
	m.attempts[attemptKey{r.Provider, model, string(r.Outcome)}]++
	if r.Sent && !r.Usage.Reported {
		m.unmeasured[r.Provider]++
	}
	for kind, n := range map[string]int{
		"prompt":      r.Usage.Prompt,
		"completion":  r.Usage.Completion,
		"cache_read":  r.Usage.CacheRead,
		"cache_write": r.Usage.CacheWrite,
		"reasoning":   r.Usage.Reasoning,
	} {
		if n > 0 {
			m.tokens[tokenKey{r.Provider, model, kind}] += int64(n)
		}
	}
	return nil
}

func (m *Metrics) request(format string, status int, took time.Duration) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.requests[requestKey{format, class(status)}]++
	d := m.latency[format]
	d.count++
	d.total += took.Seconds()
	m.latency[format] = d
}

func (m *Metrics) entered(delta int64) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.inflight += delta
	m.mu.Unlock()
}

// A model name can be whatever the client wrote, and every distinct one is a
// series kept for the life of the process and rendered into /metrics. Long ones
// are not model names, and past the cap the rest are counted together rather
// than each buying a row of their own. Called with m.mu held.
const (
	// Long enough for the longest real model id: a Bedrock inference-profile
	// ARN or a Vertex projects/.../models/... path, both of which run well past
	// a hundred characters.
	maxLabel  = 256
	maxSeries = 512
)

func (m *Metrics) label(model string) string {
	if len(model) > maxLabel {
		return "other"
	}
	if len(m.attempts) < maxSeries {
		return model
	}
	for k := range m.attempts {
		if k.model == model {
			return model
		}
	}
	return "other"
}

func (m *Metrics) throttle() {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.throttled++
	m.mu.Unlock()
}

func (m *Metrics) refused() {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.shed++
	m.mu.Unlock()
}

func class(status int) string {
	switch {
	case status >= 500:
		return "5xx"
	case status >= 400:
		return "4xx"
	case status >= 200:
		return "2xx"
	default:
		return "other"
	}
}

func (m *Metrics) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var b strings.Builder
		m.mu.Lock()
		defer m.mu.Unlock()

		write(&b, "goose_requests_total", "counter", "requests served, by client format and status class",
			flatten(m.requests, func(k requestKey) []string { return []string{"format", k.format, "status", k.status} }))
		write(&b, "goose_request_duration_seconds_count", "counter", "requests measured for latency",
			flatten(m.latency, func(k string) []string { return []string{"format", k} }, func(d duration) int64 { return d.count }))

		b.WriteString("# HELP goose_request_duration_seconds_sum total seconds spent serving, by client format\n")
		b.WriteString("# TYPE goose_request_duration_seconds_sum counter\n")
		for _, k := range sortedKeys(m.latency) {
			fmt.Fprintf(&b, "goose_request_duration_seconds_sum{format=%s} %s\n",
				quote(k), strconv.FormatFloat(m.latency[k].total, 'f', 6, 64))
		}

		write(&b, "goose_upstream_attempts_total", "counter", "upstream attempts, including the ones a chain discarded",
			flatten(m.attempts, func(k attemptKey) []string {
				return []string{"provider", k.provider, "model", k.model, "outcome", k.outcome}
			}))
		write(&b, "goose_tokens_total", "counter", "tokens attributed to the backend that served them",
			flatten(m.tokens, func(k tokenKey) []string {
				return []string{"provider", k.provider, "model", k.model, "kind", k.kind}
			}))
		write(&b, "goose_upstream_unmeasured_total", "counter", "delivered attempts the backend reported no usage for",
			flatten(m.unmeasured, func(k string) []string { return []string{"provider", k} }))

		b.WriteString("# HELP goose_inflight_requests requests being served right now\n")
		b.WriteString("# TYPE goose_inflight_requests gauge\n")
		fmt.Fprintf(&b, "goose_inflight_requests %d\n", m.inflight)
		b.WriteString("# HELP goose_shed_total requests refused because the proxy was at capacity\n")
		b.WriteString("# TYPE goose_shed_total counter\n")
		fmt.Fprintf(&b, "goose_shed_total %d\n", m.shed)
		b.WriteString("# HELP goose_throttled_total requests refused because one caller asked too often\n")
		b.WriteString("# TYPE goose_throttled_total counter\n")
		fmt.Fprintf(&b, "goose_throttled_total %d\n", m.throttled)

		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		_, _ = w.Write([]byte(b.String()))
	})
}

type series struct {
	labels []string
	value  int64
}

func write(b *strings.Builder, name, kind, help string, rows []series) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, kind)
	for _, s := range rows {
		b.WriteString(name)
		if len(s.labels) > 0 {
			b.WriteString("{")
			for i := 0; i < len(s.labels); i += 2 {
				if i > 0 {
					b.WriteString(",")
				}
				b.WriteString(s.labels[i])
				b.WriteString("=")
				b.WriteString(quote(s.labels[i+1]))
			}
			b.WriteString("}")
		}
		fmt.Fprintf(b, " %d\n", s.value)
	}
}

func flatten[K comparable, V any](m map[K]V, labels func(K) []string, value ...func(V) int64) []series {
	out := make([]series, 0, len(m))
	for k, v := range m {
		n := int64(0)
		if len(value) > 0 {
			n = value[0](v)
		} else if i, ok := any(v).(int64); ok {
			n = i
		}
		out = append(out, series{labels: labels(k), value: n})
	}
	sort.Slice(out, func(i, j int) bool {
		return strings.Join(out[i].labels, "\x00") < strings.Join(out[j].labels, "\x00")
	})
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func quote(v string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range v {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString("\\n")
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

var _ usage.Sink = (*Metrics)(nil)

func (s *Server) observe(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if operational(r.URL.Path) {
			next.ServeHTTP(w, r)
			return
		}
		started := s.client.Env().Clock()
		o := &observer{ResponseWriter: w}
		s.opts.Metrics.entered(1)
		defer s.opts.Metrics.entered(-1)
		next.ServeHTTP(o, r)
		status := o.status
		if status == 0 {
			status = http.StatusOK
		}
		s.opts.Metrics.request(s.formatFor(r.URL.Path), status, s.client.Env().Clock().Sub(started))
	})
}

func (s *Server) formatFor(path string) string {
	if codec := s.codecFor(path); codec != nil {
		return codec.Name()
	}
	return "other"
}

func (s *Server) codecFor(path string) wire.Codec {
	if codec, ok := s.byPath[path]; ok {
		return codec
	}
	best := ""
	var found wire.Codec
	for prefix, codec := range s.byPrefix {
		if strings.HasPrefix(path, prefix) && len(prefix) > len(best) {
			best, found = prefix, codec
		}
	}
	return found
}
