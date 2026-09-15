// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package usage carries what a call cost to the caller's own accounts.
package usage

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/leelsey/goose/llm"
)

type Record struct {
	Time      time.Time        `json:"time"`
	Route     string           `json:"route,omitempty"`
	Provider  string           `json:"provider"`
	Model     string           `json:"model,omitempty"`
	Requested string           `json:"requested,omitempty"`
	Format    string           `json:"format,omitempty"`
	Stream    bool             `json:"stream,omitempty"`
	Index     int              `json:"index"`
	Sent      bool             `json:"sent"`
	Outcome   llm.Outcome      `json:"outcome"`
	Elapsed   int64            `json:"elapsed_ms"`
	Usage     llm.Usage        `json:"usage"`
	Err       string           `json:"err,omitempty"`
	Detail    string           `json:"detail,omitempty"`
	Status    int              `json:"status,omitempty"`
	Diags     []llm.Diagnostic `json:"diagnostics,omitempty"`
}

type Sink interface {
	Record(ctx context.Context, r Record) error
}

type SinkFunc func(context.Context, Record) error

func (f SinkFunc) Record(ctx context.Context, r Record) error { return f(ctx, r) }

type Memory struct {
	mu         sync.Mutex
	records    []Record
	total      llm.Usage
	calls      int
	unmeasured int
}

func NewMemory() *Memory { return &Memory{} }

func (m *Memory) Record(_ context.Context, r Record) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.records = append(m.records, r)
	m.calls++
	if r.Sent && !r.Usage.Reported {
		m.unmeasured++
	}
	m.total = m.total.Add(r.Usage)
	return nil
}

func (m *Memory) Records() []Record {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]Record(nil), m.records...)
}

func (m *Memory) Total() llm.Usage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.total
}

func (m *Memory) Stats() (attempts, unmeasured int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls, m.unmeasured
}

func (m *Memory) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.records, m.total, m.calls, m.unmeasured = nil, llm.Usage{}, 0, 0
}

type JSONL struct {
	mu  sync.Mutex
	w   io.Writer
	enc *json.Encoder
}

func NewJSONL(w io.Writer) *JSONL {
	return &JSONL{w: w, enc: json.NewEncoder(w)}
}

func (j *JSONL) Record(_ context.Context, r Record) error {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.enc.Encode(r)
}

type Multi []Sink

func (m Multi) Record(ctx context.Context, r Record) error {
	var first error
	for _, s := range m {
		if s == nil {
			continue
		}
		if err := s.Record(ctx, r); err != nil && first == nil {
			first = err
		}
	}
	return first
}

func FromAttempt(a llm.Attempted, now time.Time, format string, stream bool) Record {
	when := a.Started
	if when.IsZero() {
		when = now
	}
	return Record{
		Time: when, Route: a.Route, Provider: a.Provider, Model: a.Model,
		Requested: a.Requested, Format: format, Stream: stream, Index: a.Index,
		Sent: a.Sent, Outcome: a.Outcome, Elapsed: a.Elapsed, Usage: a.Usage,
		Err: a.Err, Detail: a.Detail, Status: a.Status, Diags: a.Diagnostics,
	}
}
