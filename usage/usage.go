// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package usage carries what a call cost to the caller's own accounts.
package usage

import (
	"context"
	"encoding/json"
	"fmt"
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

// Every row holds the upstream's own words, so a long-running proxy that kept
// them all would grow without end. The counts stay exact; only the rows
// themselves are capped, newest kept.
const maxRecords = 10000

func (m *Memory) Record(_ context.Context, r Record) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.records = append(m.records, r)
	if len(m.records) > 2*maxRecords {
		m.records = append(m.records[:0], m.records[len(m.records)-maxRecords:]...)
	}
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

// Record writes the row to every sink at once: a call waits for the slowest
// ledger rather than for all of them in turn. The error reported is the first
// in configured order, not whichever failed soonest.
// only writes one row to one sink. A single sink is someone else's code as much
// as several are, and a panic in it must fail the row rather than the process.
func only(ctx context.Context, s Sink, r Record) (err error) {
	defer func() {
		if p := recover(); p != nil {
			err = fmt.Errorf("usage sink panicked: %v", p)
		}
	}()
	return s.Record(ctx, r)
}

func (m Multi) Record(ctx context.Context, r Record) error {
	if len(m) == 1 && m[0] != nil {
		return only(ctx, m[0], r)
	}
	errs := make([]error, len(m))
	var wg sync.WaitGroup
	for i, s := range m {
		if s == nil {
			continue
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			// A sink is someone else's code; on its own goroutine a panic would
			// take the process down rather than failing this one row.
			defer func() {
				if p := recover(); p != nil {
					errs[i] = fmt.Errorf("usage sink panicked: %v", p)
				}
			}()
			errs[i] = s.Record(ctx, r)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
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
