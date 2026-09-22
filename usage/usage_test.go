// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package usage

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

func rec(provider string, u llm.Usage, sent bool) Record {
	return Record{Time: time.Unix(0, 0), Provider: provider, Usage: u, Sent: sent, Outcome: llm.OutcomeOK}
}

func TestMemoryTotalsAcrossAttempts(t *testing.T) {
	m := NewMemory()
	ctx := context.Background()
	_ = m.Record(ctx, rec("a", llm.Usage{Prompt: 100, Completion: 10, Turns: 1, Reported: true}, true))
	_ = m.Record(ctx, rec("b", llm.Usage{Prompt: 140, Completion: 20, Turns: 1, Reported: true}, true))
	total := m.Total()
	if total.Prompt != 240 || total.Completion != 30 || total.Turns != 2 {
		t.Fatalf("total = %+v", total)
	}
	if n, unmeasured := m.Stats(); n != 2 || unmeasured != 0 {
		t.Fatalf("stats = %d/%d", n, unmeasured)
	}
}

func TestMemoryCountsWhatCouldNotBeMeasured(t *testing.T) {
	m := NewMemory()
	_ = m.Record(context.Background(), rec("a", llm.Usage{}, true))
	if _, unmeasured := m.Stats(); unmeasured != 1 {
		t.Fatal("a backend that reported nothing is a blind call, not a free one")
	}
}

func TestJSONLKeepsEveryAccountingField(t *testing.T) {
	var buf bytes.Buffer
	j := NewJSONL(&buf)
	u := llm.Usage{
		Prompt: 100, Completion: 10, CacheRead: 40, CacheWrite: 20, CacheWrite5m: 20,
		Reasoning: 5, Reported: true, Tier: "priority", Speed: "fast", Turns: 2,
		Cost: &llm.Cost{Amount: 0.02, Currency: "USD"},
	}
	if err := j.Record(context.Background(), rec("a", u, true)); err != nil {
		t.Fatal(err)
	}
	var back Record
	if err := json.Unmarshal(buf.Bytes(), &back); err != nil {
		t.Fatal(err)
	}
	if back.Usage.Cost == nil || back.Usage.Cost.Amount != 0.02 {
		t.Fatalf("cost lost across the hop: %+v", back.Usage)
	}
	if back.Usage.Tier != "priority" || back.Usage.Speed != "fast" || !back.Usage.Reported {
		t.Fatalf("usage = %+v: a paid call must not book as unmeasured downstream", back.Usage)
	}
	if !strings.HasSuffix(buf.String(), "\n") {
		t.Fatal("a ledger line must be newline terminated")
	}
}

func TestMultiFansOutAndKeepsGoing(t *testing.T) {
	var seen int
	failing := SinkFunc(func(context.Context, Record) error { return context.Canceled })
	counting := SinkFunc(func(context.Context, Record) error { seen++; return nil })
	err := Multi{failing, counting}.Record(context.Background(), rec("a", llm.Usage{}, true))
	if err == nil {
		t.Fatal("the first failure must be reported")
	}
	if seen != 1 {
		t.Fatal("a failing sink must not stop the others")
	}
}

// A sink is someone else's code whether there is one of it or several, and a
// panic in it fails the row rather than the process.
func TestASinglePanickingSinkFailsTheRowAndNotTheProcess(t *testing.T) {
	m := Multi{SinkFunc(func(context.Context, Record) error { panic("the ledger fell over") })}
	err := m.Record(context.Background(), Record{Provider: "one"})
	if err == nil {
		t.Fatal("a sink that panicked was reported as having written the row")
	}
	if !strings.Contains(err.Error(), "panicked") {
		t.Errorf("err = %v, want it to say what happened", err)
	}
}
