// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import "testing"

func TestNormaliseCacheOutsidePrompt(t *testing.T) {
	u := RawUsage{Prompt: 12, Completion: 40, CacheRead: 9100, CacheWrite: 33096, CacheOutsidePrompt: true, Reported: true}.Normalise()
	if u.Prompt != 12+9100+33096 {
		t.Fatalf("prompt = %d, want %d", u.Prompt, 12+9100+33096)
	}
	if n, ok := u.Uncached(); !ok || n != 12 {
		t.Fatalf("uncached = %d,%v, want 12,true", n, ok)
	}
	if u.Total != u.Prompt+u.Completion {
		t.Fatalf("total = %d", u.Total)
	}
}

func TestNormaliseCacheInsidePrompt(t *testing.T) {
	u := RawUsage{Prompt: 1000, Completion: 20, CacheRead: 800}.Normalise()
	if u.Prompt != 1000 {
		t.Fatalf("prompt = %d, want 1000 (cache already inside)", u.Prompt)
	}
}

func TestNormaliseReasoningOutsideCompletion(t *testing.T) {
	u := RawUsage{Prompt: 10, Completion: 30, Reasoning: 70, ReasoningOutsideCompletion: true}.Normalise()
	if u.Completion != 100 {
		t.Fatalf("completion = %d, want 100", u.Completion)
	}
	if u.Reasoning != 70 {
		t.Fatalf("reasoning = %d, want 70", u.Reasoning)
	}
}

func TestSettleClampsFiveMinuteWrite(t *testing.T) {
	u := RawUsage{Prompt: 10, CacheWrite: 0, CacheWrite5m: 9000}.Normalise()
	if u.CacheWrite5m != 0 {
		t.Fatalf("cache_write_5m = %d, want 0", u.CacheWrite5m)
	}
	if !hasAnomaly(u, AnomalyCacheSplit) {
		t.Fatal("clamped split must be recorded, not silently corrected")
	}
}

func TestSettleMarksImpossiblePromptBreakdown(t *testing.T) {
	u := Usage{Prompt: 5, CacheRead: 10}.settle()
	if _, ok := u.Uncached(); ok {
		t.Fatal("uncached must report unknown rather than a clamped zero")
	}
	if !hasAnomaly(u, AnomalyInputUnknown) {
		t.Fatal("anomaly missing")
	}
}

func TestMergeIsComplementaryNotCumulative(t *testing.T) {
	start := Usage{Prompt: 33108, CacheRead: 9100, CacheWrite: 24000, Reported: true}
	delta := Usage{Prompt: 0, Completion: 480, Reported: true}
	got := start.Merge(delta)
	if got.Prompt != 33108 {
		t.Fatalf("prompt = %d, want 33108 (a later frame must not zero it)", got.Prompt)
	}
	if got.CacheWrite != 24000 {
		t.Fatalf("cache_write = %d, want 24000", got.CacheWrite)
	}
	if got.Completion != 480 {
		t.Fatalf("completion = %d, want 480", got.Completion)
	}
	if got.Total != 33108+480 {
		t.Fatalf("total = %d", got.Total)
	}
}

func TestMergeKeepsReportedOnce(t *testing.T) {
	got := Usage{Reported: true}.Merge(Usage{Completion: 3})
	if !got.Reported {
		t.Fatal("Reported must be OR-ed across frames")
	}
}

func TestMergeTakesLateTierAndSpeed(t *testing.T) {
	got := Usage{Prompt: 10}.Merge(Usage{Completion: 5, Tier: "priority", Speed: "fast"})
	if got.Tier != "priority" || got.Speed != "fast" {
		t.Fatalf("tier/speed = %q/%q", got.Tier, got.Speed)
	}
}

func TestAddSumsTurns(t *testing.T) {
	a := Usage{Prompt: 100, Completion: 20, Turns: 1, Reported: true, Cost: &Cost{Amount: 0.01, Currency: "USD"}}
	b := Usage{Prompt: 140, Completion: 30, Turns: 1, Reported: true, Cost: &Cost{Amount: 0.02, Currency: "USD"}}
	got := a.Add(b)
	if got.Prompt != 240 || got.Completion != 50 {
		t.Fatalf("counts = %d/%d", got.Prompt, got.Completion)
	}
	if got.Cost == nil || got.Cost.Amount < 0.0299 || got.Cost.Amount > 0.0301 {
		t.Fatalf("cost = %+v, want 0.03", got.Cost)
	}
	if got.Turns != 2 {
		t.Fatalf("turns = %d, want 2", got.Turns)
	}
}

func TestAddKeepsLaterTier(t *testing.T) {
	got := Usage{Prompt: 1, Tier: ""}.Add(Usage{Prompt: 1, Tier: "batch"})
	if got.Tier != "batch" {
		t.Fatalf("tier = %q", got.Tier)
	}
}

func TestCoverTakesComponentMaximum(t *testing.T) {
	summary := Usage{Prompt: 100, Completion: 480}
	iterations := Usage{Prompt: 340, Completion: 12}
	got := summary.Cover(iterations)
	if got.Prompt != 340 || got.Completion != 480 {
		t.Fatalf("cover = %d/%d, want 340/480", got.Prompt, got.Completion)
	}
}

func TestEmptySeparatesZeroFromSilence(t *testing.T) {
	if !(Usage{}).Empty() {
		t.Fatal("zero usage is empty")
	}
	if (Usage{Reported: true}).Empty() {
		t.Fatal("a backend that reported zero is not empty")
	}
}

func hasAnomaly(u Usage, code string) bool {
	for _, a := range u.Anomalies {
		if a.Code == code {
			return true
		}
	}
	return false
}

func TestAddRecordsMixedCurrency(t *testing.T) {
	a := Usage{Prompt: 1, Cost: &Cost{Amount: 1, Currency: "USD"}}
	b := Usage{Prompt: 1, Cost: &Cost{Amount: 2, Currency: "GBP"}}
	got := a.Add(b)
	if got.Cost.Amount != 1 || got.Cost.Currency != "USD" {
		t.Fatalf("cost = %+v: currencies must not be added", got.Cost)
	}
	if !hasAnomaly(got, AnomalyCurrencyMixed) {
		t.Fatal("a total that is short must say so")
	}
}

func TestCoverDoesNotAliasCost(t *testing.T) {
	other := Usage{Cost: &Cost{Amount: 5, Currency: "USD"}}
	got := Usage{}.Cover(other)
	got.Cost.Amount = 99
	if other.Cost.Amount != 5 {
		t.Fatal("cover aliased the source cost")
	}
}

func TestAFigureTheWireCannotCarryIsReported(t *testing.T) {
	u := Usage{Prompt: 100, CacheRead: 40, CacheWrite: 10, Completion: 20, Reasoning: 5}
	u.Cost = &Cost{Amount: 0.01, Currency: "USD"}

	all := u.Uncarried("ollama")
	if len(all) != 4 {
		t.Fatalf("uncarried = %+v: this wire has a field for none of them", all)
	}
	for _, d := range all {
		if d.Code != AcctFigureUncarried || d.Class != ClassAccounting || d.Severity != SeverityInfo {
			t.Fatalf("diagnostic = %+v", d)
		}
	}

	some := u.Uncarried("anthropic", FigureCacheRead, FigureCacheWrite)
	fields := []string{}
	for _, d := range some {
		fields = append(fields, d.Field)
	}
	if len(fields) != 2 || fields[0] != FigureReasoning || fields[1] != FigureCost {
		t.Fatalf("fields = %v: a column the wire wrote is not a loss", fields)
	}

	if got := (Usage{Prompt: 1, Completion: 1}).Uncarried("ollama"); len(got) != 0 {
		t.Fatalf("uncarried = %+v: nothing was measured to lose", got)
	}
}
