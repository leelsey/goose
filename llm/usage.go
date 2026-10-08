// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

type Usage struct {
	Prompt     int `json:"prompt"`
	Completion int `json:"completion"`
	Total      int `json:"total"`

	CacheRead    int `json:"cache_read,omitempty"`
	CacheWrite   int `json:"cache_write,omitempty"`
	CacheWrite5m int `json:"cache_write_5m,omitempty"`
	CacheWrite1h int `json:"cache_write_1h,omitempty"`
	Reasoning    int `json:"reasoning,omitempty"`

	Reported bool `json:"reported"`

	Cost  *Cost  `json:"cost,omitempty"`
	Tier  string `json:"tier,omitempty"`
	Speed string `json:"speed,omitempty"`

	Turns      int   `json:"turns,omitempty"`
	DurationMS int64 `json:"duration_ms,omitempty"`

	Anomalies []Anomaly `json:"anomalies,omitempty"`
}

type Cost struct {
	Amount   float64 `json:"amount"`
	Currency string  `json:"currency"`
}

type Anomaly struct {
	Code   string `json:"code"`
	Detail string `json:"detail,omitempty"`
}

const (
	AnomalyInputUnknown  = "input_unknown"
	AnomalyOutputUnknown = "output_unknown"
	AnomalyCacheSplit    = "cache_split_unreconcilable"
	AnomalyNotMeasurable = "backend_reports_no_usage"
	AnomalyViewsDisagree = "views_disagree"
	AnomalyCurrencyMixed = "cost_currency_mixed"
	AnomalyPriceMissing  = "price_missing"
)

type RawUsage struct {
	Prompt       int
	Completion   int
	Reasoning    int
	CacheRead    int
	CacheWrite   int
	CacheWrite5m int
	CacheWrite1h int
	ToolPrompt   int

	CacheOutsidePrompt         bool
	ReasoningOutsideCompletion bool
	ToolPromptOutsidePrompt    bool

	Reported bool
	Cost     *Cost
	Tier     string
	Speed    string
	Turns    int
}

func (r RawUsage) Normalise() Usage {
	u := Usage{
		Prompt:       r.Prompt,
		Completion:   r.Completion,
		Reasoning:    r.Reasoning,
		CacheRead:    r.CacheRead,
		CacheWrite:   r.CacheWrite,
		CacheWrite5m: r.CacheWrite5m,
		CacheWrite1h: r.CacheWrite1h,
		Reported:     r.Reported,
		Cost:         r.Cost,
		Tier:         r.Tier,
		Speed:        r.Speed,
		Turns:        r.Turns,
	}
	if r.CacheOutsidePrompt {
		u.Prompt += r.CacheRead + r.CacheWrite
	}
	if r.ToolPromptOutsidePrompt {
		u.Prompt += r.ToolPrompt
	}
	if r.ReasoningOutsideCompletion {
		u.Completion += r.Reasoning
	}
	return u.settle()
}

func (u Usage) settle() Usage {
	if u.CacheWrite5m > u.CacheWrite {
		u.CacheWrite5m = u.CacheWrite
		u = u.WithAnomaly(Anomaly{Code: AnomalyCacheSplit})
	}
	if u.CacheWrite1h > u.CacheWrite-u.CacheWrite5m {
		u.CacheWrite1h = u.CacheWrite - u.CacheWrite5m
		u = u.WithAnomaly(Anomaly{Code: AnomalyCacheSplit})
	}
	if u.Prompt < u.CacheRead+u.CacheWrite {
		u = u.WithAnomaly(Anomaly{Code: AnomalyInputUnknown})
	}
	if u.Reasoning > u.Completion {
		u = u.WithAnomaly(Anomaly{Code: AnomalyOutputUnknown})
	}
	u.Total = u.Prompt + u.Completion
	return u
}

func (u Usage) Unmeasured() Usage {
	if u.Reported || !u.counted() {
		return u
	}
	return u.WithAnomaly(Anomaly{Code: AnomalyNotMeasurable})
}

func (u Usage) Unpriced(prices bool) Usage {
	if !prices || u.Cost != nil {
		return u
	}
	return u.WithAnomaly(Anomaly{
		Code:   AnomalyPriceMissing,
		Detail: "this backend prices its own calls and priced neither this one nor part of it",
	})
}

const (
	FigureCacheRead  = "cache_read"
	FigureCacheWrite = "cache_write"
	FigureReasoning  = "reasoning"
	FigureCost       = "cost"
	FigureTier       = "tier"
)

func (u Usage) Uncarried(format string, carried ...string) Diagnostics {
	held := map[string]bool{}
	for _, name := range carried {
		held[name] = true
	}
	set := []struct {
		name string
		on   bool
	}{
		{FigureCacheRead, u.CacheRead != 0},
		{FigureCacheWrite, u.CacheWrite != 0},
		{FigureReasoning, u.Reasoning != 0},
		{FigureCost, u.Cost != nil},
		{FigureTier, u.Tier != ""},
	}
	var out Diagnostics
	for _, f := range set {
		if !f.on || held[f.name] {
			continue
		}
		out = append(out, Diagnostic{
			Code: AcctFigureUncarried, Class: ClassAccounting, Stage: StageEncode,
			Severity: SeverityInfo, Format: format, Field: f.name,
			Detail: "this wire has no field for this figure, so the client's copy of the bill is short of it",
		})
	}
	return out
}

func disagree(a, b Usage) bool {
	if a.Empty() || b.Empty() {
		return false
	}
	for _, pair := range [][2]int{
		{a.Prompt, b.Prompt},
		{a.Completion, b.Completion},
		{a.CacheRead, b.CacheRead},
		{a.CacheWrite, b.CacheWrite},
		{a.Reasoning, b.Reasoning},
	} {
		if pair[0] != 0 && pair[1] != 0 && pair[0] != pair[1] {
			return true
		}
	}
	return false
}

func (u Usage) Uncached() (int, bool) {
	n := u.Prompt - u.CacheRead - u.CacheWrite
	if n < 0 {
		return 0, false
	}
	return n, true
}

// Empty reports a usage carrying nothing at all, metadata included.
func (u Usage) Empty() bool {
	return u.counted() && u.Turns == 0 && u.DurationMS == 0
}

// counted reports a usage nobody put a number on. How long a call took and how
// many turns it ran are worth keeping, but neither says what it cost, so
// neither answers for the columns that do.
func (u Usage) counted() bool {
	return u.Prompt == 0 && u.Completion == 0 && u.CacheRead == 0 && u.CacheWrite == 0 &&
		u.Reasoning == 0 && u.Cost == nil && !u.Reported
}

func (u Usage) Merge(next Usage) Usage {
	u.Prompt = newest(u.Prompt, next.Prompt)
	u.Completion = newest(u.Completion, next.Completion)
	u.CacheRead = newest(u.CacheRead, next.CacheRead)
	u.CacheWrite = newest(u.CacheWrite, next.CacheWrite)
	u.CacheWrite5m = newest(u.CacheWrite5m, next.CacheWrite5m)
	u.CacheWrite1h = newest(u.CacheWrite1h, next.CacheWrite1h)
	u.Reasoning = newest(u.Reasoning, next.Reasoning)
	u.Turns = newest(u.Turns, next.Turns)
	u.DurationMS = newest64(u.DurationMS, next.DurationMS)
	u.Reported = u.Reported || next.Reported
	if next.Cost != nil {
		c := *next.Cost
		u.Cost = &c
	}
	if next.Tier != "" {
		u.Tier = next.Tier
	}
	if next.Speed != "" {
		u.Speed = next.Speed
	}
	u.Anomalies = mergeAnomalies(u.Anomalies, next.Anomalies)
	return u.settle()
}

func (u Usage) Add(next Usage) Usage {
	// One turn priced and one not is not a priced total, whichever way round
	// they were added: the sum would read as the whole bill. An accumulator
	// nobody has added to yet is not a turn that went free, so this is asked
	// before the two are joined and while both are still in view.
	partlyPriced := (u.Cost == nil) != (next.Cost == nil) && !u.Empty() && !next.Empty()
	u.Prompt += next.Prompt
	u.Completion += next.Completion
	u.CacheRead += next.CacheRead
	u.CacheWrite += next.CacheWrite
	u.CacheWrite5m += next.CacheWrite5m
	u.CacheWrite1h += next.CacheWrite1h
	u.Reasoning += next.Reasoning
	u.Turns += next.Turns
	u.DurationMS += next.DurationMS
	u.Reported = u.Reported || next.Reported
	if partlyPriced {
		u.Anomalies = appendAnomaly(u.Anomalies, Anomaly{
			Code:   AnomalyPriceMissing,
			Detail: "one of the turns in this total carried no price",
		})
	}
	u.Cost, u.Anomalies = addCost(u.Cost, next.Cost, u.Anomalies)
	if next.Tier != "" {
		u.Tier = next.Tier
	}
	if next.Speed != "" {
		u.Speed = next.Speed
	}
	u.Anomalies = mergeAnomalies(u.Anomalies, next.Anomalies)
	return u.settle()
}

func (u Usage) Cover(other Usage) Usage {
	if disagree(u, other) {
		u = u.WithAnomaly(Anomaly{
			Code:   AnomalyViewsDisagree,
			Detail: "the turns and the backend's own summary reported different figures",
		})
	}
	u.Prompt = max(u.Prompt, other.Prompt)
	u.Completion = max(u.Completion, other.Completion)
	u.CacheRead = max(u.CacheRead, other.CacheRead)
	u.CacheWrite = max(u.CacheWrite, other.CacheWrite)
	u.CacheWrite5m = max(u.CacheWrite5m, other.CacheWrite5m)
	u.CacheWrite1h = max(u.CacheWrite1h, other.CacheWrite1h)
	u.Reasoning = max(u.Reasoning, other.Reasoning)
	u.Turns = max(u.Turns, other.Turns)
	u.DurationMS = max(u.DurationMS, other.DurationMS)
	u.Reported = u.Reported || other.Reported
	switch {
	case other.Cost == nil:
	case u.Cost == nil:
		c := *other.Cost
		u.Cost = &c
	case u.Cost.Currency != other.Cost.Currency && u.Cost.Currency != "" && other.Cost.Currency != "":
		// Two prices in two currencies do not compare, so the larger number is
		// not the larger cost and neither may stand for the other.
		u.Anomalies = appendAnomaly(u.Anomalies,
			Anomaly{Code: AnomalyCurrencyMixed, Detail: u.Cost.Currency + "+" + other.Cost.Currency})
	case other.Cost.Amount > u.Cost.Amount:
		c := *other.Cost
		u.Cost = &c
	}
	if u.Tier == "" {
		u.Tier = other.Tier
	}
	if u.Speed == "" {
		u.Speed = other.Speed
	}
	u.Anomalies = mergeAnomalies(u.Anomalies, other.Anomalies)
	return u.settle()
}

func (u Usage) WithAnomaly(a Anomaly) Usage {
	u.Anomalies = appendAnomaly(u.Anomalies, a)
	return u
}

func newest(cur, next int) int {
	if next != 0 {
		return next
	}
	return cur
}

func newest64(cur, next int64) int64 {
	if next != 0 {
		return next
	}
	return cur
}

func addCost(a, b *Cost, an []Anomaly) (*Cost, []Anomaly) {
	switch {
	case b == nil:
		return a, an
	case a == nil:
		c := *b
		return &c, an
	case a.Currency != b.Currency && a.Currency != "" && b.Currency != "":
		return a, appendAnomaly(an, Anomaly{Code: AnomalyCurrencyMixed, Detail: a.Currency + "+" + b.Currency})
	default:
		c := Cost{Amount: a.Amount + b.Amount, Currency: a.Currency}
		if c.Currency == "" {
			c.Currency = b.Currency
		}
		return &c, an
	}
}

// appendAnomaly never writes into the slice it was handed. Usage is a value
// copied all over, and two copies each noticing something would otherwise write
// into the same spare capacity and overwrite one another.
func appendAnomaly(list []Anomaly, a Anomaly) []Anomaly {
	for _, x := range list {
		if x.Code == a.Code {
			return list
		}
	}
	out := make([]Anomaly, len(list), len(list)+1)
	copy(out, list)
	return append(out, a)
}

func mergeAnomalies(a, b []Anomaly) []Anomaly {
	for _, x := range b {
		a = appendAnomaly(a, x)
	}
	return a
}
