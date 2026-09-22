// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import "github.com/leelsey/goose/llm"

// ToolMerger folds streamed tool-call fragments back into whole calls.
//
// A fragment names its call with Index wherever the wire carries one. Where it
// does not, a fragment holding neither a fresh id nor a name continues the call
// already open, because argument bytes arrive on their own; anything else opens
// the next call.
// Arguments are accumulated until the turn ends, so an upstream that streams
// fragments without ever stopping would be held in full. This is the ceiling
// for one turn's worth across every call in it; a codec that keeps its own
// budget passes one to Merge instead.
const DefaultToolArgsBudget = 8 << 20

type ToolMerger struct {
	byKey   map[int]*llm.ToolUse
	order   []*llm.ToolUse
	current int
	next    int
	left    int
	over    bool
	shared  *ToolMerger
}

func NewToolMerger() *ToolMerger {
	return &ToolMerger{byKey: map[int]*llm.ToolUse{}, current: -1, left: DefaultToolArgsBudget}
}

// Share makes this merger spend another's budget, so that a wire keeping one
// merger per candidate still holds one turn's worth in total rather than that
// much for every candidate the upstream cares to name.
func (m *ToolMerger) Share(with *ToolMerger) {
	if with == nil {
		return
	}
	m.shared = with.purse()
}

func (m *ToolMerger) purse() *ToolMerger {
	if m.shared != nil {
		return m.shared
	}
	return m
}

// Overflowed reports whether argument bytes were dropped for want of budget,
// so that a codec can say so rather than emitting a call that lost its tail.
func (m *ToolMerger) Overflowed() bool { return m.purse().over }

func (m *ToolMerger) afford(n int) bool {
	p := m.purse()
	if n > p.left {
		p.over = true
		return false
	}
	p.left -= n
	return true
}

// Merge folds a fragment into its call and returns the whole call so far. keep
// decides whether the fragment's argument bytes fit a budget; a nil keep takes
// them all.
func (m *ToolMerger) Merge(v *llm.ToolUse, keep func(int) bool) *llm.ToolUse {
	if m.byKey == nil {
		m.byKey, m.current, m.left = map[int]*llm.ToolUse{}, -1, DefaultToolArgsBudget
	}
	idx := m.key(v)
	m.current = idx
	t, open := m.byKey[idx]
	if !open {
		t = &llm.ToolUse{ID: v.ID, Name: v.Name}
		m.byKey[idx] = t
		m.order = append(m.order, t)
	}
	if v.ID != "" {
		t.ID = v.ID
	}
	if v.Name != "" {
		t.Name = v.Name
	}
	if len(v.Args) > 0 {
		afford := keep
		if afford == nil {
			afford = m.afford
		}
		if afford(len(v.Args)) {
			t.Args = append(t.Args, v.Args...)
		} else {
			// Whichever budget refused, the call now ends short of where the
			// upstream left it, which is what Overflowed is asked about.
			m.purse().over = true
		}
	}
	return t
}

func (m *ToolMerger) key(v *llm.ToolUse) int {
	switch {
	case v.Index != nil:
		if *v.Index >= m.next {
			m.next = *v.Index + 1
		}
		return *v.Index
	case v.ID != "" && m.current >= 0 && m.byKey[m.current] != nil && m.byKey[m.current].ID == v.ID:
		return m.current
	case v.ID == "" && v.Name == "" && m.current >= 0:
		return m.current
	}
	m.next++
	return m.next - 1
}

// Tools returns the calls in the order they were opened.
func (m *ToolMerger) Tools() []*llm.ToolUse { return m.order }

// Reset drops what has been folded, for a wire that closes one turn and opens
// another on the same stream.
func (m *ToolMerger) Reset() {
	m.byKey, m.order, m.current, m.next = map[int]*llm.ToolUse{}, nil, -1, 0
	// Only a budget of our own is refilled: one candidate's turn ending is not a
	// fresh allowance for every candidate sharing one.
	m.left, m.over = DefaultToolArgsBudget, false
}
