// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"slices"
	"sort"
	"strings"
	"testing"
)

// A field sent again is the same field, and its new value weighs what it
// weighs. Counting only the first sighting lets an upstream name one field and
// then fill it without limit.
func TestAFieldSentAgainIsWeighedAgain(t *testing.T) {
	var h Held
	var named int
	for h.weight+8 <= maxHeld {
		before := h.weight
		h.Keep(map[string]json.RawMessage{fmt.Sprintf("k%04d", named): json.RawMessage(`1`)})
		if h.weight == before {
			break
		}
		named++
	}
	if named == 0 {
		t.Fatal("nothing was held")
	}
	fat := json.RawMessage(`"` + strings.Repeat("x", 4096) + `"`)
	var said bool
	for i := range named {
		if h.Keep(map[string]json.RawMessage{fmt.Sprintf("k%04d", i): fat}) {
			said = true
		}
	}
	var weight int
	for k, v := range h.fields {
		weight += len(k) + len(v)
	}
	if weight > maxHeld {
		t.Errorf("held %d bytes against a ceiling of %d: a field named once was then filled for free", weight, maxHeld)
	}
	if !said {
		t.Error("what would not fit was dropped and nothing said so")
	}
}

// The ceiling is said once for the stream it applies to. Saying it again every
// time a chunk empties the hold is the shape of loss the ceiling exists to
// stop, arriving as diagnostics instead of as fields.
func TestTheHeldCeilingIsSaidOnceForTheWholeStream(t *testing.T) {
	fat := json.RawMessage(`"` + strings.Repeat("x", maxHeld+1) + `"`)
	var h Held
	var said int
	for i := range 2000 {
		if h.Keep(map[string]json.RawMessage{fmt.Sprintf("s%d", i): json.RawMessage(`1`)}) {
			said++
		}
		if h.Keep(map[string]json.RawMessage{fmt.Sprintf("big%d", i): fat}) {
			said++
		}
		h.Carry(nil)
	}
	if said != 1 {
		t.Errorf("the ceiling was said %d times, want once for the stream", said)
	}
}

// What the hold thinks it weighs has to be what it weighs, whatever order the
// fields arrive in and whatever replaces what.
func TestWhatTheHoldWeighsIsWhatItHolds(t *testing.T) {
	var h Held
	r := rand.New(rand.NewPCG(1, 2))
	for step := range 20000 {
		switch r.IntN(8) {
		case 0:
			h.Carry(nil)
		default:
			k := fmt.Sprintf("k%d", r.IntN(40))
			v := json.RawMessage(`"` + strings.Repeat("v", r.IntN(3000)) + `"`)
			h.Keep(map[string]json.RawMessage{k: v})
		}
		var weight int
		for k, v := range h.fields {
			weight += len(k) + len(v)
		}
		if weight != h.weight {
			t.Fatalf("step %d: the hold holds %d bytes and thinks it holds %d", step, weight, h.weight)
		}
		if weight > maxHeld {
			t.Fatalf("step %d: held %d bytes against a ceiling of %d", step, weight, maxHeld)
		}
	}
}

// What is kept when not all of it fits must be the same every time, or the
// same answer replayed gives two different bodies.
func TestWhatSurvivesTheCeilingIsTheSameEveryTime(t *testing.T) {
	frame := map[string]json.RawMessage{}
	for i := range 40 {
		frame[fmt.Sprintf("f%02d", i)] = json.RawMessage(`"` + strings.Repeat("x", 4096) + `"`)
	}
	var first []string
	for range 50 {
		var h Held
		if !h.Keep(frame) {
			t.Fatal("this frame does not fit and should have said so")
		}
		kept := make([]string, 0, len(h.fields))
		for k := range h.fields {
			kept = append(kept, k)
		}
		sort.Strings(kept)
		if first == nil {
			first = kept
			continue
		}
		if !slices.Equal(first, kept) {
			t.Fatalf("one run kept %v and another kept %v", first, kept)
		}
	}
}
