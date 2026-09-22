// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"fmt"
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
