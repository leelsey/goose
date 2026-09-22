// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"sort"
)

// A frame that puts nothing on a wire can still carry fields that wire has no
// place for, and they wait for the next chunk that can carry them. How many
// such frames arrive, and how much each brings, is the upstream's to choose, so
// what waits is bounded. Nothing an upstream actually sends is near this.
const maxHeld = 64 << 10

// Held is what a codec is carrying forward from frames that wrote nothing.
//
// Fields still waiting when a stream ends are not reported. It was tried twice
// and neither mark was worth what it cost. Remembering a field by name alone
// misses the case the report was written for, a field that rode every chunk as
// an identifier and arrives on the last one carrying the payload. Remembering
// it by name and value catches that, and also calls every field a backend
// changes between chunks a loss, which on a stream that ends with a chunk
// carrying no content — the shape goose asks for whenever it asks for usage —
// is most of them. Reading every value on every chunk to find that out cost
// twenty seconds a stream where it had cost microseconds.
//
// If the signal is ever wanted back, it is the first case that can be reported
// without ambiguity: a name that rode no chunk at all, marked in a fixed size.
// The second cannot be, and nothing cheap recovers it.
type Held struct {
	fields map[string]json.RawMessage
	weight int
	full   bool
}

// Keep takes a frame's unmodelled fields. Two frames that each carry some are
// two sets of them, not one replacing the other, and the map is the codec's own
// so a frame's is never written through. It reports true the first time
// something did not fit, so a loss is said once rather than once a frame.
func (h *Held) Keep(extra map[string]json.RawMessage) (overfull bool) {
	// Taken in order, so that which of them fits is the same answer twice: a
	// map hands them over in whatever order it likes, and the same reply
	// replayed would then come out differently.
	names := make([]string, 0, len(extra))
	for k := range extra {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		v := extra[k]
		// A field sent again is the same field, and its new value weighs what
		// it weighs: counting only the first sighting would let an upstream
		// name a field cheaply and then fill it for nothing. One that will not
		// fit leaves the value already held, which did.
		weight := h.weight + len(k) + len(v)
		if old, known := h.fields[k]; known {
			weight -= len(k) + len(old)
		}
		if weight > maxHeld {
			if !h.full {
				h.full = true
				overfull = true
			}
			continue
		}
		if h.fields == nil {
			h.fields = make(map[string]json.RawMessage, len(extra))
		}
		h.weight = weight
		h.fields[k] = v
	}
	return overfull
}

// Carry merges what has waited into the fields of the chunk now being written
// and empties the hold. The chunk's own win, since they describe it. Having
// been full is not emptied with the rest: the ceiling belongs to the stream, and
// saying it again for every chunk that empties the hold would be the same shape
// of loss it is there to stop, arriving as diagnostics rather than as fields.
func (h *Held) Carry(own map[string]json.RawMessage) map[string]json.RawMessage {
	if len(h.fields) == 0 {
		return own
	}
	merged := make(map[string]json.RawMessage, len(h.fields)+len(own))
	for k, v := range h.fields {
		merged[k] = v
	}
	for k, v := range own {
		merged[k] = v
	}
	h.fields, h.weight = nil, 0
	return merged
}
