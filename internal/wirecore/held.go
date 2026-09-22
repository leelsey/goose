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
type Held struct {
	fields  map[string]json.RawMessage
	carried map[string]bool
	weight  int
	full    bool
	unsure  bool
}

// A field name is remembered so that one arriving again after the last chunk
// can be told from one arriving for the first time. How many distinct names a
// stream uses is the upstream's to choose, so past this many the question is
// answered the quiet way: an upstream that names this many fields is not one
// whose last frame is worth crying loss over.
const maxNames = 256

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

// Waiting reports fields still held with no chunk left to carry them, which is
// a loss like any other and not one to go quietly. A field that already rode a
// chunk of this answer is not one: the last frame of a stream routinely repeats
// what every chunk before it carried, and calling that a loss would call nearly
// every stream one.
func (h *Held) Waiting() bool {
	if h.unsure {
		return false
	}
	for k := range h.fields {
		if !h.carried[k] {
			return true
		}
	}
	return false
}

// Carry merges what has waited into the fields of the chunk now being written
// and empties the hold. The chunk's own win, since they describe it. Having
// been full is not emptied with the rest: the ceiling belongs to the stream, and
// saying it again for every chunk that empties the hold would be the same shape
// of loss it is there to stop, arriving as diagnostics rather than as fields.
func (h *Held) Carry(own map[string]json.RawMessage) map[string]json.RawMessage {
	h.remember(own)
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
	h.remember(h.fields)
	h.fields, h.weight = nil, 0
	return merged
}

// remember notes the names that have now ridden a chunk of this answer.
func (h *Held) remember(fields map[string]json.RawMessage) {
	for k := range fields {
		if len(h.carried) >= maxNames {
			h.unsure = true
			return
		}
		if h.carried == nil {
			h.carried = make(map[string]bool, len(fields))
		}
		h.carried[k] = true
	}
}
