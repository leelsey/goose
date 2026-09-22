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
	carried map[uint64]struct{}
	weight  int
	full    bool
	unsure  bool
}

// What has already gone out is remembered as a fixed-size mark apiece rather
// than as the field itself, because what a field is called and what it says are
// both the upstream's to choose the size of, and remembering them would put no
// ceiling on what a stream costs to encode. A field repeated unchanged is one
// mark however many chunks carry it, so an ordinary stream makes a handful;
// past this many the question is answered the quiet way rather than wrongly.
const maxCarried = 1024

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
	for k, v := range h.fields {
		if _, gone := h.carried[mark(k, v)]; !gone {
			return true
		}
	}
	return false
}

// mark stands for one field and what it said. A name that went out saying one
// thing and is stranded saying another has not gone out: the field is the pair,
// not the name.
func mark(name string, value json.RawMessage) uint64 {
	const (
		offset = 1469598103934665603
		prime  = 1099511628211
	)
	h := uint64(offset)
	for i := range len(name) {
		h = (h ^ uint64(name[i])) * prime
	}
	h = (h ^ 0) * prime
	for i := range len(value) {
		h = (h ^ uint64(value[i])) * prime
	}
	return h
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

// remember notes the fields that have now ridden a chunk of this answer.
func (h *Held) remember(fields map[string]json.RawMessage) {
	for k, v := range fields {
		if len(h.carried) >= maxCarried {
			h.unsure = true
			return
		}
		if h.carried == nil {
			h.carried = map[uint64]struct{}{}
		}
		h.carried[mark(k, v)] = struct{}{}
	}
}
