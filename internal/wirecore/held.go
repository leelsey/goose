// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import "encoding/json"

// A frame that puts nothing on a wire can still carry fields that wire has no
// place for, and they wait for the next chunk that can carry them. How many
// such frames arrive, and how much each brings, is the upstream's to choose, so
// what waits is bounded. Nothing an upstream actually sends is near this.
const maxHeld = 64 << 10

// Held is what a codec is carrying forward from frames that wrote nothing.
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
	for k, v := range extra {
		if _, known := h.fields[k]; known {
			h.fields[k] = v
			continue
		}
		if h.weight+len(k)+len(v) > maxHeld {
			if !h.full {
				h.full = true
				overfull = true
			}
			continue
		}
		if h.fields == nil {
			h.fields = make(map[string]json.RawMessage, len(extra))
		}
		h.weight += len(k) + len(v)
		h.fields[k] = v
	}
	return overfull
}

// Carry merges what has waited into the fields of the chunk now being written
// and empties the hold. The chunk's own win, since they describe it.
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
	h.fields, h.weight, h.full = nil, 0, false
	return merged
}
