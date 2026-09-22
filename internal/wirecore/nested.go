// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// foldKey reduces a key the way encoding/json matches one: ASCII case is ignored,
// and so are the non-ASCII letters that case-fold onto ASCII ones — a Kelvin
// sign matches k, a long s matches s. Folding with ToLower instead both misses
// those and wrongly joins letters like a dotted capital I that json keeps apart.
func foldKey(key string) string {
	var b strings.Builder
	b.Grow(len(key))
	for _, r := range key {
		switch {
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r + ('a' - 'A'))
		case r < utf8.RuneSelf:
			b.WriteRune(r)
		default:
			b.WriteRune(simpleFold(r))
		}
	}
	return b.String()
}

// simpleFold walks a rune's fold orbit and takes the ASCII letter in it, if the
// orbit has one; otherwise the rune stands for itself.
func simpleFold(r rune) rune {
	for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
		if f < utf8.RuneSelf {
			if f >= 'A' && f <= 'Z' {
				return f + ('a' - 'A')
			}
			return f
		}
	}
	return r
}

func NestedExtra(raw json.RawMessage, modelled map[string]bool) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var all map[string]json.RawMessage
	if json.Unmarshal(raw, &all) != nil {
		return nil
	}
	rest := map[string]json.RawMessage{}
	for k, v := range all {
		if !modelled[foldKey(k)] {
			rest[k] = v
		}
	}
	if len(rest) == 0 {
		return nil
	}
	out, err := json.Marshal(rest)
	if err != nil {
		return nil
	}
	return out
}

// Shadowed names the keys that a JSON object spells more than one way for the
// same modelled field. encoding/json matches field names without regard to
// case, so "top_p" and "top_P" both land on the same field and the later one
// wins; the earlier is read and then quietly replaced.
func Shadowed(raw json.RawMessage, modelled map[string]bool) []string {
	if len(raw) == 0 {
		return nil
	}
	var all map[string]json.RawMessage
	if json.Unmarshal(raw, &all) != nil {
		return nil
	}
	seen := map[string]int{}
	for k := range all {
		folded := foldKey(k)
		if modelled[folded] {
			seen[folded]++
		}
	}
	var out []string
	for k, n := range seen {
		if n > 1 {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func NestedFrom(extra map[string]json.RawMessage, key string) map[string]json.RawMessage {
	raw, ok := extra[key]
	if !ok {
		return nil
	}
	var out map[string]json.RawMessage
	if json.Unmarshal(raw, &out) != nil {
		return nil
	}
	return out
}
