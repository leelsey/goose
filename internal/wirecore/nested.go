// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"strings"
)

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
		if !modelled[strings.ToLower(k)] {
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
