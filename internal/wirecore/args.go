// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"bytes"
	"encoding/json"
)

// ObjectArgs shapes a tool call's arguments into the object a wire expects.
// Broken returns whether what arrived could not be used, so that a call cut off
// mid-argument is reported rather than handed over as a call with none.
func ObjectArgs(raw json.RawMessage) json.RawMessage {
	out, _ := ObjectArgsBroken(raw)
	return out
}

func ObjectArgsBroken(raw json.RawMessage) (json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return json.RawMessage(`{}`), false
	}
	if !json.Valid(trimmed) {
		return json.RawMessage(`{}`), true
	}
	if trimmed[0] == '{' {
		return trimmed, false
	}
	wrapped, err := json.Marshal(map[string]json.RawMessage{"_raw": trimmed})
	if err != nil {
		return json.RawMessage(`{}`), true
	}
	return wrapped, false
}
