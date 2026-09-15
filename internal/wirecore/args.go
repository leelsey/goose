// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"bytes"
	"encoding/json"
)

func ObjectArgs(raw json.RawMessage) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" || !json.Valid(trimmed) {
		return json.RawMessage(`{}`)
	}
	if trimmed[0] == '{' {
		return trimmed
	}
	out, err := json.Marshal(map[string]json.RawMessage{"_raw": trimmed})
	if err != nil {
		return json.RawMessage(`{}`)
	}
	return out
}
