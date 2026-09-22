// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/leelsey/goose/llm"
)

// Field reads one key of an already-parsed object into dst. A key that is
// absent, empty or null leaves dst as it was, so a caller can tell "not said"
// from "said to be nothing".
func Field[T any](m map[string]json.RawMessage, key string, dst *T) error {
	raw, ok := m[key]
	if !ok || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return llm.Errorf(llm.KindInvalidRequest, "%s: %v", key, err)
	}
	return nil
}

// Bearer returns the credential an Authorization header carries, or "".
func Bearer(h string) string {
	const p = "Bearer "
	if len(h) > len(p) && strings.EqualFold(h[:len(p)], p) {
		return h[len(p):]
	}
	return ""
}

// DataURL decodes a base64 data URL into its media type and bytes.
func DataURL(s string) (mime string, data []byte, ok bool) {
	if !strings.HasPrefix(s, "data:") {
		return "", nil, false
	}
	rest := s[len("data:"):]
	i := strings.Index(rest, ",")
	if i < 0 {
		return "", nil, false
	}
	meta, payload := rest[:i], rest[i+1:]
	if !strings.HasSuffix(meta, ";base64") {
		return "", nil, false
	}
	b, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		return "", nil, false
	}
	return strings.TrimSuffix(meta, ";base64"), b, true
}
