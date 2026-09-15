// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package spawn

import (
	"crypto/sha256"
	"encoding/hex"
)

func shortHash(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:8])
}
