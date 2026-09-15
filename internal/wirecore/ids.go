// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import "fmt"

type IDMinter struct{ seen map[string]int }

func NewIDMinter() *IDMinter { return &IDMinter{seen: map[string]int{}} }

func (m *IDMinter) Next(name string) string {
	if m.seen == nil {
		m.seen = map[string]int{}
	}
	n := m.seen[name]
	m.seen[name] = n + 1
	return fmt.Sprintf("call_%s_%d", name, n)
}
