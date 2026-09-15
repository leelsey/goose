// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

//go:build !unix

package main

import "testing"

func needsSignal(t *testing.T) {
	t.Helper()
	t.Skip("this platform has no signal a process can raise on itself")
}

func raiseTerm(t *testing.T) {
	t.Helper()
	t.Fatal("a signal was raised on a platform that cannot deliver one")
}
