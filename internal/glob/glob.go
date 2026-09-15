// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package glob matches model names with * and ?, case-insensitively.
package glob

import "strings"

func Match(pattern, name string) bool {
	if pattern == "" {
		return false
	}
	if pattern == "*" {
		return true
	}
	p, n := strings.ToLower(pattern), strings.ToLower(name)
	if match(p, n) {
		return true
	}
	if strings.ContainsAny(pattern, separators) {
		return false
	}
	for i := 0; i < len(n); i++ {
		if strings.IndexByte(separators, n[i]) >= 0 && match(p, n[i+1:]) {
			return true
		}
	}
	return false
}

const separators = "/.:"

func match(pattern, s string) bool {
	pi, si := 0, 0
	star, mark := -1, 0
	for si < len(s) {
		switch {
		case pi < len(pattern) && (pattern[pi] == '?' || pattern[pi] == s[si]):
			pi++
			si++
		case pi < len(pattern) && pattern[pi] == '*':
			star = pi
			mark = si
			pi++
		case star >= 0:
			pi = star + 1
			mark++
			si = mark
		default:
			return false
		}
	}
	for pi < len(pattern) && pattern[pi] == '*' {
		pi++
	}
	return pi == len(pattern)
}
