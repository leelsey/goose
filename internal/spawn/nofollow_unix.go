// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

//go:build !windows

package spawn

import "syscall"

const openNoFollow = syscall.O_NOFOLLOW
