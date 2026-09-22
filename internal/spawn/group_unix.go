// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

//go:build !windows

package spawn

import (
	"errors"
	"os/exec"
	"syscall"
)

func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// interrupt signals the child and everything it started. The group carries the
// grandchildren — an agent CLI's own helpers — which the direct signal never
// reaches, so the group is what is signalled.
//
// reaped says the child has already been waited on. Its pid is then the host's
// to hand out again, and a group signal sent to it could land on a stranger, so
// the only safe thing left is nothing.
func interrupt(cmd *exec.Cmd, reaped bool) error {
	if cmd.Process == nil || reaped {
		return nil
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGINT); err != nil {
		return cmd.Process.Signal(syscall.SIGINT)
	}
	return nil
}

func kill(cmd *exec.Cmd, reaped bool) {
	if cmd.Process == nil || reaped {
		return
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		_ = cmd.Process.Kill()
	}
}

func isSignalled(err error) bool {
	var exit *exec.ExitError
	if !errors.As(err, &exit) {
		return false
	}
	status, ok := exit.Sys().(syscall.WaitStatus)
	return ok && status.Signaled()
}
