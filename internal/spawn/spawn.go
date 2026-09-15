// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package spawn runs a child process and owns its lifetime.
package spawn

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	DefaultGrace = 3 * time.Second
	stderrTail   = 16 << 10
)

type Options struct {
	Binary string
	Args   []string
	Dir    string
	// Env is the child's whole environment, never inherited; nil is empty.
	Env   []string
	Stdin []byte
	Grace time.Duration
}

type Process struct {
	cmd    *exec.Cmd
	stdout io.ReadCloser
	stdin  io.WriteCloser
	stderr *ring

	once sync.Once
	err  error
}

func StartPiped(ctx context.Context, o Options) (*Process, error) {
	return start(ctx, o, true)
}

func Start(ctx context.Context, o Options) (*Process, error) {
	return start(ctx, o, false)
}

func (p *Process) Write(b []byte) (int, error) {
	if p.stdin == nil {
		return 0, errors.New("spawn: process has no open stdin")
	}
	return p.stdin.Write(b)
}

func start(ctx context.Context, o Options, piped bool) (*Process, error) {
	if o.Binary == "" {
		return nil, errors.New("spawn: no binary")
	}
	if o.Grace <= 0 {
		o.Grace = DefaultGrace
	}
	cmd := exec.CommandContext(ctx, o.Binary, o.Args...)
	cmd.Dir = o.Dir
	cmd.Env = o.Env
	if cmd.Env == nil {
		cmd.Env = []string{}
	}
	setProcessGroup(cmd)
	cmd.Cancel = func() error { return interrupt(cmd) }
	cmd.WaitDelay = o.Grace

	var stdin io.WriteCloser
	switch {
	case piped:
		var err error
		if stdin, err = cmd.StdinPipe(); err != nil {
			return nil, err
		}
	case len(o.Stdin) > 0:
		cmd.Stdin = bytes.NewReader(o.Stdin)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	errBuf := newRing(stderrTail)
	cmd.Stderr = errBuf

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("spawn %s: %w", o.Binary, err)
	}
	return &Process{cmd: cmd, stdout: stdout, stdin: stdin, stderr: errBuf}, nil
}

func (p *Process) Stdout() io.Reader { return p.stdout }

func (p *Process) Stderr() string { return strings.TrimSpace(string(p.stderr.Bytes())) }

func (p *Process) Wait() error {
	p.once.Do(func() { p.err = p.cmd.Wait() })
	return p.err
}

func (p *Process) Close() error {
	if p.stdin != nil {
		_ = p.stdin.Close()
	}
	if p.cmd.Process != nil {
		_ = interrupt(p.cmd)
	}
	_ = p.stdout.Close()
	err := p.Wait()
	if err != nil && isSignalled(err) {
		return nil
	}
	return err
}

func (p *Process) ExitCode() int {
	if p.cmd.ProcessState == nil {
		return -1
	}
	return p.cmd.ProcessState.ExitCode()
}

type Vault struct {
	dir string
	mu  sync.Mutex
}

func OpenVault(parent, prefix string) (*Vault, error) {
	dir, err := os.MkdirTemp(parent, prefix)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(dir, 0o700); err != nil {
		_ = os.RemoveAll(dir)
		return nil, err
	}
	return &Vault{dir: dir}, nil
}

func (v *Vault) Dir() string { return v.dir }

func (v *Vault) File(prefix, content string) (string, error) {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.dir == "" {
		return "", errors.New("spawn: the vault is closed")
	}
	name := filepath.Join(v.dir, prefix+"-"+shortHash(content)+".txt")
	if info, err := os.Lstat(name); err == nil {
		if info.Mode().IsRegular() && info.Size() == int64(len(content)) {
			return name, nil
		}
		if err := os.Remove(name); err != nil {
			return "", err
		}
	}
	f, err := os.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY|openNoFollow, 0o600)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := f.WriteString(content); err != nil {
		return "", err
	}
	return name, nil
}

func (v *Vault) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.dir == "" {
		return nil
	}
	dir := v.dir
	v.dir = ""
	return os.RemoveAll(dir)
}

func ScrubEnv(environ []string, allow []string, extra map[string]string) []string {
	allowed := make(map[string]bool, len(allow))
	for _, name := range allow {
		allowed[name] = true
	}
	out := make([]string, 0, len(allow)+len(extra))
	for _, kv := range environ {
		name, _, ok := strings.Cut(kv, "=")
		if ok && allowed[name] {
			out = append(out, kv)
		}
	}
	for k, v := range extra {
		out = append(out, k+"="+v)
	}
	return out
}

type ring struct {
	mu   sync.Mutex
	buf  []byte
	size int
	full bool
	pos  int
}

func newRing(size int) *ring { return &ring{buf: make([]byte, size), size: size} }

func (r *ring) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := len(p)
	if n >= r.size {
		copy(r.buf, p[n-r.size:])
		r.pos, r.full = 0, true
		return n, nil
	}
	written := copy(r.buf[r.pos:], p)
	if written < n {
		copy(r.buf, p[written:])
		r.full = true
	}
	r.pos = (r.pos + n) % r.size
	return n, nil
}

func (r *ring) Bytes() []byte {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.full {
		return append([]byte(nil), r.buf[:r.pos]...)
	}
	out := make([]byte, 0, r.size)
	out = append(out, r.buf[r.pos:]...)
	return append(out, r.buf[:r.pos]...)
}
