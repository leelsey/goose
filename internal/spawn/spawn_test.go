// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package spawn

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVaultDirectoryIsPrivate(t *testing.T) {
	v, err := OpenVault(t.TempDir(), "goose-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()
	info, err := os.Stat(v.Dir())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o700 {
		t.Fatalf("mode = %o: a directory holding system prompts must not be readable by anyone else", perm)
	}
}

func TestVaultFileIsPrivateAndMemoised(t *testing.T) {
	v, err := OpenVault(t.TempDir(), "goose-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()

	first, err := v.File("system", "be brief")
	if err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(first)
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("mode = %o: the prompt must not be world-readable", perm)
	}
	body, err := os.ReadFile(first)
	if err != nil || string(body) != "be brief" {
		t.Fatalf("content = %q, err = %v", body, err)
	}

	second, err := v.File("system", "be brief")
	if err != nil {
		t.Fatal(err)
	}
	if second != first {
		t.Fatalf("%s != %s: the same prompt must map to one file", second, first)
	}
	other, err := v.File("system", "be verbose")
	if err != nil {
		t.Fatal(err)
	}
	if other == first {
		t.Fatal("two prompts must not share a file")
	}
}

func TestVaultReplacesAnythingItDidNotWrite(t *testing.T) {
	v, err := OpenVault(t.TempDir(), "goose-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer v.Close()

	content := "be brief"
	predicted := filepath.Join(v.Dir(), "system-"+shortHash(content)+".txt")
	victim := filepath.Join(t.TempDir(), "victim.txt")
	if err := os.WriteFile(victim, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(victim, predicted); err != nil {
		t.Fatal(err)
	}

	path, err := v.File("system", content)
	if err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(path)
	if err != nil || string(body) != content {
		t.Fatalf("the child would have read %q, not the operator's prompt", body)
	}
	if got, _ := os.ReadFile(victim); string(got) != "original" {
		t.Fatalf("victim = %q: the write followed a link out of the vault", got)
	}
	if info, err := os.Lstat(path); err != nil || !info.Mode().IsRegular() {
		t.Fatalf("lstat = %v, %v: the planted link survived", info, err)
	}
}

func TestVaultCloseRemovesEverything(t *testing.T) {
	v, err := OpenVault(t.TempDir(), "goose-test-")
	if err != nil {
		t.Fatal(err)
	}
	dir := v.Dir()
	if _, err := v.File("system", "be brief"); err != nil {
		t.Fatal(err)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("stat = %v: a prompt must not outlive the provider that wrote it", err)
	}
	if err := v.Close(); err != nil {
		t.Fatalf("closing twice = %v", err)
	}
}

func TestScrubEnvWithholdsEverythingNotAllowed(t *testing.T) {
	env := []string{"HOME=/home/x", "GOOSE_PROXY_KEY=secret", "PATH=/bin", "MALFORMED"}
	out := ScrubEnv(env, []string{"HOME", "PATH"}, map[string]string{"EXTRA": "1"})
	joined := strings.Join(out, " ")
	if strings.Contains(joined, "secret") {
		t.Fatalf("env = %v: a spawned agent received this proxy's own key", out)
	}
	if strings.Contains(joined, "MALFORMED") {
		t.Fatalf("env = %v: an entry with no separator is not a variable", out)
	}
	for _, want := range []string{"HOME=/home/x", "PATH=/bin", "EXTRA=1"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("env = %v: missing %s", out, want)
		}
	}
}

func TestRingKeepsTheLastBytes(t *testing.T) {
	r := newRing(8)
	if got := string(r.Bytes()); got != "" {
		t.Fatalf("empty ring = %q", got)
	}
	r.Write([]byte("abc"))
	if got := string(r.Bytes()); got != "abc" {
		t.Fatalf("short write = %q", got)
	}
	r.Write([]byte("defghij"))
	if got := string(r.Bytes()); got != "cdefghij" {
		t.Fatalf("wrapped write = %q, want the last eight bytes", got)
	}
	r.Write([]byte("0123456789"))
	if got := string(r.Bytes()); got != "23456789" {
		t.Fatalf("oversized write = %q, want the tail", got)
	}
}

func TestProcessDeliversStdinAndReapsCleanly(t *testing.T) {
	p, err := Start(context.Background(), Options{
		Binary: "/bin/sh", Args: []string{"-c", "cat; echo done >&2"},
		Stdin: []byte("hello from stdin"),
	})
	if err != nil {
		t.Fatal(err)
	}
	out, err := io.ReadAll(p.Stdout())
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != "hello from stdin" {
		t.Fatalf("stdout = %q", out)
	}
	if err := p.Wait(); err != nil {
		t.Fatal(err)
	}
	if p.Stderr() != "done" {
		t.Fatalf("stderr = %q", p.Stderr())
	}
	if code := p.ExitCode(); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if err := p.Wait(); err != nil {
		t.Fatalf("waiting twice = %v", err)
	}
}

func TestCancelledProcessIsNotOrphaned(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	p, err := Start(ctx, Options{
		Binary: "/bin/sh", Args: []string{"-c", "sleep 30"}, Grace: 200 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	done := make(chan struct{})
	go func() { _ = p.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("a cancelled child outlived its grace period")
	}
}

func TestCloseEndsARunningChild(t *testing.T) {
	p, err := Start(context.Background(), Options{
		Binary: "/bin/sh", Args: []string{"-c", "sleep 30"}, Grace: 200 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- p.Close() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("close = %v: a signalled child is not a failure", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close did not end the child")
	}
}

func TestStartRefusesWithoutABinary(t *testing.T) {
	if _, err := Start(context.Background(), Options{}); err == nil {
		t.Fatal("a spawn with no binary must fail rather than guess")
	}
}

func TestClosedVaultRefusesRatherThanFallingBackToTheWorkingDirectory(t *testing.T) {
	v, err := OpenVault(t.TempDir(), "goose-test-")
	if err != nil {
		t.Fatal(err)
	}
	if err := v.Close(); err != nil {
		t.Fatal(err)
	}
	path, err := v.File("system", "be brief")
	if err == nil {
		t.Fatalf("wrote %s after the vault was closed", path)
	}
	if _, err := os.Stat("system-" + shortHash("be brief") + ".txt"); err == nil {
		t.Fatal("a system prompt was left in the working directory")
	}
}

func TestAScrubThatMatchedNothingIsEmptyNotInherited(t *testing.T) {
	env := ScrubEnv([]string{"GOOSE_PROXY_KEY=secret", "ANTHROPIC_API_KEY=sk"}, []string{"PATH"}, nil)
	if env == nil {
		t.Fatal("a scrub that matched nothing returned nil, which exec reads as inherit")
	}
	if len(env) != 0 {
		t.Fatalf("env = %v", env)
	}
}

func TestAChildWithNoEnvironmentInheritsNone(t *testing.T) {
	t.Setenv("GOOSE_TEST_SECRET", "leaked")
	dir := t.TempDir()
	script := filepath.Join(dir, "env.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nenv\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	p, err := Start(context.Background(), Options{Binary: script, Grace: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	out, _ := io.ReadAll(p.Stdout())
	_ = p.Close()
	if strings.Contains(string(out), "GOOSE_TEST_SECRET") {
		t.Fatalf("the child inherited this process's environment:\n%s", out)
	}
}

func TestRingKeepsAWriteThatLandsOnTheBoundary(t *testing.T) {
	r := newRing(8)
	if _, err := r.Write([]byte("abcde")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Write([]byte("fgh")); err != nil {
		t.Fatal(err)
	}
	if got := string(r.Bytes()); got != "abcdefgh" {
		t.Fatalf("tail = %q: filling the buffer exactly is still filling it", got)
	}
}

func TestRingKeepsTheTailOnceItWraps(t *testing.T) {
	r := newRing(8)
	if _, err := r.Write([]byte("abcdefgh")); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Write([]byte("ij")); err != nil {
		t.Fatal(err)
	}
	if got := string(r.Bytes()); got != "cdefghij" {
		t.Fatalf("tail = %q", got)
	}
}

func TestAChildThatIgnoresTheHintIsKilled(t *testing.T) {
	p, err := StartPiped(context.Background(), Options{
		Binary: "/bin/sh",
		Args:   []string{"-c", "trap '' INT; while :; do sleep 1; done"},
		Grace:  300 * time.Millisecond,
	})
	if err != nil {
		t.Skipf("no shell to spawn: %v", err)
	}
	time.Sleep(100 * time.Millisecond)

	// Without a bound here this blocks for as long as the child cares to live,
	// and exec's own watchdog does not run for the context such a child is
	// started with.
	start := time.Now()
	done := make(chan struct{})
	go func() { _ = p.Close(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("a server that traps the signal and stays up held shutdown open")
	}
	if took := time.Since(start); took > 3*time.Second {
		t.Fatalf("close took %v: the grace period is what bounds this", took)
	}
}
