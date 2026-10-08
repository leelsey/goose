// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/usage"
)

// A stand-in for an agent CLI that is part way through an answer when goose is
// stopped, and that takes a moment to let go once it is interrupted, as a real
// one does. Its row can only be booked after that moment.
func slowToLetGo(t *testing.T) (script string, started func()) {
	t.Helper()
	needsSignal(t)
	dir := t.TempDir()
	marker := filepath.Join(dir, "started")
	script = filepath.Join(dir, "fake-claude.sh")
	body := "#!/bin/sh\n" +
		"trap 'sleep 1; exit 130' INT\n" +
		": > '" + marker + "'\n" +
		`printf '%s\n' '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}}'` + "\n" +
		"sleep 30\n"
	if err := os.WriteFile(script, []byte(body), 0o700); err != nil {
		t.Fatal(err)
	}
	return script, func() {
		t.Helper()
		for deadline := time.Now().Add(20 * time.Second); time.Now().Before(deadline); time.Sleep(10 * time.Millisecond) {
			if _, err := os.Stat(marker); err == nil {
				return
			}
		}
		t.Fatal("the call never reached the CLI")
	}
}

func TestAShutdownWaitsForTheCallsItCutOffToBookWhatTheySpent(t *testing.T) {
	script, started := slowToLetGo(t)
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	mem := usage.NewMemory()
	cfg := &Config{Providers: []Provider{{ID: "cli", Kind: "claude", BaseURL: script}}}
	client, err := build(cfg, logger, []usage.Sink{mem}, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { client.Close(context.Background()) })
	codecs, err := buildCodecs([]string{"openai"})
	if err != nil {
		t.Fatal(err)
	}
	srv, err := httpx.New(client, httpx.Options{Codecs: codecs, Logger: logger})
	if err != nil {
		t.Fatal(err)
	}
	calls := &inflight{}
	server := &http.Server{Handler: calls.handler(srv.Handler())}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go server.Serve(l)

	resp, err := http.Post("http://"+l.Addr().String()+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	started()

	shutdown(server, calls, 50*time.Millisecond, logger, nil)
	if attempts, _ := mem.Stats(); attempts != 1 {
		t.Fatalf("attempts = %d once shutdown returned: the call was sent and billed, and the ledger would close before its row", attempts)
	}
}

func TestGooseMCPBooksTheCallInFlightWhenItIsStopped(t *testing.T) {
	script, started := slowToLetGo(t)
	dir := t.TempDir()
	ledger := filepath.Join(dir, "usage.jsonl")
	cfg := writeConfig(t, `
[[provider]]
id = "cli"
kind = "claude"
base_url = '`+script+`'
[usage]
log = '`+ledger+`'
summary = true
`)
	in, feed, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	stdin := os.Stdin
	os.Stdin = in
	t.Cleanup(func() {
		os.Stdin = stdin
		in.Close()
		feed.Close()
	})
	out, _ := os.Create(filepath.Join(dir, "out"))
	errf, _ := os.Create(filepath.Join(dir, "err"))
	defer out.Close()
	defer errf.Close()

	done := make(chan error, 1)
	go func() { done <- run([]string{"mcp", "-c", cfg}, out, errf) }()
	if _, err := fmt.Fprintln(feed, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ask","arguments":{"prompt":"hi"}}}`); err != nil {
		t.Fatal(err)
	}
	started()

	raiseTerm(t)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("goose mcp did not come down on a signal")
	}
	said, err := os.ReadFile(filepath.Join(dir, "err"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(said), "usage: 1 upstream attempts") {
		t.Fatalf("stderr = %q: the summary was printed before the call it counts had settled", said)
	}
	body, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if rows := strings.Count(string(body), "\n"); rows != 1 {
		t.Fatalf("rows = %d: the call was sent and billed, and the ledger closed before its row", rows)
	}
}
