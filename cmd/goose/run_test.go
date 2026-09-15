// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func runWith(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	dir := t.TempDir()
	outPath := filepath.Join(dir, "out")
	errPath := filepath.Join(dir, "err")
	out, err := os.Create(outPath)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	errf, err := os.Create(errPath)
	if err != nil {
		t.Fatal(err)
	}
	defer errf.Close()

	runErr := run(args, out, errf)
	outBytes, _ := os.ReadFile(outPath)
	errBytes, _ := os.ReadFile(errPath)
	return string(outBytes), string(errBytes), runErr
}

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "goose.toml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestVersionPrintsAndStops(t *testing.T) {
	out, _, err := runWith(t, "-version")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out, "goose ") {
		t.Fatalf("stdout = %q", out)
	}
}

func TestCheckValidatesWithoutServing(t *testing.T) {
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = 11234
[[provider]]
id = "local"
kind = "ollama"
[[mcp]]
name = "files"
command = "/nonexistent/mcp-server-that-would-fail-to-start"
`)
	out, _, err := runWith(t, "-c", cfg, "-check")
	if err != nil {
		t.Fatalf("a sound configuration was refused: %v", err)
	}
	if !strings.Contains(out, "is valid") {
		t.Fatalf("stdout = %q", out)
	}
	if !strings.Contains(out, "not started") {
		t.Fatalf("a check that spawns nothing must say so: %q", out)
	}
	if !strings.Contains(out, "POST /v1/chat/completions") {
		t.Fatalf("a check must print what would be served: %q", out)
	}
}

func TestRefusesAnOpenProxyOffLoopback(t *testing.T) {
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "0.0.0.0"
port = 11234
[[provider]]
id = "local"
kind = "ollama"
`)
	_, _, err := runWith(t, "-c", cfg, "-check")
	if err == nil {
		t.Fatal("served a routable address with no API key")
	}
	if !strings.Contains(err.Error(), "API key") {
		t.Fatalf("error = %v", err)
	}
}

func TestRefusesToolServersOnARoutableAddressWithoutAKey(t *testing.T) {
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "0.0.0.0"
port = 11234
api_key = "secret"
[[provider]]
id = "local"
kind = "ollama"
[[mcp]]
name = "files"
command = "mcp-server-filesystem"
`)
	if _, _, err := runWith(t, "-c", cfg, "-check"); err != nil {
		t.Fatalf("a keyed configuration was refused: %v", err)
	}
	body, err := os.ReadFile(cfg)
	if err != nil {
		t.Fatal(err)
	}
	noKey := writeConfig(t, strings.ReplaceAll(string(body), `api_key = "secret"`, ""))
	if _, _, err := runWith(t, "-c", noKey, "-check"); err == nil {
		t.Fatal("served tool servers on a routable address with no API key")
	}
}

func TestUnknownFormatIsRefused(t *testing.T) {
	cfg := writeConfig(t, `
formats = ["not-a-format"]
[server]
host = "127.0.0.1"
port = 11234
[[provider]]
id = "local"
kind = "ollama"
`)
	if _, _, err := runWith(t, "-c", cfg, "-check"); err == nil {
		t.Fatal("accepted a format it cannot serve")
	}
}

func TestProviderFlagReplacesTheFilesProviders(t *testing.T) {
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = 11234
[[provider]]
id = "stale"
kind = "ollama"
[[provider]]
id = "other"
kind = "ollama"
[[route]]
name = "r"
chain = ["stale"]
`)
	out, _, err := runWith(t, "-c", cfg, "-check", "-p", "ollama", "-m", "llama3.1:8b")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "stale") {
		t.Fatalf("the file's providers survived the flag: %q", out)
	}
}

func TestRunServesAndShutsDownOnASignal(t *testing.T) {
	needsSignal(t)
	port := freePort(t)
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = `+strconv.Itoa(port)+`
metrics = true
[[provider]]
id = "local"
kind = "ollama"
base_url = "http://127.0.0.1:1"
`)
	dir := t.TempDir()
	out, _ := os.Create(filepath.Join(dir, "out"))
	errf, _ := os.Create(filepath.Join(dir, "err"))
	defer out.Close()
	defer errf.Close()

	done := make(chan error, 1)
	go func() { done <- run([]string{"-c", cfg}, out, errf) }()

	base := "http://127.0.0.1:" + strconv.Itoa(port)
	waitFor(t, base+"/health")

	for _, path := range []string{"/health", "/ready", "/v1/models", "/metrics"} {
		resp, err := http.Get(base + path)
		if err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		if resp.StatusCode >= 500 && path != "/ready" {
			t.Fatalf("%s answered %d", path, resp.StatusCode)
		}
	}

	resp, err := http.Post(base+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	io.Copy(io.Discard, resp.Body)
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		t.Fatal("a backend that is not listening served a request")
	}

	raiseTerm(t)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the server did not come down on a signal")
	}
	l, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(port))
	if err != nil {
		t.Fatalf("the port is still held: %v", err)
	}
	l.Close()
}

func TestRunReportsAPortItCannotHave(t *testing.T) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	port := l.Addr().(*net.TCPAddr).Port

	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = `+strconv.Itoa(port)+`
[[provider]]
id = "local"
kind = "ollama"
`)
	dir := t.TempDir()
	out, _ := os.Create(filepath.Join(dir, "out"))
	errf, _ := os.Create(filepath.Join(dir, "err"))
	defer out.Close()
	defer errf.Close()

	if err := run([]string{"-c", cfg}, out, errf); err == nil {
		t.Fatal("a port it cannot have is not a start")
	}
}

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func waitFor(t *testing.T, url string) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get(url)
		if err == nil {
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("%s never answered", url)
}

func TestTheBannerNamesTheToolServersItConnected(t *testing.T) {
	script, err := filepath.Abs(filepath.Join("..", "..", "mcp", "testdata", "fake-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	cfg := &Config{MCP: []MCP{{Name: "files", Command: script}}}
	tools, err := connectTools(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer tools.Close()

	path := filepath.Join(t.TempDir(), "out")
	out, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	banner(out, cfg, "test", "127.0.0.1:0", []string{"POST /mcp -> tool servers"}, tools)
	out.Close()

	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"tools", "files", "fake", "9.9", "2 tools"} {
		if !strings.Contains(string(body), want) {
			t.Fatalf("banner = %q: missing %q", body, want)
		}
	}
}

func TestTheNamedForwardedHeaderSeparatesCallers(t *testing.T) {
	needsSignal(t)
	port := freePort(t)
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = `+strconv.Itoa(port)+`
max_per_minute = 1
forwarded_header = "X-Forwarded-For"
[[provider]]
id = "local"
kind = "ollama"
base_url = "http://127.0.0.1:1"
`)
	dir := t.TempDir()
	out, _ := os.Create(filepath.Join(dir, "out"))
	errf, _ := os.Create(filepath.Join(dir, "err"))
	defer out.Close()
	defer errf.Close()

	done := make(chan error, 1)
	go func() { done <- run([]string{"-c", cfg}, out, errf) }()

	base := "http://127.0.0.1:" + strconv.Itoa(port)
	waitFor(t, base+"/health")

	ask := func(client string) int {
		req, err := http.NewRequest(http.MethodGet, base+"/v1/models", nil)
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("X-Forwarded-For", client)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp.StatusCode
	}

	for _, client := range []string{"203.0.113.7", "203.0.113.8"} {
		if status := ask(client); status == http.StatusTooManyRequests {
			t.Fatalf("%s was throttled for another caller's request", client)
		}
	}
	if status := ask("203.0.113.7"); status != http.StatusTooManyRequests {
		t.Fatalf("second request from one caller answered %d, want 429", status)
	}

	raiseTerm(t)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the server did not come down on a signal")
	}
}

func TestAModelFlagWithNoProviderIsRefused(t *testing.T) {
	cfg := writeConfig(t, `
[[provider]]
id = "local"
kind = "ollama"
`)
	for _, args := range [][]string{
		{"-c", cfg, "-m", "llama3.1:8b", "-check"},
		{"-c", cfg, "-u", "http://127.0.0.1:1", "-check"},
	} {
		if _, _, err := runWith(t, args...); err == nil {
			t.Fatalf("%v was accepted", args)
		}
	}
	if _, _, err := runWith(t, "-c", cfg, "-p", "ollama", "-m", "llama3.1:8b", "-check"); err != nil {
		t.Fatalf("with a provider: %v", err)
	}
}
