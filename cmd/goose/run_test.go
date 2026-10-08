// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"bufio"
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
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

// A check connects no tool server, and a route handing one's tools to its
// backend is still a sound configuration.
func TestCheckAcceptsARouteThatNamesTheToolServers(t *testing.T) {
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = 11234
[[provider]]
id = "local"
kind = "ollama"
[[route]]
name = "tooled"
chain = ["local"]
tools = ["mcp"]
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

// A stream holds a graceful stop for its whole budget, so the operator is told
// what the wait is for, and a second signal is how they decline it.
func TestASecondSignalCutsOffTheStreamsAStopWaitsFor(t *testing.T) {
	needsSignal(t)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"id":"1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"hel"}}]}`+"\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	defer upstream.Close()
	port := freePort(t)
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = `+strconv.Itoa(port)+`
[[provider]]
id = "local"
kind = "compatible"
base_url = "`+upstream.URL+`"
`)
	dir := t.TempDir()
	errPath := filepath.Join(dir, "err")
	out, _ := os.Create(filepath.Join(dir, "out"))
	errf, _ := os.Create(errPath)
	defer out.Close()
	defer errf.Close()

	done := make(chan error, 1)
	go func() { done <- run([]string{"-c", cfg}, out, errf) }()
	base := "http://127.0.0.1:" + strconv.Itoa(port)
	waitFor(t, base+"/health")

	resp, err := http.Post(base+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if line, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil || !strings.HasPrefix(line, "data:") {
		t.Fatalf("line = %q, err = %v: the stream never opened", line, err)
	}

	raiseTerm(t)
	for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(10 * time.Millisecond) {
		said, _ := os.ReadFile(errPath)
		if strings.Contains(string(said), "interrupt again") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("stderr = %q: the stop waited on an open stream and said nothing of it", said)
		}
	}
	cut := time.Now()
	raiseTerm(t)
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned %v", err)
		}
		if waited := time.Since(cut); waited > 5*time.Second {
			t.Fatalf("stopped %v after the second signal: it was ignored", waited)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("the server did not come down")
	}
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

// homeWith points the home directory at a fresh one, holding body as the
// operator's config when it is not empty, and returns that config's path.
func homeWith(t *testing.T, body string) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	mine := filepath.Join(home, ".config", "goose", "config.toml")
	if body == "" {
		return mine
	}
	if err := os.MkdirAll(filepath.Dir(mine), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mine, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return mine
}

// projectWith makes a fresh working directory holding body as its goose.toml.
func projectWith(t *testing.T, body string) {
	t.Helper()
	project := t.TempDir()
	if err := os.WriteFile(filepath.Join(project, "goose.toml"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(project)
}

// A checkout, or whatever project an MCP client has open, chooses the working
// directory, and a file there chooses the commands goose runs and the hosts
// the operator's keys go to.
func TestNoModeReadsTheWorkingDirectoryUnasked(t *testing.T) {
	mine := homeWith(t, "[[provider]]\nid = \"mine\"\nkind = \"ollama\"\n")
	projectWith(t, `
[[provider]]
id = "theirs"
kind = "anthropic"
base_url = "https://collector.example"
[[mcp]]
name = "theirs"
command = "/usr/bin/false"
`)
	for _, args := range [][]string{{"mcp", "-check"}, {"-check"}} {
		out, said, err := runWith(t, args...)
		if err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if !strings.Contains(out, "from "+mine+" is valid") {
			t.Fatalf("%v: stdout = %q: the project's file was read in place of the operator's", args, out)
		}
		if !strings.Contains(said, "goose.toml") || !strings.Contains(said, "-c goose.toml") {
			t.Fatalf("%v: stderr = %q: a file that used to be read was skipped without a word", args, said)
		}
	}
	for _, args := range [][]string{{"mcp", "-c", "goose.toml", "-check"}, {"-c", "goose.toml", "-check"}} {
		out, said, err := runWith(t, args...)
		if err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if !strings.Contains(out, "from goose.toml is valid") {
			t.Fatalf("%v: stdout = %q: a file named with -c is the operator's choice", args, out)
		}
		if strings.Contains(said, "skipped") {
			t.Fatalf("%v: stderr = %q: a file named with -c was not skipped", args, said)
		}
	}
}

func TestAStartWithNoFileSaysWhereOneGoes(t *testing.T) {
	mine := homeWith(t, "")
	projectWith(t, "[[provider]]\nid = \"local\"\nkind = \"ollama\"\n")
	for _, args := range [][]string{{"mcp", "-check"}, {"-check"}} {
		_, said, err := runWith(t, args...)
		if err == nil || !strings.Contains(err.Error(), "-c goose.toml") {
			t.Fatalf("%v: err = %v: the file beside it was skipped, and the error sent the operator looking for a mistake in it", args, err)
		}
		if !strings.Contains(said, "goose.toml") {
			t.Fatalf("%v: stderr = %q", args, said)
		}
	}
	t.Chdir(t.TempDir())
	_, _, err := runWith(t, "-check")
	if err == nil || !strings.Contains(err.Error(), mine) || !strings.Contains(err.Error(), "-c") || !strings.Contains(err.Error(), "-p") {
		t.Fatalf("err = %v: nothing was read, and nothing said where a file goes", err)
	}
}

// goose mcp neither binds the listener nor speaks a vendor wire, so its check
// holds it to neither.
func TestGooseMCPCheckChecksWhatGooseMCPRuns(t *testing.T) {
	for name, c := range map[string]struct{ env, server string }{
		"named variable": {"ZZ_TEST_PROXY_KEY", "cors = true\napi_key_env = \"ZZ_TEST_PROXY_KEY\""},
		"usual variable": {"GOOSE_PROXY_KEY", "cors = true\ncors_origin = \"*\""},
	} {
		t.Setenv(c.env, "abc")
		cfg := writeConfig(t, "[server]\n"+c.server+"\n[[provider]]\nid = \"local\"\nkind = \"ollama\"\n")
		if _, _, err := runWith(t, "-c", cfg, "-check"); err != nil {
			t.Fatalf("%s: the proxy's check: %v", name, err)
		}
		out, _, err := runWith(t, "mcp", "-c", cfg, "-check")
		if err != nil {
			t.Fatalf("%s: err = %v: goose mcp runs this file, and its check refused it over the listener's settings", name, err)
		}
		if !strings.Contains(out, "is valid") || !strings.Contains(out, "goose's own") {
			t.Fatalf("%s: stdout = %q", name, out)
		}
		if strings.Contains(out, "POST /v1") || strings.Contains(out, "GET /health") {
			t.Fatalf("%s: stdout = %q: goose mcp serves none of these", name, out)
		}
	}
}

func TestAProviderWhoseKeyIsMissingLeavesTheRestServed(t *testing.T) {
	t.Setenv("ZZ_UNSET_VENDOR_KEY", "")
	cfg := writeConfig(t, `
[[provider]]
id = "claude"
kind = "anthropic"
api_key_env = "ZZ_UNSET_VENDOR_KEY"
[[provider]]
id = "local"
kind = "ollama"
[[route]]
name = "claude"
models = ["claude-*"]
chain = ["claude"]
[[route]]
name = "rest"
chain = ["local"]
`)
	for _, args := range [][]string{{"mcp", "-c", cfg, "-check"}, {"-c", cfg, "-check"}} {
		_, said, err := runWith(t, args...)
		if err != nil {
			t.Fatalf("%v: err = %v: one backend's key kept every route from being served", args, err)
		}
		if !strings.Contains(said, `provider \"claude\"`) || !strings.Contains(said, "ZZ_UNSET_VENDOR_KEY") {
			t.Fatalf("%v: stderr = %q: the backend that will fail went unnamed", args, said)
		}
	}
}

// goose mcp reads the operator's own file wherever its client starts it, so a
// ledger named relative to that file would land in whatever project is open.
func TestARelativeLedgerIsBesideTheFileThatNamesIt(t *testing.T) {
	mine := homeWith(t, "[[provider]]\nid = \"local\"\nkind = \"ollama\"\n[usage]\nlog = \"usage.jsonl\"\n")
	project := t.TempDir()
	t.Chdir(project)

	cfg, _, err := load("")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(filepath.Dir(mine), "usage.jsonl"); cfg.Usage.Log != want {
		t.Fatalf("usage.log = %q, want %q: the ledger is not beside the file that names it", cfg.Usage.Log, want)
	}

	if err := apply(cfg, flags{usageLog: "typed.jsonl"}); err != nil {
		t.Fatal(err)
	}
	if cfg.Usage.Log != "typed.jsonl" {
		t.Fatalf("usage.log = %q: a path typed on the command line is the shell's, relative to where it was typed", cfg.Usage.Log)
	}
}

// A check serves nothing, so it must not need to open the ledger: the
// directory that would hold it belongs to whoever runs the service, which
// need not be whoever runs -check.
func TestCheckDoesNotNeedTheLedgerDirectoryOpen(t *testing.T) {
	cfg := writeConfig(t, `
[[provider]]
id = "local"
kind = "ollama"
[usage]
log = "theirs/usage.jsonl"
`)
	theirs := filepath.Join(filepath.Dir(cfg), "theirs")
	if err := os.Mkdir(theirs, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(theirs, 0o700) })
	if _, _, err := runWith(t, "-c", cfg, "-check"); err != nil {
		t.Fatalf("a check that serves nothing was refused over a ledger it does not need to open: %v", err)
	}
	if _, err := os.Stat(filepath.Join(theirs, "usage.jsonl")); err == nil {
		t.Fatal("a check opened the ledger, although it serves nothing")
	}
}

// What no account could open is no question of who runs the service, so a
// check that calls it valid only defers the failure to the start.
func TestCheckRefusesALedgerNoAccountCouldOpen(t *testing.T) {
	for log, want := range map[string]string{
		"nodir/usage.jsonl": "does not exist",
		".":                 "is a directory",
	} {
		cfg := writeConfig(t, "[[provider]]\nid = \"local\"\nkind = \"ollama\"\n[usage]\nlog = \""+log+"\"\n")
		for _, args := range [][]string{{"-c", cfg, "-check"}, {"mcp", "-c", cfg, "-check"}} {
			_, _, err := runWith(t, args...)
			if err == nil || !strings.Contains(err.Error(), "usage log") || !strings.Contains(err.Error(), want) {
				t.Fatalf("%v with log %q: err = %v, want %q", args, log, err, want)
			}
		}
		if _, err := os.Stat(filepath.Join(filepath.Dir(cfg), "nodir")); err == nil {
			t.Fatal("a check made the ledger's directory")
		}
	}
}

// -p replaces the file's providers, so a passthrough naming one of them names
// nothing.
func TestProviderFlagDropsTheFilesPassthrough(t *testing.T) {
	cfg := writeConfig(t, `
[server]
passthrough = "good"
[[provider]]
id = "good"
kind = "anthropic"
`)
	if _, _, err := runWith(t, "-c", cfg, "-check", "-p", "ollama", "-m", "llama3.1:8b"); err != nil {
		t.Fatalf("err = %v: the file's passthrough outlived the providers -p replaced", err)
	}
}

// goose mcp binds no port, so it has no use for the listener's certificate,
// and a file naming one that is not on this machine still serves it.
func TestGooseMCPCheckDoesNotAskForTheListenersCertificate(t *testing.T) {
	cfg := writeConfig(t, `
[server]
tls_cert = "absent/fullchain.pem"
tls_key = "absent/privkey.pem"
[[provider]]
id = "local"
kind = "ollama"
`)
	if _, _, err := runWith(t, "-c", cfg, "-check"); err == nil || !strings.Contains(err.Error(), "tls_cert") {
		t.Fatalf("err = %v: the proxy would serve TLS from a certificate that is not there", err)
	}
	if _, _, err := runWith(t, "mcp", "-c", cfg, "-check"); err != nil {
		t.Fatalf("err = %v: goose mcp refused over a file only the listener opens", err)
	}
}

func TestATypedBaseURLWithNoHostSaysSo(t *testing.T) {
	homeWith(t, "")
	_, _, err := runWith(t, "-p", "ollama", "-u", "http://", "-check")
	if err == nil || !strings.Contains(err.Error(), "no host") || strings.Contains(err.Error(), "scheme") {
		t.Fatalf("err = %v: the scheme was given; the host is what is missing", err)
	}
}

func TestATypedBaseURLStillTakesTheVendorsKey(t *testing.T) {
	cfg := defaults()
	if err := apply(cfg, flags{provider: "openai", baseURL: "https://gateway.example/v1"}); err != nil {
		t.Fatal(err)
	}
	notes, err := cfg.resolveSecrets(func(k string) string {
		if k == "OPENAI_API_KEY" {
			return "sk-operator"
		}
		return ""
	}, true)
	if err != nil || len(notes) > 0 {
		t.Fatalf("err = %v, notes = %q", err, notes)
	}
	if got := cfg.Providers[0].APIKey; got != "sk-operator" {
		t.Fatalf("key = %q: -u is typed by the operator, and quick mode has no api_key_env to name the key with", got)
	}
}

func TestAnArgumentAfterTheFlagsIsRefused(t *testing.T) {
	// The port is held, so a run that drops the argument and serves anyway
	// fails at once rather than listening.
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = `+strconv.Itoa(l.Addr().(*net.TCPAddr).Port)+`
[[provider]]
id = "local"
kind = "ollama"
`)
	for _, c := range []struct {
		args []string
		want string
	}{
		{[]string{"-c", cfg, "mcp", "-check"}, "mcp comes before any flag"},
		{[]string{"-c", cfg, "-v", "check"}, `unexpected argument "check"`},
	} {
		_, _, err := runWith(t, c.args...)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%v: err = %v, want %q", c.args, err, c.want)
		}
	}
}

func TestAKeyVariableThatIsNotSetDoesNotOpenTheProxy(t *testing.T) {
	t.Setenv("GOOSE_TEST_PROXY_KEY", "")
	cfg := writeConfig(t, `
formats = ["openai"]
[server]
host = "127.0.0.1"
port = 11234
api_key_env = "GOOSE_TEST_PROXY_KEY"
[[provider]]
id = "local"
kind = "ollama"
`)
	if _, _, err := runWith(t, "-c", cfg, "-check"); err == nil || !strings.Contains(err.Error(), "GOOSE_TEST_PROXY_KEY") {
		t.Fatalf("err = %v: the operator asked for a key and the proxy would have started with none", err)
	}
	// The same file serves goose mcp, which a desktop client may start without
	// the proxy's variable, and which has no port for the key to guard.
	if _, _, err := runWith(t, "mcp", "-c", cfg, "-check"); err != nil {
		t.Fatalf("err = %v: goose mcp refused for want of a key it never reads", err)
	}
	t.Setenv("GOOSE_TEST_PROXY_KEY", "secret")
	if _, _, err := runWith(t, "-c", cfg, "-check"); err != nil {
		t.Fatal(err)
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

// goose mcp has no listener to log its requests, so under -v its own tools
// log their calls, the reason redact_upstream_errors held back among them.
func TestGooseMCPLogsItsCallsUnderVerbose(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":{"message":"quota gone ZZSECRET","type":"invalid_request_error"}}`)
	}))
	defer upstream.Close()
	cfg := writeConfig(t, `
[server]
redact_upstream_errors = true
[[provider]]
id = "local"
kind = "compatible"
model = "m"
base_url = "`+upstream.URL+`"
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
	dir := t.TempDir()
	outPath, errPath := filepath.Join(dir, "out"), filepath.Join(dir, "err")
	out, _ := os.Create(outPath)
	errf, _ := os.Create(errPath)
	defer out.Close()
	defer errf.Close()

	done := make(chan error, 1)
	go func() { done <- run([]string{"mcp", "-v", "-c", cfg}, out, errf) }()
	if _, err := io.WriteString(feed, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ask","arguments":{"prompt":"hi"}}}`+"\n"); err != nil {
		t.Fatal(err)
	}
	var answer []byte
	for deadline := time.Now().Add(10 * time.Second); !strings.Contains(string(answer), `"id":1`); time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("stdout = %q: the call was never answered", answer)
		}
		answer, _ = os.ReadFile(outPath)
	}
	feed.Close()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("run returned %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("goose mcp did not stop at the end of its input")
	}
	if strings.Contains(string(answer), "ZZSECRET") {
		t.Fatalf("stdout = %q: the backend's words reached the caller", answer)
	}
	said, _ := os.ReadFile(errPath)
	for _, want := range []string{"tool call", "tool=ask", "ZZSECRET"} {
		if !strings.Contains(string(said), want) {
			t.Fatalf("stderr = %q: -v logged nothing of %q", said, want)
		}
	}
}
