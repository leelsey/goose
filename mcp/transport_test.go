// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

// scripted is a tool server written in the test that needs it, for behaviour
// the servers in testdata have no reason to have. It answers the handshake and
// lists one tool, "run"; what a call to it does is the test's to say.
func scripted(name, call string) ServerConfig {
	script := `while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p')
  case "$line" in
    *notifications/initialized*) ;;
    *server/discover*) printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32601,"message":"unknown method"}}\n' "$id" ;;
    *initialize*) printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"scripted"}}}\n' "$id" ;;
    *tools/list*) printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"run","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
    *tools/call*)
` + call + `
      printf '{"jsonrpc":"2.0","id":"%s","result":{"content":[{"type":"text","text":"ok"}]}}\n' "$id" ;;
  esac
done`
	return ServerConfig{Name: name, Command: "sh", Args: []string{"-c", script}, Env: os.Environ()}
}

func connected(t *testing.T, cfg ServerConfig) *Client {
	t.Helper()
	c, err := Connect(context.Background(), cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func timed(t *testing.T, d time.Duration) context.Context {
	ctx, cancel := context.WithTimeout(context.Background(), d)
	t.Cleanup(cancel)
	return ctx
}

// A child that stops reading its input holds a write for as long as it likes.
// The call caught behind it gives up when its caller does, and the child is
// treated as gone rather than written to again.
func TestAServerThatStopsReadingCostsOneCall(t *testing.T) {
	c := connected(t, scripted("stuck", `case "$line" in *'"hang"'*) while true; do sleep 3600; done ;; esac`))

	if _, err := c.Call(timed(t, 300*time.Millisecond), "run", json.RawMessage(`{"mode":"hang"}`)); err == nil {
		t.Fatal("a server that never answers has not answered")
	}
	big := json.RawMessage(`{"pad":"` + strings.Repeat("x", 1<<20) + `"}`)
	returned := make(chan error, 1)
	go func() {
		_, err := c.Call(timed(t, 500*time.Millisecond), "run", big)
		returned <- err
	}()
	select {
	case err := <-returned:
		if err == nil {
			t.Fatal("a server that is not reading answered")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("a call written to a server that stopped reading ignored its deadline")
	}

	result, err := c.Call(timed(t, 10*time.Second), "run", nil)
	if err != nil {
		t.Fatalf("the call after it was lost as well: %v", err)
	}
	if llm.TextOf(result.Content) != "ok" {
		t.Fatalf("result = %+v", result.Content)
	}
}

// stalled is a source whose catalogue never arrives.
type stalled struct{ name string }

func (s *stalled) Name() string               { return s.name }
func (s *stalled) Info() ServerInfo           { return ServerInfo{Name: s.name} }
func (s *stalled) Cached() []llm.Tool         { return nil }
func (s *stalled) Raw(string) json.RawMessage { return nil }
func (s *stalled) Refused() []string          { return nil }
func (s *stalled) Truncated() bool            { return false }
func (s *stalled) Close() error               { return nil }
func (s *stalled) Tools(ctx context.Context) ([]llm.Tool, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}
func (s *stalled) Call(context.Context, string, json.RawMessage) (*Result, error) {
	return nil, errors.New("stalled")
}

// One server that does not answer a refresh costs its own refresh, not the
// refresh of every server listed after it.
func TestAStalledServerDoesNotHoldBackTheOthersRefresh(t *testing.T) {
	path, err := filepath.Abs(filepath.Join("testdata", "changing-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	m := NewManager()
	defer m.Close()
	m.sources = append(m.sources, &stalled{name: "stalled"})
	if err := m.Add(context.Background(), ServerConfig{Name: "changing", Command: path, Env: os.Environ()}, nil); err != nil {
		t.Fatal(err)
	}
	if got := m.Tools(); len(got) != 1 {
		t.Fatalf("tools = %+v", got)
	}
	if err := m.Refresh(timed(t, 500*time.Millisecond)); err == nil {
		t.Fatal("a refresh that never arrived is not a success")
	}
	if got := m.Tools(); len(got) != 2 {
		t.Fatalf("tools = %+v: the server behind a stalled one was never asked", got)
	}
}

// A call that gave up while it waited for its turn is not sent when the turn
// comes: the tool would run for a caller that has already been told it failed.
func TestACallThatGaveUpWaitingIsNotSent(t *testing.T) {
	log := filepath.Join(t.TempDir(), "ran")
	cfg := scripted("slow", `printf '%s\n' "$line" >> "$LOG"; sleep 1`)
	cfg.Env = append(cfg.Env, "LOG="+log)
	c := connected(t, cfg)
	ran := func(what string) bool {
		raw, _ := os.ReadFile(log)
		return strings.Contains(string(raw), what)
	}

	first := make(chan error, 1)
	go func() {
		_, err := c.Call(context.Background(), "run", json.RawMessage(`{"n":"first"}`))
		first <- err
	}()
	for deadline := time.Now().Add(5 * time.Second); !ran("first"); time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the first call never ran")
		}
	}

	start := time.Now()
	if _, err := c.Call(timed(t, 200*time.Millisecond), "run", json.RawMessage(`{"n":"second"}`)); err == nil {
		t.Fatal("a call that ran out of time waiting has no answer")
	}
	if took := time.Since(start); took > 800*time.Millisecond {
		t.Fatalf("took %s: a call waiting its turn must give up when its caller does", took)
	}
	if err := <-first; err != nil {
		t.Fatal(err)
	}
	for deadline := time.Now().Add(1500 * time.Millisecond); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if ran("second") {
			t.Fatal("the call its caller gave up on was sent and run anyway")
		}
	}
}

// Opening the older transport's stream is bounded like any other call: a
// server that takes the connection and never names an endpoint does not hold
// whoever is connecting for ever.
func TestOpeningAStreamIsBounded(t *testing.T) {
	quit := make(chan struct{})
	silent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-quit:
		}
	}))
	t.Cleanup(silent.Close)
	nameless := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, ": waiting\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-quit:
		}
	}))
	t.Cleanup(nameless.Close)
	t.Cleanup(func() { close(quit) })

	for _, tc := range []struct {
		what    string
		srv     *httptest.Server
		ctx     time.Duration
		timeout time.Duration
	}{
		{"a server that never answers, held to the caller's deadline", silent, 300 * time.Millisecond, time.Hour},
		{"a stream that names no endpoint, held to the caller's deadline", nameless, 300 * time.Millisecond, time.Hour},
		{"a server that never answers, held to the timeout", silent, 0, 300 * time.Millisecond},
		{"a stream that names no endpoint, held to the timeout", nameless, 0, 300 * time.Millisecond},
	} {
		t.Run(tc.what, func(t *testing.T) {
			ctx := context.Background()
			if tc.ctx > 0 {
				ctx = timed(t, tc.ctx)
			}
			done := make(chan error, 1)
			go func() {
				m := NewManager()
				defer m.Close()
				done <- m.Add(ctx, ServerConfig{Name: "sse", Transport: SSE, URL: tc.srv.URL, Timeout: tc.timeout}, tc.srv.Client())
			}()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("connected to a server that named no endpoint")
				}
			case <-time.After(3 * time.Second):
				t.Fatal("still connecting long after the deadline passed")
			}
		})
	}
}

// A server that restarts, or lets a session lapse, forgets the one goose was
// given. It is greeted again rather than written off for the life of the
// process.
func TestAServerThatForgetsTheSessionIsGreetedAgain(t *testing.T) {
	var greeted atomic.Int32
	var mu sync.Mutex
	live := map[string]bool{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(body, "server/discover"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
			return
		case strings.Contains(body, `"initialize"`):
			session := fmt.Sprintf("s%d", greeted.Add(1))
			mu.Lock()
			live[session] = true
			mu.Unlock()
			w.Header().Set("Mcp-Session-Id", session)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"forgetful"}}}`))
			return
		}
		mu.Lock()
		known := live[r.Header.Get("Mcp-Session-Id")]
		mu.Unlock()
		switch {
		case !known:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32001,"message":"Session not found"}}`))
		case id == "":
			w.WriteHeader(http.StatusAccepted)
		case strings.Contains(body, "tools/list"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[{"name":"remote","inputSchema":{"type":"object"}}]}}`))
		default:
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"content":[{"type":"text","text":"ok"}]}}`))
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, ServerConfig{Name: "h", Transport: Streamable, URL: srv.URL}, srv.Client()); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	clear(live)
	mu.Unlock()

	if _, err := m.Call(ctx, "remote", nil); err != nil {
		t.Fatalf("a server that forgot the session was not greeted again: %v", err)
	}
	if err := m.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	if got := m.Tools(); len(got) != 1 {
		t.Fatalf("tools = %+v", got)
	}
	if n := greeted.Load(); n != 2 {
		t.Fatalf("greeted %d times, want once at the start and once after the session was lost", n)
	}
}

// A call made while another is greeting the server again is not sent without
// a session: a server refuses that for the missing session, not the lost one,
// and the call would fail on a server that is about to be reachable again.
func TestACallMadeWhileTheServerIsGreetedAgainWaitsForIt(t *testing.T) {
	var greeted atomic.Int32
	regreeting, release := make(chan struct{}), make(chan struct{})
	var mu sync.Mutex
	live := map[string]bool{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(body, "server/discover"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
			return
		case strings.Contains(body, `"initialize"`):
			n := greeted.Add(1)
			if n == 2 {
				close(regreeting)
				<-release
			}
			session := fmt.Sprintf("s%d", n)
			mu.Lock()
			live[session] = true
			mu.Unlock()
			w.Header().Set("Mcp-Session-Id", session)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"forgetful"}}}`))
			return
		}
		session := r.Header.Get("Mcp-Session-Id")
		mu.Lock()
		known := live[session]
		mu.Unlock()
		switch {
		case session == "":
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":null,"error":{"code":-32000,"message":"Bad Request: Mcp-Session-Id header is required"}}`))
		case !known:
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32001,"message":"Session not found"}}`))
		case id == "":
			w.WriteHeader(http.StatusAccepted)
		case strings.Contains(body, "tools/list"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[{"name":"remote","inputSchema":{"type":"object"}}]}}`))
		default:
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"content":[{"type":"text","text":"ok"}]}}`))
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, ServerConfig{Name: "h", Transport: Streamable, URL: srv.URL}, srv.Client()); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	clear(live)
	mu.Unlock()

	first := make(chan error, 1)
	go func() {
		_, err := m.Call(ctx, "remote", nil)
		first <- err
	}()
	select {
	case <-regreeting:
	case <-time.After(5 * time.Second):
		t.Fatal("the server that forgot the session was never greeted again")
	}
	second := make(chan error, 1)
	go func() {
		_, err := m.Call(ctx, "remote", nil)
		second <- err
	}()
	refreshed := make(chan error, 1)
	go func() { refreshed <- m.Refresh(ctx) }()
	time.Sleep(200 * time.Millisecond)
	close(release)

	for what, done := range map[string]chan error{"first call": first, "second call": second, "refresh": refreshed} {
		if err := <-done; err != nil {
			t.Fatalf("%s: %v", what, err)
		}
	}
	if got := m.Tools(); len(got) != 1 {
		t.Fatalf("tools = %+v: a server greeted again lost its tools on the way", got)
	}
	if n := greeted.Load(); n != 2 {
		t.Fatalf("greeted %d times, want once at the start and once after the session was lost", n)
	}
}

func ended(t *testing.T, c *Client) {
	t.Helper()
	c.mu.Lock()
	tr := c.transport.(*stdioTransport)
	c.mu.Unlock()
	select {
	case <-tr.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the server exited and its transport never noticed")
	}
}

// A stdio server that exits is started again for the next call, rather than
// every later call failing on a pipe nobody reads.
func TestAServerThatExitsIsStartedAgain(t *testing.T) {
	c := connected(t, scripted("brief", `printf '{"jsonrpc":"2.0","id":"%s","result":{"content":[{"type":"text","text":"ok"}]}}\n' "$id"; exit 0`))
	for i := range 3 {
		result, err := c.Call(timed(t, 10*time.Second), "run", nil)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if llm.TextOf(result.Content) != "ok" {
			t.Fatalf("call %d: result = %+v", i, result.Content)
		}
		ended(t, c)
	}
}

// A server that goes away while it is being greeted is not dialled again from
// inside the greeting. That would greet the fresh one twice, which a strict
// server refuses, and leave it running behind a connection that failed.
func TestAServerThatExitsWhileGreetedIsNotStartedAgain(t *testing.T) {
	log := filepath.Join(t.TempDir(), "started")
	script := `echo $$ >> "$LOG"
[ "$(wc -l < "$LOG")" -eq 1 ] && exit 0
greeted=
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p')
  case "$line" in
    *notifications/initialized*) ;;
    *server/discover*) printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32601,"message":"unknown method"}}\n' "$id" ;;
    *initialize*)
      if [ -n "$greeted" ]; then
        printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32600,"message":"duplicate initialize"}}\n' "$id"
      else
        greeted=1
        printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"strict"}}}\n' "$id"
      fi ;;
    *tools/list*) printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[]}}\n' "$id" ;;
  esac
done`
	cfg := ServerConfig{Name: "flaky", Command: "sh", Args: []string{"-c", script}, Env: append(os.Environ(), "LOG="+log)}
	c, err := Connect(timed(t, 10*time.Second), cfg, nil)
	if err == nil {
		_ = c.Close()
		t.Fatal("connected to a server that exited before it was greeted")
	}
	raw, _ := os.ReadFile(log)
	if started := strings.Count(string(raw), "\n"); started != 1 {
		t.Fatalf("started %d servers for one connection that failed (%v)", started, err)
	}
}

// A server that cannot be reached serves none of its tools, so none are
// offered while it is away, and they come back with it.
func TestAServerThatCannotBeReachedIsNotAdvertised(t *testing.T) {
	var down atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		if down.Load() {
			http.Error(w, "unavailable", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(body, "server/discover"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
		case strings.Contains(body, `"initialize"`):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"away"}}}`))
		case strings.Contains(body, "tools/list"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[{"name":"remote","inputSchema":{"type":"object"}}]}}`))
		default:
			w.WriteHeader(http.StatusAccepted)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, ServerConfig{Name: "away", Transport: Streamable, URL: srv.URL}, srv.Client()); err != nil {
		t.Fatal(err)
	}
	down.Store(true)
	if err := m.Refresh(ctx); err == nil {
		t.Fatal("a server that cannot be reached was refreshed")
	}
	if got, offered := m.Tools(), m.Proxied(); len(got) != 0 || len(offered) != 0 {
		t.Fatalf("tools = %+v, offered = %+v: a server that is not there is still advertised", got, offered)
	}
	if got := m.Servers(); len(got) != 1 || got[0].Tools != 0 {
		t.Fatalf("servers = %+v", got)
	}
	down.Store(false)
	if err := m.Refresh(ctx); err != nil {
		t.Fatal(err)
	}
	if got := m.Tools(); len(got) != 1 {
		t.Fatalf("tools = %+v: the server came back and its tools did not", got)
	}
}

// One answer past the ceiling costs the call it answers. The server behind
// it goes on serving the calls after.
func TestAnAnswerTooLargeToHoldCostsOnlyItsCall(t *testing.T) {
	cfg := scripted("large", `case "$line" in *'"big"'*)
        printf '{"jsonrpc":"2.0","id":"%s","result":{"content":[{"type":"text","text":"%s"}]}}\n' "$id" "$(head -c 4096 /dev/zero | tr '\0' x)"
        continue ;;
      esac`)
	cfg.MaxFrame = 1024
	c := connected(t, cfg)

	if _, err := c.Call(timed(t, 10*time.Second), "run", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Call(timed(t, 10*time.Second), "run", json.RawMessage(`{"size":"big"}`)); !errors.Is(err, wirecore.ErrLineTooLong) {
		t.Fatalf("err = %v: an answer past the ceiling is refused, and says why", err)
	}
	for i := range 2 {
		if _, err := c.Call(timed(t, 10*time.Second), "run", nil); err != nil {
			t.Fatalf("call %d after the large one: %v", i, err)
		}
	}
	if _, err := c.Tools(timed(t, 10*time.Second)); err != nil {
		t.Fatalf("refresh after the large one: %v", err)
	}
}

// fading is a server of the older transport that goose can lose. end says
// what it is given on every open stream, answers every call held on it, and
// then ends the stream and forgets it, the way a server that restarts does.
// forget forgets them and leaves them open. A message sent to a stream it has
// forgotten is refused with refuse, and a tool call asked to hold is answered
// only when its stream ends.
type fading struct {
	*httptest.Server
	refuse         int
	opened, closed atomic.Int32

	mu      sync.Mutex
	streams map[string]chan string
	held    map[string][]string
}

func newFading(t *testing.T, refuse int) *fading {
	t.Helper()
	f := &fading{refuse: refuse, streams: map[string]chan string{}, held: map[string][]string{}}
	f.Server = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.Close)
	return f
}

func answered(id string) string {
	return `{"jsonrpc":"2.0","id":"` + id + `","result":{"content":[{"type":"text","text":"ok"}]}}`
}

func (f *fading) serve(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		defer f.closed.Add(1)
		name := fmt.Sprint(f.opened.Add(1))
		frames := make(chan string, 8)
		f.mu.Lock()
		f.streams[name] = frames
		f.mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "event: endpoint\ndata: /rpc?stream=%s\n\n", name)
		w.(http.Flusher).Flush()
		for {
			select {
			case <-r.Context().Done():
				return
			case frame, open := <-frames:
				if !open {
					return
				}
				fmt.Fprintf(w, "data: %s\n\n", frame)
				w.(http.Flusher).Flush()
			}
		}
	}
	raw, _ := io.ReadAll(r.Body)
	body := string(raw)
	id := requestID(body)
	stream := r.URL.Query().Get("stream")
	f.mu.Lock()
	defer f.mu.Unlock()
	frames := f.streams[stream]
	if frames == nil {
		http.Error(w, "no such session", f.refuse)
		return
	}
	w.WriteHeader(http.StatusAccepted)
	switch {
	case strings.Contains(body, `"initialize"`):
		frames <- `{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"sse"}}}`
	case strings.Contains(body, "tools/list"):
		frames <- `{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[{"name":"remote","inputSchema":{"type":"object"}}]}}`
	case strings.Contains(body, `"hold"`):
		f.held[stream] = append(f.held[stream], id)
	case strings.Contains(body, "tools/call"):
		frames <- answered(id)
	}
}

func (f *fading) end(say ...string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for name, frames := range f.streams {
		for _, id := range f.held[name] {
			frames <- answered(id)
		}
		for _, frame := range say {
			frames <- frame
		}
		close(frames)
	}
	clear(f.streams)
	clear(f.held)
}

func (f *fading) forget() {
	f.mu.Lock()
	defer f.mu.Unlock()
	clear(f.streams)
	clear(f.held)
}

// reachable says that c, having lost its server, dials it again: the calls
// after it are answered, and its tools are still offered.
func reachable(t *testing.T, c *Client) {
	t.Helper()
	for i := range 2 {
		result, err := c.Call(timed(t, 10*time.Second), "remote", nil)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if llm.TextOf(result.Content) != "ok" {
			t.Fatalf("call %d: result = %+v", i, result.Content)
		}
	}
	if tools, err := c.Tools(timed(t, 10*time.Second)); err != nil || len(tools) != 1 {
		t.Fatalf("tools = %+v, err = %v", tools, err)
	}
}

// A stream of the older transport that ends is opened again for the next call,
// and it is seen to have ended even when the last thing on it was a frame no
// call was waiting for: a log line, or the answer to a call whose caller had
// already given up. The server refuses a forgotten stream with 400 here, as some
// do, so the end of the stream is the only thing that says to open another.
func TestAStreamThatEndsIsOpenedAgain(t *testing.T) {
	for _, tc := range []struct {
		what string
		hold bool
		say  []string
	}{
		{what: "with nothing after the answer"},
		{what: "behind a log line", say: []string{`{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"bye"}}`}},
		{what: "behind an answer its caller gave up on", hold: true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			f := newFading(t, http.StatusBadRequest)
			c, err := Connect(context.Background(), ServerConfig{
				Name: "sse", Transport: SSE, URL: f.URL, Timeout: 5 * time.Second,
			}, f.Client())
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			if tc.hold {
				if _, err := c.Call(timed(t, 200*time.Millisecond), "remote", json.RawMessage(`{"hold":true}`)); err == nil {
					t.Fatal("a call the server held has no answer")
				}
			} else if _, err := c.Call(timed(t, 10*time.Second), "remote", nil); err != nil {
				t.Fatal(err)
			}
			tr := c.current().(*sseTransport)
			f.end(tc.say...)
			select {
			case <-tr.done:
			case <-time.After(5 * time.Second):
				t.Fatal("the stream ended and its transport never noticed")
			}
			reachable(t, c)
			if n := f.opened.Load(); n != 2 {
				t.Fatalf("opened %d streams, want one before the end and one after", n)
			}
		})
	}
}

// A server that no longer knows the endpoint its stream named refuses what is
// sent there with 404 and has acted on none of it. That is answer enough to
// open a fresh stream, whether or not the old one has been seen to end.
func TestAStreamWhoseEndpointIsForgottenIsOpenedAgain(t *testing.T) {
	f := newFading(t, http.StatusNotFound)
	c, err := Connect(context.Background(), ServerConfig{
		Name: "sse", Transport: SSE, URL: f.URL, Timeout: 5 * time.Second,
	}, f.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Call(timed(t, 10*time.Second), "remote", nil); err != nil {
		t.Fatal(err)
	}
	f.forget()
	reachable(t, c)
	if n := f.opened.Load(); n != 2 {
		t.Fatalf("opened %d streams, want one before the server forgot it and one after", n)
	}
	for deadline := time.Now().Add(5 * time.Second); f.closed.Load() != 1; time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the stream the server forgot was left open")
		}
	}
}
