// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

// pipe serves m on a pair of pipes and hands back a way to send a frame and
// read the lines that come back.
type pipe struct {
	in    *io.PipeWriter
	lines chan string
	done  chan error
}

func servePipe(t *testing.T, ctx context.Context, m *Manager) *pipe {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	p := &pipe{in: inW, lines: make(chan string, 16), done: make(chan error, 1)}
	go func() {
		err := m.ServeStdio(ctx, inR, outW)
		outW.Close()
		p.done <- err
	}()
	go func() {
		sc := bufio.NewScanner(outR)
		for sc.Scan() {
			p.lines <- sc.Text()
		}
		close(p.lines)
	}()
	t.Cleanup(func() { inW.Close() })
	return p
}

func (p *pipe) send(t *testing.T, frames ...string) {
	t.Helper()
	if _, err := io.WriteString(p.in, strings.Join(frames, "\n")+"\n"); err != nil {
		t.Fatal(err)
	}
}

func (p *pipe) next(t *testing.T) string {
	t.Helper()
	select {
	case line, open := <-p.lines:
		if !open {
			t.Fatal("the server stopped writing")
		}
		return line
	case <-time.After(5 * time.Second):
		t.Fatal("nothing came back")
	}
	return ""
}

// A tool whose annotation goose cannot read is still callable where nothing is
// mirrored, which a pipe never is. Where a client does mirror, it is not
// listed, rather than listed and refused on every call.
func TestAnAnnotationGooseCannotReadCostsOnlyTheChannelThatMirrors(t *testing.T) {
	ft := &fakeTransport{answer: func(out Outbound) (*response, error) {
		switch out.Method {
		case "server/discover":
			return &response{Error: &rpcError{Code: -32601, Message: "no"}}, nil
		case "initialize":
			return &response{Result: json.RawMessage(`{"protocolVersion":"2025-06-18","serverInfo":{"name":"x"}}`)}, nil
		case "tools/list":
			return &response{Result: json.RawMessage(`{"tools":[` +
				`{"name":"scale","inputSchema":{"type":"object","properties":{"factor":{"type":"number","x-mcp-header":"Factor"}}}},` +
				`{"name":"plain","inputSchema":{"type":"object"}}]}`)}, nil
		}
		return &response{Result: json.RawMessage(`{"content":[{"type":"text","text":"scaled"}]}`)}, nil
	}}
	c := newClient("up", ft, Options{})
	if err := c.connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Tools(context.Background()); err != nil {
		t.Fatal(err)
	}
	m := NewManager()
	defer m.Close()
	m.sources = append(m.sources, c)
	m.reindex()

	p := servePipe(t, context.Background(), m)
	p.send(t, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"scale","arguments":{"factor":2.5},`+modernMeta()+`}}`)
	if line := p.next(t); !strings.Contains(line, "scaled") {
		t.Fatalf("answer = %s: a pipe mirrors nothing, so there was nothing to refuse", line)
	}

	srv := httptest.NewServer(m.Handler())
	defer srv.Close()
	names := func(out map[string]any) []string {
		var got []string
		for _, tool := range out["result"].(map[string]any)["tools"].([]any) {
			got = append(got, tool.(map[string]any)["name"].(string))
		}
		return got
	}
	_, out := modernPost(t, srv, "tools/list", "",
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{`+modernMeta()+`}}`)
	if got := names(out); len(got) != 1 || got[0] != "plain" {
		t.Fatalf("tools = %q: a tool every mirroring call to would be refused was listed to a client that mirrors", got)
	}
	var legacy map[string]any
	if err := json.Unmarshal([]byte(rpc(t, srv, `{"jsonrpc":"2.0","id":3,"method":"tools/list"}`)), &legacy); err != nil {
		t.Fatal(err)
	}
	if got := names(legacy); len(got) != 2 {
		t.Fatalf("tools = %q: the older era mirrors nothing and can call both", got)
	}
}

// An id is the client's, and it comes back exactly as it was sent: a number
// past what a float holds is not rounded, and a string that reads as a number
// stays a string.
func TestAnIdComesBackAsItWasSent(t *testing.T) {
	m := NewManager()
	defer m.Close()
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "slow"},
		Call: func(ctx context.Context, _ json.RawMessage) (*Result, error) {
			Report(ctx, Progress{Done: 1})
			return &Result{Content: []llm.Part{&llm.Text{Text: "done"}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}

	p := servePipe(t, context.Background(), m)
	for _, id := range []string{"9007199254740993", "1727600000000000123"} {
		p.send(t, `{"jsonrpc":"2.0","id":`+id+`,"method":"ping"}`)
		if line := p.next(t); !strings.Contains(line, `"id":`+id+`,`) && !strings.HasSuffix(line, `"id":`+id+`}`) {
			t.Fatalf("answer = %s, want id %s", line, id)
		}
	}
	p.send(t, `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"slow","arguments":{},"_meta":{"progressToken":9007199254740993}}}`)
	if line := p.next(t); !strings.Contains(line, `"progressToken":9007199254740993`) {
		t.Fatalf("report = %s: the token the client chose did not come back as it chose it", line)
	}
	p.next(t)
	p.send(t, `{"jsonrpc":"2.0","id":"42","method":"subscriptions/listen","params":{"notifications":{"toolsListChanged":true},`+modernMeta()+`}}`)
	if line := p.next(t); !strings.Contains(line, `"`+metaSubscription+`":"42"`) {
		t.Fatalf("acknowledgement = %s: a string id was named as a number", line)
	}

	srv := httptest.NewServer(m.Handler())
	defer srv.Close()
	if body := rpc(t, srv, `{"jsonrpc":"2.0","id":9007199254740993,"method":"ping"}`); !strings.Contains(body, `"id":9007199254740993`) {
		t.Fatalf("answer = %s", body)
	}
}

// A mirrored integer is said as the body wrote it. Read through a float, one
// past 2^53 came out as its neighbour, so a client that mirrored it exactly was
// refused and an upstream was sent a header that disagreed with the body.
func TestAMirroredIntegerIsSaidAsTheBodyWroteIt(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{"account":{"type":"integer","x-mcp-header":"Account"}}}`)
	for args, want := range map[string]string{
		`{"account":9007199254740993}`:     "9007199254740993",
		`{"account":-9007199254740993}`:    "-9007199254740993",
		`{"account":9223372036854775808}`:  "9223372036854775808",
		`{"account":18446744073709551617}`: "18446744073709551617",
		`{"account":-0}`:                   "0",
		`{"account":42.0}`:                 "42",
		`{"account":1e3}`:                  "1000",
		`{"account":1.5}`:                  "",
	} {
		headers, err := paramHeaders(schema, json.RawMessage(args))
		if err != nil {
			t.Fatal(err)
		}
		if got := headers["Mcp-Param-Account"]; got != want {
			t.Fatalf("%s: header = %q, want %q", args, got, want)
		}
	}

	m := NewManager()
	defer m.Close()
	var called atomic.Int32
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "charge", Schema: schema},
		Call: func(context.Context, json.RawMessage) (*Result, error) {
			called.Add(1)
			return &Result{Content: []llm.Part{&llm.Text{Text: "charged"}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var sent []string
	served := m.Handler()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if header := r.Header.Get("Mcp-Param-Account"); header != "" {
			mu.Lock()
			sent = append(sent, header)
			mu.Unlock()
		}
		served.ServeHTTP(w, r)
	}))
	defer srv.Close()

	post := func(header string) int {
		body := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"charge",
			"arguments":{"account":9007199254740993},` + modernMeta() + `}}`
		req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("MCP-Protocol-Version", modernVersion)
		req.Header.Set("Mcp-Method", "tools/call")
		req.Header.Set("Mcp-Name", "charge")
		req.Header.Set("Mcp-Param-Account", header)
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
		return resp.StatusCode
	}
	if status := post("9007199254740992"); status != http.StatusBadRequest || called.Load() != 0 {
		t.Fatalf("status = %d: a header naming another account was served", status)
	}
	if status := post("9007199254740993"); status != http.StatusOK || called.Load() != 1 {
		t.Fatalf("status = %d: a client that mirrored the account exactly was refused", status)
	}

	c, err := Connect(context.Background(), ServerConfig{Name: "up", Transport: Streamable, URL: srv.URL}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Tools(context.Background()); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	sent = nil
	mu.Unlock()
	if _, err := c.Call(context.Background(), "charge", json.RawMessage(`{"account":9007199254740993}`)); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(sent) != 1 || sent[0] != "9007199254740993" {
		t.Fatalf("mirrored %q upstream, where the body said 9007199254740993", sent)
	}
}

// A cancellation can arrive right behind the call it names, before the call
// has started running. It still withdraws it.
func TestACancellationRightBehindItsCallWithdrawsIt(t *testing.T) {
	m := NewManager()
	defer m.Close()
	withdrawn := make(chan struct{}, 8)
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "slow"},
		Call: func(ctx context.Context, _ json.RawMessage) (*Result, error) {
			select {
			case <-ctx.Done():
				withdrawn <- struct{}{}
				return nil, ctx.Err()
			case <-time.After(2 * time.Second):
				return &Result{Content: []llm.Part{&llm.Text{Text: "finished"}}}, nil
			}
		},
	}); err != nil {
		t.Fatal(err)
	}

	p := servePipe(t, context.Background(), m)
	for i := range 5 {
		p.send(t,
			fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/call","params":{"name":"slow","arguments":{}}}`, i),
			fmt.Sprintf(`{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":%d}}`, i))
		select {
		case <-withdrawn:
		case <-time.After(time.Second):
			t.Fatalf("call %d: the cancellation right behind it was lost and the tool ran on", i)
		}
		p.send(t, fmt.Sprintf(`{"jsonrpc":"2.0","id":"ping-%d","method":"ping"}`, i))
		if line := p.next(t); !strings.Contains(line, fmt.Sprintf(`"ping-%d"`, i)) {
			t.Fatalf("frame = %s: the withdrawn call was answered", line)
		}
	}
}

// Whoever serves the pipe reads what a call left behind once serving stops,
// so it does not stop while a call is still settling.
func TestServingAPipeStopsOnlyOnceItsCallsHaveSettled(t *testing.T) {
	for _, how := range []string{"cancelled", "end of input"} {
		t.Run(how, func(t *testing.T) {
			m := NewManager()
			defer m.Close()
			started := make(chan struct{})
			var settled atomic.Bool
			if err := m.Serve("goose", Tool{
				Definition: llm.Tool{Name: "ask"},
				Call: func(ctx context.Context, _ json.RawMessage) (*Result, error) {
					close(started)
					<-ctx.Done()
					time.Sleep(150 * time.Millisecond)
					settled.Store(true)
					return nil, ctx.Err()
				},
			}); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p := servePipe(t, ctx, m)
			p.send(t, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"ask","arguments":{}}}`)
			<-started
			if how == "cancelled" {
				cancel()
			} else {
				p.in.Close()
			}
			select {
			case <-p.done:
			case <-time.After(3 * grace):
				t.Fatal("serving never stopped")
			}
			if !settled.Load() {
				t.Fatal("serving stopped while a call was still settling")
			}
		})
	}
}

// A stream opened to be told when the tools move spends its life waiting, and
// gives back whatever admitted it once it is open rather than holding a
// concurrency slot for as long as the client listens.
func TestAListeningStreamGivesBackWhatAdmittedIt(t *testing.T) {
	m := NewManager()
	defer m.Close()
	released := make(chan struct{})
	var once sync.Once
	admitted := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := httpcore.WithRelease(r.Context(), func() { once.Do(func() { close(released) }) })
		m.Handler().ServeHTTP(w, r.WithContext(ctx))
	})
	srv := httptest.NewServer(admitted)
	defer srv.Close()

	body := `{"jsonrpc":"2.0","id":1,"method":"subscriptions/listen","params":{
		"notifications":{"toolsListChanged":true},` + modernMeta() + `}}`
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(body))
	req.Header.Set("MCP-Protocol-Version", modernVersion)
	req.Header.Set("Mcp-Method", "subscriptions/listen")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil || !strings.Contains(line, "subscriptions/acknowledged") {
		t.Fatalf("first line = %q, %v", line, err)
	}
	select {
	case <-released:
	case <-time.After(2 * time.Second):
		t.Fatal("an open stream held on to what admitted it while it only waited")
	}
}

// A listening stream gives back the slot that admitted it, so how many can be
// open at once is bounded on its own: past the ceiling one is refused, and once
// the others close the next is served.
func TestListeningStreamsAreBoundedOnTheirOwn(t *testing.T) {
	m := NewManager()
	defer m.Close()
	h := m.Handler()
	listen := func(ctx context.Context) int {
		body := `{"jsonrpc":"2.0","id":1,"method":"subscriptions/listen","params":{
			"notifications":{"toolsListChanged":true},` + modernMeta() + `}}`
		req := httptest.NewRequestWithContext(ctx, http.MethodPost, "/", strings.NewReader(body))
		req.Header.Set("MCP-Protocol-Version", modernVersion)
		req.Header.Set("Mcp-Method", "subscriptions/listen")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		return w.Code
	}
	open := func() int {
		m.subs.Lock()
		defer m.subs.Unlock()
		return len(m.streams)
	}

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	defer wg.Wait()
	defer cancel()
	for range maxListeners {
		wg.Go(func() { listen(ctx) })
	}
	for deadline := time.Now().Add(5 * time.Second); open() < maxListeners; time.Sleep(10 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d streams opened", open(), maxListeners)
		}
	}
	if code := listen(timed(t, time.Second)); code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d: a stream past the ceiling was opened", code)
	}
	cancel()
	wg.Wait()
	if code := listen(timed(t, 100*time.Millisecond)); code != http.StatusOK {
		t.Fatalf("status = %d: the streams closed and the next was still refused", code)
	}
}
