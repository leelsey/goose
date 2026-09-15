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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

func stdioServer(t *testing.T) ServerConfig {
	t.Helper()
	path, err := filepath.Abs(filepath.Join("testdata", "fake-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	return ServerConfig{Name: "fake", Command: path, Env: os.Environ()}
}

func TestStdioHandshakeAndPaginatedTools(t *testing.T) {
	ctx := context.Background()
	c, err := Connect(ctx, stdioServer(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	info := c.Info()
	if info.Name != "fake" || info.ProtocolVersion != "2025-06-18" {
		t.Fatalf("info = %+v: the handshake result must not be thrown away", info)
	}
	if info.Instructions != "be careful" {
		t.Fatalf("instructions lost: %+v", info)
	}

	tools, err := c.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 2 || tools[0].Name != "first" || tools[1].Name != "second" {
		t.Fatalf("tools = %+v: a list that stops at the first page hides the rest", tools)
	}
}

func TestServerInitiatedFramesAreSteppedOver(t *testing.T) {
	c, err := Connect(context.Background(), stdioServer(t), nil)
	if err != nil {
		t.Fatalf("a server-initiated frame broke the correlation: %v", err)
	}
	_ = c.Close()
}

func TestNonTextToolResultsSurvive(t *testing.T) {
	ctx := context.Background()
	c, err := Connect(ctx, stdioServer(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	result, err := c.Call(ctx, "first", json.RawMessage(`{"q":1}`))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) != 2 {
		t.Fatalf("content = %+v", result.Content)
	}
	img, ok := result.Content[1].(*llm.Image)
	if !ok || string(img.Data) != "hi" || img.MIME != "image/png" {
		t.Fatalf("image = %+v: a picture flattened to text is destroyed", result.Content[1])
	}
}

func TestStreamableTransport(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		w.Header().Set("Mcp-Session-Id", "sess-1")
		switch {
		case strings.Contains(body, "server/discover"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
		case strings.Contains(body, "initialize"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"http-fake"}}}`))
		case strings.Contains(body, "tools/list"):
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}\n\n"))
			_, _ = w.Write([]byte("data: {\"jsonrpc\":\"2.0\",\"id\":\"" + id + "\",\"result\":{\"tools\":[{\"name\":\"remote\",\"inputSchema\":{\"type\":\"object\"}}]}}\n\n"))
		default:
			w.WriteHeader(http.StatusAccepted)
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	c, err := Connect(ctx, ServerConfig{Name: "http", Transport: Streamable, URL: srv.URL}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	tools, err := c.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "remote" {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestSSEEndpointMustStayOnTheSameOrigin(t *testing.T) {
	if _, err := resolveEndpoint("https://tools.example/sse", "https://elsewhere.example/messages"); err == nil {
		t.Fatal("following an endpoint to another host sends credentials where they were never meant to go")
	}
	got, err := resolveEndpoint("https://tools.example/sse", "/messages?session=1")
	if err != nil {
		t.Fatal(err)
	}
	if got != "https://tools.example/messages?session=1" {
		t.Fatalf("resolved = %q", got)
	}
	if _, err := resolveEndpoint("https://tools.example:443/sse", "https://tools.example/messages"); err != nil {
		t.Fatalf("an implicit port is the same origin: %v", err)
	}
}

func TestManagerQualifiesCollidingNames(t *testing.T) {
	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	cfg := stdioServer(t)
	if err := m.Add(ctx, cfg, nil); err != nil {
		t.Fatal(err)
	}
	second := cfg
	second.Name = "other"
	if err := m.Add(ctx, second, nil); err != nil {
		t.Fatal(err)
	}
	names := map[string]bool{}
	for _, tool := range m.Tools() {
		names[tool.Name] = true
	}
	if !names["fake__first"] || !names["other__first"] {
		t.Fatalf("tools = %v: first-wins hides a tool completely", names)
	}
	result, err := m.Call(ctx, "other__first", nil)
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(result.Content) != "done" {
		t.Fatalf("result = %+v", result.Content)
	}
}

func TestHandlerServesToolsBackToAClient(t *testing.T) {
	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, stdioServer(t), nil); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	post := func(body string) map[string]any {
		resp, err := srv.Client().Post(srv.URL+"/", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	out := post(`{"jsonrpc":"2.0","id":1,"method":"tools/list"}`)
	tools := out["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %v", tools)
	}
	out = post(`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"first","arguments":{}}}`)
	content := out["result"].(map[string]any)["content"].([]any)
	if len(content) != 2 || content[1].(map[string]any)["type"] != "image" {
		t.Fatalf("content = %v: the picture must reach the caller as a picture", content)
	}
	out = post(`{"jsonrpc":"2.0","id":3,"method":"nonsense"}`)
	if out["error"] == nil {
		t.Fatal("an unknown method must be refused")
	}
}

func TestHandlerRefusesABrowserOrigin(t *testing.T) {
	m := NewManager()
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	body := `{"jsonrpc":"2.0","id":"1","method":"tools/list"}`
	req, _ := http.NewRequest(http.MethodPost, srv.URL, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Origin", "https://an-unrelated-page.example")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(out), "browser origins") {
		t.Fatalf("body = %s", out)
	}

	req, _ = http.NewRequest(http.MethodPost, srv.URL, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err = srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ = io.ReadAll(resp.Body)
	if !strings.Contains(string(out), `"tools"`) {
		t.Fatalf("body = %s", out)
	}
}

func TestAWedgedServerCostsOneCallAndNoOther(t *testing.T) {
	path, err := filepath.Abs(filepath.Join("testdata", "wedged-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	c, err := Connect(context.Background(), ServerConfig{
		Name: "wedged", Command: path, Env: os.Environ(), Timeout: time.Second,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	for i := range 3 {
		start := time.Now()
		if _, err := c.Tools(context.Background()); err == nil {
			t.Fatalf("call %d: a server that never answers has not answered", i)
		}
		if took := time.Since(start); took > 5*time.Second {
			t.Fatalf("call %d took %s: the timeout the client declares must fire", i, took)
		}
	}
}

func TestAWaitingCallHonoursItsContext(t *testing.T) {
	path, err := filepath.Abs(filepath.Join("testdata", "wedged-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	c, err := Connect(context.Background(), ServerConfig{
		Name: "wedged", Command: path, Env: os.Environ(), Timeout: time.Hour,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := c.Tools(ctx); err == nil {
		t.Fatal("a cancelled call has no answer")
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Fatalf("took %s: the caller's cancellation must reach a waiting call", took)
	}
}

func TestToolListIsRefreshedWhileTheProxyRuns(t *testing.T) {
	path, err := filepath.Abs(filepath.Join("testdata", "changing-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	m := NewManager()
	defer m.Close()
	ctx := context.Background()
	if err := m.Add(ctx, ServerConfig{Name: "changing", Command: path, Env: os.Environ()}, nil); err != nil {
		t.Fatal(err)
	}
	if got := m.Tools(); len(got) != 1 {
		t.Fatalf("tools = %+v, want the one the server started with", got)
	}

	m.Watch(ctx, 20*time.Millisecond, nil)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if got := m.Tools(); len(got) == 2 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("tools = %+v: a server that gained a tool is still advertised without it", m.Tools())
}

func sseServer(t *testing.T, wedge bool) *httptest.Server {
	t.Helper()
	answers := make(chan string, 16)
	var srv *httptest.Server
	srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			fmt.Fprintf(w, "event: endpoint\ndata: %s/rpc\n\n", srv.URL)
			if flusher != nil {
				flusher.Flush()
			}
			for {
				select {
				case <-r.Context().Done():
					return
				case answer := <-answers:
					fmt.Fprintf(w, "data: %s\n\n", answer)
					if flusher != nil {
						flusher.Flush()
					}
				}
			}
		}
		body, _ := io.ReadAll(r.Body)
		var req struct {
			ID     string `json:"id"`
			Method string `json:"method"`
		}
		_ = json.Unmarshal(body, &req)
		w.WriteHeader(http.StatusAccepted)
		if req.Method == "" || wedge && req.Method != "initialize" {
			return
		}
		switch req.Method {
		case "initialize":
			answers <- fmt.Sprintf(`{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"sse"}}}`, req.ID)
		case "tools/list":
			answers <- `{"jsonrpc":"2.0","method":"notifications/message","params":{}}`
			answers <- fmt.Sprintf(`{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"remote","inputSchema":{"type":"object"}}]}}`, req.ID)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestSSETransportAnswersAndStepsOverServerFrames(t *testing.T) {
	srv := sseServer(t, false)
	c, err := Connect(context.Background(), ServerConfig{
		Name: "sse", Transport: SSE, URL: srv.URL, Timeout: 5 * time.Second,
	}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if c.Info().Name != "sse" {
		t.Fatalf("info = %+v", c.Info())
	}
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "remote" {
		t.Fatalf("tools = %+v", tools)
	}
}

func TestSSETransportGivesUpOnAServerThatStopsAnswering(t *testing.T) {
	srv := sseServer(t, true)
	c, err := Connect(context.Background(), ServerConfig{
		Name: "sse", Transport: SSE, URL: srv.URL, Timeout: 300 * time.Millisecond,
	}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for i := range 3 {
		start := time.Now()
		if _, err := c.Tools(context.Background()); err == nil {
			t.Fatalf("call %d: a server that never answers has not answered", i)
		}
		if took := time.Since(start); took > 3*time.Second {
			t.Fatalf("call %d took %s: the timeout the client declares must fire", i, took)
		}
	}
}

func TestSSERefusesAnEndpointOnAnotherOrigin(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "event: endpoint\ndata: https://an-unrelated-host.example/rpc\n\n")
	}))
	defer srv.Close()
	_, err := Connect(context.Background(), ServerConfig{
		Name: "sse", Transport: SSE, URL: srv.URL, Timeout: time.Second,
	}, srv.Client())
	if err == nil {
		t.Fatal("followed an endpoint to a host the stream does not serve")
	}
	if !strings.Contains(err.Error(), "same origin") {
		t.Fatalf("error = %v", err)
	}
}

func TestTheMountedEndpointServesItsOwnPath(t *testing.T) {
	m := NewManager()
	outer := http.NewServeMux()
	outer.Handle("/mcp", http.StripPrefix("/mcp", m.Handler()))
	outer.Handle("/mcp/", http.StripPrefix("/mcp", m.Handler()))
	srv := httptest.NewServer(outer)
	defer srv.Close()

	for _, path := range []string{"/mcp", "/mcp/"} {
		req, _ := http.NewRequest(http.MethodPost, srv.URL+path,
			strings.NewReader(`{"jsonrpc":"2.0","id":"1","method":"tools/list"}`))
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}}
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		out, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("POST %s = %d, want it served where it was asked for", path, resp.StatusCode)
		}
		if !strings.Contains(string(out), `"tools"`) {
			t.Fatalf("POST %s body = %s", path, out)
		}
	}
}

func TestAnAudioResultKeepsItsKind(t *testing.T) {
	parts := decodeContent([]json.RawMessage{
		json.RawMessage(`{"type":"audio","data":"AAEC","mimeType":"audio/wav"}`),
		json.RawMessage(`{"type":"image","data":"AAEC","mimeType":"image/png"}`),
	})
	out := encodeContent(parts)
	if len(out) != 2 {
		t.Fatalf("blocks = %d", len(out))
	}
	if !strings.Contains(string(out[0]), `"type":"audio"`) {
		t.Errorf("audio came back as %s", out[0])
	}
	if !strings.Contains(string(out[1]), `"type":"image"`) {
		t.Errorf("image came back as %s", out[1])
	}
}

func TestTheAgreedProtocolVersionTravelsOnEveryRequest(t *testing.T) {
	var sent []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		sent = append(sent, r.Header.Get("Mcp-Method")+" "+r.Header.Get("MCP-Protocol-Version"))
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(body, "server/discover"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
		case strings.Contains(body, "initialize"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-03-26","serverInfo":{"name":"http-fake"}}}`))
		default:
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[]}}`))
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	c, err := Connect(ctx, ServerConfig{Name: "http", Transport: Streamable, URL: srv.URL}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if _, err := c.Tools(ctx); err != nil {
		t.Fatal(err)
	}
	if len(sent) < 3 {
		t.Fatalf("requests = %v", sent)
	}
	if sent[0] != "server/discover "+modernVersion {
		t.Errorf("the probe carried %q", sent[0])
	}
	if sent[1] != "initialize "+protocolVersion {
		t.Errorf("the handshake carried %q, want the version goose speaks", sent[1])
	}
	if last := sent[len(sent)-1]; last != "tools/list 2025-03-26" {
		t.Errorf("a later request carried %q, want the agreed version", last)
	}
}

func requestID(body string) string {
	var req struct {
		ID json.RawMessage `json:"id"`
	}
	if json.Unmarshal([]byte(body), &req) != nil {
		return ""
	}
	return strings.Trim(string(req.ID), `"`)
}

func TestAToolsStructuredHalfSurvivesBothWays(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(string(raw), "initialize"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"1","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"f"}}}`))
		case strings.Contains(string(raw), "tools/list"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"2","result":{"tools":[{"name":"t",
				"inputSchema":{"type":"object"},"outputSchema":{"type":"object","properties":{"n":{"type":"number"}}}}]}}`))
		default:
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"3","result":{"content":[{"type":"text","text":"42"}],
				"structuredContent":{"n":42}}}`))
		}
	}))
	defer srv.Close()

	m := NewManager()
	defer m.Close()
	if err := m.Add(context.Background(), ServerConfig{Name: "s", Transport: Streamable, URL: srv.URL}, srv.Client()); err != nil {
		t.Fatal(err)
	}

	front := httptest.NewServer(m.Handler())
	defer front.Close()

	list := rpc(t, front, `{"jsonrpc":"2.0","id":"1","method":"tools/list"}`)
	if !strings.Contains(list, `"outputSchema"`) {
		t.Errorf("the schema a client validates against was dropped:\n%s", list)
	}
	call := rpc(t, front, `{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"t","arguments":{}}}`)
	if !strings.Contains(call, `"structuredContent"`) || !strings.Contains(call, `"n":42`) {
		t.Errorf("the structured half of the answer was dropped:\n%s", call)
	}
}

func rpc(t *testing.T, srv *httptest.Server, body string) string {
	t.Helper()
	resp, err := srv.Client().Post(srv.URL, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return string(out)
}

func TestClosingATransportEndsTheSession(t *testing.T) {
	deleted := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			deleted <- r.Header.Get("Mcp-Session-Id")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Mcp-Session-Id", "sess-1")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"1","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"f"}}}`))
	}))
	defer srv.Close()

	c, err := Connect(context.Background(), ServerConfig{Name: "s", Transport: Streamable, URL: srv.URL}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case id := <-deleted:
		if id != "sess-1" {
			t.Fatalf("session = %q", id)
		}
	default:
		t.Fatal("the session was left open")
	}
}

func TestHeadersAreNotCarriedToARemoteServerInTheClear(t *testing.T) {
	_, err := Connect(context.Background(), ServerConfig{
		Name: "s", Transport: Streamable, URL: "http://tools.example/mcp",
		Headers: map[string]string{"Authorization": "Bearer secret"},
	}, nil)
	if err == nil {
		t.Fatal("a bearer token was sent across a network in the clear")
	}
	if !strings.Contains(err.Error(), "https") {
		t.Fatalf("err = %v", err)
	}
	if err := credentialsOverTLS("http://tools.example/mcp", nil); err != nil {
		t.Fatalf("a plaintext server with nothing to leak was refused: %v", err)
	}
	if err := credentialsOverTLS("http://127.0.0.1:3000/mcp", map[string]string{"Authorization": "x"}); err != nil {
		t.Fatalf("a server on this machine was refused: %v", err)
	}
}

func TestAnOversizeBodyIsTooLargeNotMalformed(t *testing.T) {
	m := NewManager()
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	big := `{"jsonrpc":"2.0","id":"1","method":"tools/list","params":{"pad":"` +
		strings.Repeat("x", maxRPCBody) + `"}}`
	resp, err := srv.Client().Post(srv.URL, "application/json", strings.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		out, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, out)
	}
}

func TestAToolServerOutlivesTheContextThatDialledIt(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, ServerConfig{
		Name: "s", Transport: Stdio, Command: "sh",
		Args: []string{"-c", stdioEcho},
	}, nil); err != nil {
		t.Skipf("no shell to spawn: %v", err)
	}
	cancel()

	tools := m.Tools()
	if len(tools) == 0 {
		t.Fatal("no tools")
	}
	if _, err := m.Call(context.Background(), tools[0].Name, nil); err != nil {
		t.Fatalf("the server died with the context that dialled it: %v", err)
	}
}

const stdioEcho = `while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":"\{0,1\}\([^,"}]*\)"\{0,1\}.*/\1/p')
  case "$line" in
    *server/discover*) printf '{"jsonrpc":"2.0","id":"%s","error":{"code":-32601,"message":"unknown method"}}\n' "$id" ;;
    *initialize*) printf '{"jsonrpc":"2.0","id":"%s","result":{"protocolVersion":"2025-06-18","serverInfo":{"name":"echo"}}}\n' "$id" ;;
    *tools/list*) printf '{"jsonrpc":"2.0","id":"%s","result":{"tools":[{"name":"echo","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
    *tools/call*) printf '{"jsonrpc":"2.0","id":"%s","result":{"content":[{"type":"text","text":"ok"}]}}\n' "$id" ;;
  esac
done`

func TestServersAreListedByTheNameThatQualifiesTheirTools(t *testing.T) {
	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	cfg := stdioServer(t)
	cfg.Name = "files"
	if err := m.Add(ctx, cfg, nil); err != nil {
		t.Fatal(err)
	}
	got := m.Servers()
	if len(got) != 1 {
		t.Fatalf("servers = %+v", got)
	}
	if got[0].Name != "files" {
		t.Fatalf("name = %q: the configured name is what Call and the tool prefix use", got[0].Name)
	}
	if got[0].Info.Name != "fake" || got[0].Info.ProtocolVersion != "2025-06-18" {
		t.Fatalf("info = %+v: what the server said about itself is lost", got[0].Info)
	}
	if got[0].Tools != len(m.Tools()) || got[0].Tools == 0 {
		t.Fatalf("tools = %d, manager offers %d", got[0].Tools, len(m.Tools()))
	}
}

func TestANotificationIsNotAnswered(t *testing.T) {
	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, stdioServer(t), nil); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	for _, body := range []string{
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"1"}}`,
		`{"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1}}`,
	} {
		resp, err := srv.Client().Post(srv.URL, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		out, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if len(strings.TrimSpace(string(out))) != 0 {
			t.Fatalf("%s was answered with %s", body, out)
		}
		if resp.StatusCode != http.StatusAccepted {
			t.Fatalf("%s: status = %d", body, resp.StatusCode)
		}
	}

	resp, err := srv.Client().Post(srv.URL, "application/json",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"nope"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out["error"] == nil {
		t.Fatalf("body = %v: a request is answered", out)
	}
}

func TestTheHandshakeNamesTheVersionItWasGiven(t *testing.T) {
	was := Version()
	t.Cleanup(func() { SetVersion(was) })
	SetVersion("9.9.9-test")

	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	cfg := stdioServer(t)
	if err := m.Add(ctx, cfg, nil); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(m.Handler())
	defer srv.Close()
	resp, err := srv.Client().Post(srv.URL, "application/json",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out struct {
		Result struct {
			ServerInfo struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"serverInfo"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Result.ServerInfo.Name != "goose" || out.Result.ServerInfo.Version != "9.9.9-test" {
		t.Fatalf("serverInfo = %+v", out.Result.ServerInfo)
	}

	SetVersion(was)
	if Version() == "" {
		t.Fatal("the default must still name something")
	}
}

func TestToolsServedHereAreNotOfferedToAModel(t *testing.T) {
	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, stdioServer(t), nil); err != nil {
		t.Fatal(err)
	}
	var got string
	err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "ask", Description: "ask a model"},
		Call: func(_ context.Context, args json.RawMessage) (*Result, error) {
			got = string(args)
			return &Result{Content: []llm.Part{&llm.Text{Text: "answered"}}}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	var served, offered []string
	for _, tool := range m.Tools() {
		served = append(served, tool.Name)
		if tool.Name == "ask" && string(tool.Schema) == "" {
			t.Fatal("a tool with no schema reaches a backend that rejects one")
		}
	}
	for _, tool := range m.Proxied() {
		offered = append(offered, tool.Name)
	}
	if len(served) != 3 {
		t.Fatalf("the endpoint serves %v; it must offer the servers' tools and its own", served)
	}
	for _, name := range offered {
		if name == "ask" {
			t.Fatalf("a model was offered the proxy's own tool: %v", offered)
		}
	}

	result, err := m.Call(ctx, "ask", json.RawMessage(`{"prompt":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(result.Content) != "answered" || got != `{"prompt":"x"}` {
		t.Fatalf("result = %+v, arguments = %s", result.Content, got)
	}
}

func TestAServedToolNeedsSomethingToRun(t *testing.T) {
	m := NewManager()
	defer m.Close()
	if err := m.Serve("goose", Tool{Definition: llm.Tool{Name: "ask"}}); err == nil {
		t.Fatal("a tool with nothing behind it is a name the caller cannot use")
	}
	if err := m.Serve("goose", Tool{Call: func(context.Context, json.RawMessage) (*Result, error) { return nil, nil }}); err == nil {
		t.Fatal("a tool with no name is one nothing can call")
	}
}

func TestTheProtocolIsServedOnAPipe(t *testing.T) {
	m := NewManager()
	defer m.Close()
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "ask"},
		Call: func(_ context.Context, args json.RawMessage) (*Result, error) {
			return &Result{Content: []llm.Part{&llm.Text{Text: "answered"}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}

	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() {
		err := m.ServeStdio(context.Background(), inR, outW)
		outW.Close()
		done <- err
	}()
	dec := json.NewDecoder(outR)
	answer := func(frame string) map[string]any {
		t.Helper()
		if _, err := fmt.Fprintln(inW, frame); err != nil {
			t.Fatal(err)
		}
		var out map[string]any
		if err := dec.Decode(&out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	out := answer(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`)
	info := out["result"].(map[string]any)["serverInfo"].(map[string]any)
	if info["name"] != "goose" {
		t.Fatalf("initialize = %v", out)
	}
	if _, err := fmt.Fprintln(inW, `{"jsonrpc":"2.0","method":"notifications/initialized"}`); err != nil {
		t.Fatal(err)
	}
	out = answer(`{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ask","arguments":{"prompt":"x"}}}`)
	if out["id"].(float64) != 2 {
		t.Fatalf("a notification was answered: %v", out)
	}
	content := out["result"].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["text"] != "answered" {
		t.Fatalf("call = %v", out)
	}
	out = answer(`{"not":"json-rpc"` + "\n")
	if out["error"] == nil {
		t.Fatalf("a malformed frame must be refused: %v", out)
	}

	inW.Close()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("a client closing its side is not a failure: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the server did not stop when its client went away")
	}
}

func TestTheHandshakeAgreesToTheRevisionTheClientAsked(t *testing.T) {
	m := NewManager()
	defer m.Close()
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	ask := func(params string) string {
		resp, err := srv.Client().Post(srv.URL+"/", "application/json",
			strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":`+params+`}`))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			t.Fatal(err)
		}
		result, ok := out["result"].(map[string]any)
		if !ok {
			t.Fatalf("initialize = %v", out)
		}
		return result["protocolVersion"].(string)
	}

	if v := ask(`{"protocolVersion":"2024-11-05"}`); v != "2024-11-05" {
		t.Fatalf("version = %s: a revision goose speaks is agreed to, not overruled", v)
	}
	if v := ask(`{"protocolVersion":"1999-01-01"}`); v != protocolVersion {
		t.Fatalf("version = %s", v)
	}
	if v := ask(`{}`); v != protocolVersion {
		t.Fatalf("version = %s", v)
	}
}

func TestTwoSourcesCannotShareAName(t *testing.T) {
	ctx := context.Background()
	m := NewManager()
	defer m.Close()
	if err := m.Serve("files", Tool{
		Definition: llm.Tool{Name: "ask"},
		Call:       func(context.Context, json.RawMessage) (*Result, error) { return &Result{}, nil },
	}); err != nil {
		t.Fatal(err)
	}
	cfg := stdioServer(t)
	cfg.Name = "files"
	if err := m.Add(ctx, cfg, nil); err == nil {
		t.Fatal("a second source under the same name hides one of them")
	}
	if err := m.Serve("files", Tool{
		Definition: llm.Tool{Name: "other"},
		Call:       func(context.Context, json.RawMessage) (*Result, error) { return &Result{}, nil },
	}); err == nil {
		t.Fatal("a second set of served tools under the same name hides one of them")
	}
}

func modernPost(t *testing.T, srv *httptest.Server, method, name, body string) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("MCP-Protocol-Version", modernVersion)
	req.Header.Set("Mcp-Method", method)
	if name != "" {
		req.Header.Set("Mcp-Name", name)
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

func modernMeta() string {
	return `"_meta":{"io.modelcontextprotocol/protocolVersion":"` + modernVersion +
		`","io.modelcontextprotocol/clientCapabilities":{},` +
		`"io.modelcontextprotocol/clientInfo":{"name":"probe","version":"1"}}`
}

func TestAClientOfTheCurrentRevisionIsServed(t *testing.T) {
	m := NewManager()
	defer m.Close()
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "ask", Description: "ask a model"},
		Call: func(_ context.Context, args json.RawMessage) (*Result, error) {
			return &Result{Content: []llm.Part{&llm.Text{Text: "answered"}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	status, out := modernPost(t, srv, "server/discover", "",
		`{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{`+modernMeta()+`}}`)
	if status != http.StatusOK {
		t.Fatalf("status = %d: %v", status, out)
	}
	result, ok := out["result"].(map[string]any)
	if !ok {
		t.Fatalf("discover = %v", out)
	}
	if result["resultType"] != "complete" {
		t.Errorf("result type = %v: a result of this era says what kind it is", result["resultType"])
	}
	versions, _ := result["supportedVersions"].([]any)
	if len(versions) == 0 || versions[0] != modernVersion {
		t.Fatalf("supported = %v", result["supportedVersions"])
	}
	meta, _ := result["_meta"].(map[string]any)
	if meta == nil || meta[metaServerInfo] == nil {
		t.Errorf("result = %v: the server names itself where there is no handshake to have named it in", result)
	}

	_, out = modernPost(t, srv, "tools/list", "",
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{`+modernMeta()+`}}`)
	tools := out["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", tools)
	}
	_, out = modernPost(t, srv, "tools/call", "ask",
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ask","arguments":{},`+modernMeta()+`}}`)
	content := out["result"].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["text"] != "answered" {
		t.Fatalf("call = %v", out)
	}
}

func TestTheCurrentRevisionsRefusalsCarryTheirStatus(t *testing.T) {
	m := NewManager()
	defer m.Close()
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	status, out := modernPost(t, srv, "tools/list", "",
		`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{
			"io.modelcontextprotocol/protocolVersion":"1999-01-01",
			"io.modelcontextprotocol/clientCapabilities":{}}}}`)
	refusal, _ := out["error"].(map[string]any)
	if status != http.StatusBadRequest || refusal == nil || refusal["code"].(float64) != codeUnsupportedVersion {
		t.Fatalf("status = %d, body = %v", status, out)
	}
	data, _ := refusal["data"].(map[string]any)
	if supported, _ := data["supported"].([]any); len(supported) == 0 {
		t.Fatalf("refusal = %v: a client told only that it was wrong has nothing to retry with", refusal)
	}

	status, out = modernPost(t, srv, "tools/list", "",
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{"_meta":{
			"io.modelcontextprotocol/protocolVersion":"`+modernVersion+`"}}}`)
	if status != http.StatusBadRequest || out["error"].(map[string]any)["code"].(float64) != -32602 {
		t.Fatalf("status = %d, body = %v", status, out)
	}

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(
		`{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{`+modernMeta()+`}}`))
	req.Header.Set("MCP-Protocol-Version", modernVersion)
	req.Header.Set("Mcp-Method", "tools/call")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	var mismatch map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&mismatch)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest ||
		mismatch["error"].(map[string]any)["code"].(float64) != codeHeaderMismatch {
		t.Fatalf("status = %d, body = %v", resp.StatusCode, mismatch)
	}

	status, out = modernPost(t, srv, "resources/list", "",
		`{"jsonrpc":"2.0","id":4,"method":"resources/list","params":{`+modernMeta()+`}}`)
	if status != http.StatusNotFound || out["error"].(map[string]any)["code"].(float64) != -32601 {
		t.Fatalf("status = %d, body = %v", status, out)
	}
}

func TestABrowserOriginIsRefusedWithAStatus(t *testing.T) {
	m := NewManager()
	defer m.Close()
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(
		`{"jsonrpc":"2.0","id":1,"method":"tools/list"}`))
	req.Header.Set("Origin", "https://evil.test")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("status = %d: a 200 carrying bad news is a page that reads the tools anyway", resp.StatusCode)
	}
}

func TestGooseConnectsToGooseOnTheCurrentRevision(t *testing.T) {
	ctx := context.Background()
	server := NewManager()
	defer server.Close()
	if err := server.Serve("goose", Tool{
		Definition: llm.Tool{Name: "ask"},
		Call: func(_ context.Context, args json.RawMessage) (*Result, error) {
			return &Result{Content: []llm.Part{&llm.Text{Text: "answered " + string(args)}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(server.Handler())
	defer srv.Close()

	c, err := Connect(ctx, ServerConfig{Name: "loop", Transport: Streamable, URL: srv.URL}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if v := c.Info().ProtocolVersion; v != modernVersion {
		t.Fatalf("agreed %q: two ends that both speak the newer era must not settle for the older", v)
	}
	tools, err := c.Tools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 || tools[0].Name != "ask" {
		t.Fatalf("tools = %+v", tools)
	}
	result, err := c.Call(ctx, "ask", json.RawMessage(`{"prompt":"x"}`))
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(result.Content) != `answered {"prompt":"x"}` {
		t.Fatalf("result = %+v", result.Content)
	}
}

func TestASilentProbeFallsBackToTheHandshake(t *testing.T) {
	path, err := filepath.Abs(filepath.Join("testdata", "silent-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	transport, err := startStdio(context.Background(), ServerConfig{
		Name: "silent", Command: path, Env: os.Environ(),
	}, 0)
	if err != nil {
		t.Fatal(err)
	}
	c := newClient("silent", transport, Options{Timeout: 30 * time.Second, Probe: 200 * time.Millisecond})
	defer c.Close()

	started := time.Now()
	if err := c.connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	if took := time.Since(started); took > 5*time.Second {
		t.Fatalf("the handshake waited %s on a silence", took)
	}
	if c.Info().Name != "silent" {
		t.Fatalf("info = %+v", c.Info())
	}
}

func TestAToolThatNeedsTheClientIsNotReportedAsAnAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(body, "server/discover"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"resultType":"complete",
				"supportedVersions":["` + modernVersion + `"],"capabilities":{"tools":{}},
				"_meta":{"io.modelcontextprotocol/serverInfo":{"name":"modern","version":"1"}}}}`))
		case strings.Contains(body, "tools/list"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"resultType":"complete",
				"tools":[{"name":"ask","inputSchema":{"type":"object"}}]}}`))
		default:
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"resultType":"input_required",
				"inputRequests":{"who":{"method":"elicitation/create","params":{}}}}}`))
		}
	}))
	defer srv.Close()

	ctx := context.Background()
	c, err := Connect(ctx, ServerConfig{Name: "modern", Transport: Streamable, URL: srv.URL}, srv.Client())
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if c.Info().Name != "modern" || c.Info().ProtocolVersion != modernVersion {
		t.Fatalf("info = %+v", c.Info())
	}
	if _, err := c.Call(ctx, "ask", nil); err == nil {
		t.Fatal("a request for input is not a tool result, and reporting it as one reports a tool that ran")
	}
}

func TestAnArgumentAToolAsksToMirrorIsMirrored(t *testing.T) {
	schema := json.RawMessage(`{"type":"object","properties":{
		"region":{"type":"string","x-mcp-header":"Region"},
		"rows":{"type":"integer","x-mcp-header":"Rows"},
		"dry":{"type":"boolean","x-mcp-header":"Dry"},
		"where":{"type":"object","properties":{"zone":{"type":"string","x-mcp-header":"Zone"}}},
		"query":{"type":"string"}}}`)

	headers, err := paramHeaders(schema, json.RawMessage(
		`{"region":"us-west1","rows":42,"dry":false,"where":{"zone":"a"},"query":"SELECT 1"}`))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"Mcp-Param-Region": "us-west1", "Mcp-Param-Rows": "42",
		"Mcp-Param-Dry": "false", "Mcp-Param-Zone": "a",
	}
	if len(headers) != len(want) {
		t.Fatalf("headers = %v", headers)
	}
	for name, value := range want {
		if headers[name] != value {
			t.Fatalf("%s = %q, want %q", name, headers[name], value)
		}
	}

	headers, err = paramHeaders(schema, json.RawMessage(`{"query":"SELECT 1"}`))
	if err != nil || len(headers) != 0 {
		t.Fatalf("headers = %v, err = %v", headers, err)
	}

	headers, _ = paramHeaders(schema, json.RawMessage(`{"region":"line1\nline2"}`))
	if got := headers["Mcp-Param-Region"]; !strings.HasPrefix(got, "=?base64?") || headerText(got) != "line1\nline2" {
		t.Fatalf("encoded = %q", got)
	}
}

func TestAToolThatAsksForAnImpossibleMirrorIsRefused(t *testing.T) {
	for _, schema := range []string{
		`{"type":"object","properties":{"a":{"type":"number","x-mcp-header":"A"}}}`,
		`{"type":"object","properties":{"a":{"type":"string","x-mcp-header":"bad name"}}}`,
		`{"type":"object","properties":{"a":{"type":"string","x-mcp-header":""}}}`,
		`{"type":"object","properties":{"a":{"type":"string","x-mcp-header":"X"},"b":{"type":"string","x-mcp-header":"x"}}}`,
		`{"type":"object","properties":{"a":{"x-mcp-header":"A"}}}`,
	} {
		if _, err := annotated(json.RawMessage(schema)); err == nil {
			t.Fatalf("accepted %s", schema)
		}
	}
	if _, err := annotated(json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`)); err != nil {
		t.Fatalf("a schema that asks for nothing is not an error: %v", err)
	}
}

func TestAMirroredArgumentIsHeldToTheBody(t *testing.T) {
	m := NewManager()
	defer m.Close()
	var called int
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{
			Name:   "query",
			Schema: json.RawMessage(`{"type":"object","properties":{"region":{"type":"string","x-mcp-header":"Region"}}}`),
		},
		Call: func(context.Context, json.RawMessage) (*Result, error) {
			called++
			return &Result{Content: []llm.Part{&llm.Text{Text: "ok"}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	body := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"query",
		"arguments":{"region":"us-west1"},` + modernMeta() + `}}`
	post := func(header string) (int, map[string]any) {
		req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(body))
		req.Header.Set("MCP-Protocol-Version", modernVersion)
		req.Header.Set("Mcp-Method", "tools/call")
		req.Header.Set("Mcp-Name", "query")
		if header != "" {
			req.Header.Set("Mcp-Param-Region", header)
		}
		resp, err := srv.Client().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&out)
		return resp.StatusCode, out
	}

	if status, out := post("eu-west1"); status != http.StatusBadRequest ||
		out["error"].(map[string]any)["code"].(float64) != codeHeaderMismatch {
		t.Fatalf("a header that disagreed with the body was served: %d %v", status, out)
	}
	if status, out := post(""); status != http.StatusBadRequest {
		t.Fatalf("a missing header the body needed was served: %d %v", status, out)
	}
	if called != 0 {
		t.Fatal("a refused call reached the tool")
	}
	if status, _ := post("us-west1"); status != http.StatusOK || called != 1 {
		t.Fatalf("status = %d, calls = %d", status, called)
	}
}

func TestAToolDefinitionCrossesTheProxyWhole(t *testing.T) {
	ctx := context.Background()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.Contains(body, "server/discover"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
		case strings.Contains(body, "initialize"):
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-11-25","serverInfo":{"name":"up"}}}`))
		default:
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[{"name":"look",
				"title":"Look Something Up","description":"d","inputSchema":{"type":"object"},
				"icons":[{"src":"https://tools.test/icon.png","mimeType":"image/png"}],
				"annotations":{"readOnlyHint":true}}]}}`))
		}
	}))
	defer upstream.Close()

	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, ServerConfig{Name: "up", Transport: Streamable, URL: upstream.URL}, upstream.Client()); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	_, out := modernPost(t, srv, "tools/list", "",
		`{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{`+modernMeta()+`}}`)
	tools := out["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %v", tools)
	}
	tool := tools[0].(map[string]any)
	switch {
	case tool["name"] != "look":
		t.Fatalf("tool = %v", tool)
	case tool["title"] != "Look Something Up":
		t.Fatalf("the title did not survive: %v", tool)
	case tool["icons"] == nil:
		t.Fatalf("the icons did not survive: %v", tool)
	case tool["annotations"] == nil:
		t.Fatalf("the annotations did not survive: %v", tool)
	}
}

func TestACancelledCallOnAPipeIsStopped(t *testing.T) {
	m := NewManager()
	defer m.Close()
	started := make(chan struct{})
	stopped := make(chan struct{})
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "slow"},
		Call: func(ctx context.Context, _ json.RawMessage) (*Result, error) {
			close(started)
			<-ctx.Done()
			close(stopped)
			return nil, ctx.Err()
		},
	}); err != nil {
		t.Fatal(err)
	}

	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	done := make(chan error, 1)
	go func() {
		err := m.ServeStdio(context.Background(), inR, outW)
		outW.Close()
		done <- err
	}()

	if _, err := fmt.Fprintln(inW, `{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"slow","arguments":{}}}`); err != nil {
		t.Fatal(err)
	}
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the tool never ran")
	}
	if _, err := fmt.Fprintln(inW, `{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":7}}`); err != nil {
		t.Fatal(err)
	}
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Fatal("the tool kept working on an answer nobody was waiting for")
	}

	frames := make(chan map[string]any, 1)
	go func() {
		var out map[string]any
		if json.NewDecoder(outR).Decode(&out) == nil {
			frames <- out
		}
		close(frames)
	}()
	if _, err := fmt.Fprintln(inW, `{"jsonrpc":"2.0","id":8,"method":"ping"}`); err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-frames:
		if frame == nil || fmt.Sprint(frame["id"]) != "8" {
			t.Fatalf("frame = %v: the cancelled call was answered anyway", frame)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the ping behind the cancelled call was not answered")
	}

	inW.Close()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the server did not stop")
	}
}

func TestAChangedToolSetReachesAnOpenStream(t *testing.T) {
	m := NewManager()
	defer m.Close()
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	body := `{"jsonrpc":"2.0","id":1,"method":"subscriptions/listen","params":{
		"notifications":{"toolsListChanged":true,"resourcesListChanged":true},` + modernMeta() + `}}`
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(body))
	req.Header.Set("MCP-Protocol-Version", modernVersion)
	req.Header.Set("Mcp-Method", "subscriptions/listen")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content type = %q: the answer to this request is the stream", ct)
	}

	frames := make(chan map[string]any, 4)
	go func() {
		sc := bufio.NewScanner(resp.Body)
		for sc.Scan() {
			line := strings.TrimPrefix(sc.Text(), "data: ")
			if line == sc.Text() {
				continue
			}
			var frame map[string]any
			if json.Unmarshal([]byte(line), &frame) == nil {
				frames <- frame
			}
		}
		close(frames)
	}()

	select {
	case frame := <-frames:
		if frame["method"] != "notifications/subscriptions/acknowledged" {
			t.Fatalf("first frame = %v", frame)
		}
		agreed := frame["params"].(map[string]any)["notifications"].(map[string]any)
		if agreed["toolsListChanged"] != true || agreed["resourcesListChanged"] != nil {
			t.Fatalf("agreed = %v: a client told yes to something that never arrives waits forever", agreed)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the stream said nothing")
	}

	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "ask"},
		Call:       func(context.Context, json.RawMessage) (*Result, error) { return &Result{}, nil },
	}); err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-frames:
		if frame["method"] != "notifications/tools/list_changed" {
			t.Fatalf("frame = %v", frame)
		}
		meta := frame["params"].(map[string]any)["_meta"].(map[string]any)
		if fmt.Sprint(meta[metaSubscription]) != "1" {
			t.Fatalf("meta = %v: on one channel a notification must name its subscription", meta)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("a tool appeared and nobody was told")
	}

	if err := m.Refresh(context.Background()); err != nil {
		t.Fatal(err)
	}
	select {
	case frame := <-frames:
		t.Fatalf("frame = %v: nothing moved", frame)
	case <-time.After(300 * time.Millisecond):
	}
}

func TestTheOlderEraIsToldOnItsOwnChannel(t *testing.T) {
	for _, era := range []string{"legacy", "modern"} {
		m := NewManager()
		inR, inW := io.Pipe()
		outR, outW := io.Pipe()
		done := make(chan error, 1)
		go func() {
			err := m.ServeStdio(context.Background(), inR, outW)
			outW.Close()
			done <- err
		}()
		dec := json.NewDecoder(outR)

		opening := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}`
		if era == "modern" {
			opening = `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{` + modernMeta() + `}}`
		}
		if _, err := fmt.Fprintln(inW, opening); err != nil {
			t.Fatal(err)
		}
		var first map[string]any
		if err := dec.Decode(&first); err != nil {
			t.Fatal(err)
		}
		if era == "legacy" {
			tools := first["result"].(map[string]any)["capabilities"].(map[string]any)["tools"].(map[string]any)
			if tools["listChanged"] != true {
				t.Fatalf("capabilities = %v: a pipe can carry these, so the claim is good", tools)
			}
		}

		if err := m.Serve("goose", Tool{
			Definition: llm.Tool{Name: "ask"},
			Call:       func(context.Context, json.RawMessage) (*Result, error) { return &Result{}, nil },
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := fmt.Fprintln(inW, `{"jsonrpc":"2.0","id":2,"method":"ping"}`); err != nil {
			t.Fatal(err)
		}
		told := false
		for i := 0; i < 2; i++ {
			var frame map[string]any
			if err := dec.Decode(&frame); err != nil {
				t.Fatal(err)
			}
			if frame["method"] == "notifications/tools/list_changed" {
				told = true
				continue
			}
			if fmt.Sprint(frame["id"]) != "2" {
				t.Fatalf("frame = %v", frame)
			}
			if era == "modern" {
				break
			}
		}
		if told != (era == "legacy") {
			t.Fatalf("%s era: told = %v", era, told)
		}
		inW.Close()
		<-done
		m.Close()
	}
}

func TestProgressReachesTheClientAheadOfTheAnswer(t *testing.T) {
	m := NewManager()
	defer m.Close()
	if err := m.Serve("goose", Tool{
		Definition: llm.Tool{Name: "slow"},
		Call: func(ctx context.Context, _ json.RawMessage) (*Result, error) {
			Report(ctx, Progress{Done: 1, Total: 2, Message: "half way"})
			Report(ctx, Progress{Done: 2, Total: 2})
			return &Result{Content: []llm.Part{&llm.Text{Text: "done"}}}, nil
		},
	}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(m.Handler())
	defer srv.Close()

	body := `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"slow","arguments":{},` +
		`"_meta":{"io.modelcontextprotocol/protocolVersion":"` + modernVersion + `",` +
		`"io.modelcontextprotocol/clientCapabilities":{},"progressToken":"tok-1"}}}`
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/", strings.NewReader(body))
	req.Header.Set("MCP-Protocol-Version", modernVersion)
	req.Header.Set("Mcp-Method", "tools/call")
	req.Header.Set("Mcp-Name", "slow")
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content type = %q: a call being watched is answered on a stream", ct)
	}

	var reports int
	var answered bool
	sc := bufio.NewScanner(resp.Body)
	for sc.Scan() {
		line := strings.TrimPrefix(sc.Text(), "data: ")
		if line == sc.Text() {
			continue
		}
		var frame map[string]any
		if json.Unmarshal([]byte(line), &frame) != nil {
			t.Fatalf("frame = %s", line)
		}
		if frame["method"] == "notifications/progress" {
			if answered {
				t.Fatal("a report arrived after the answer")
			}
			params := frame["params"].(map[string]any)
			if params["progressToken"] != "tok-1" {
				t.Fatalf("report = %v: it must name the token the client chose", params)
			}
			reports++
			continue
		}
		answered = true
		content := frame["result"].(map[string]any)["content"].([]any)
		if content[0].(map[string]any)["text"] != "done" {
			t.Fatalf("answer = %v", frame)
		}
	}
	if reports != 2 || !answered {
		t.Fatalf("reports = %d, answered = %v", reports, answered)
	}
}

func TestProgressFromAnUpstreamServerIsRelayed(t *testing.T) {
	ctx := context.Background()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		body := string(raw)
		id := requestID(body)
		switch {
		case strings.Contains(body, "server/discover"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","error":{"code":-32601,"message":"unknown method"}}`))
		case strings.Contains(body, "initialize"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"protocolVersion":"2025-11-25","serverInfo":{"name":"up"}}}`))
		case strings.Contains(body, "tools/list"):
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":"` + id + `","result":{"tools":[{"name":"slow","inputSchema":{"type":"object"}}]}}`))
		default:
			token := requestToken(json.RawMessage(bodyParams(body)))
			w.Header().Set("Content-Type", "text/event-stream")
			flusher := w.(http.Flusher)
			raw, _ := json.Marshal(progressFrame(token, Progress{Done: 1, Total: 3, Message: "working"}))
			fmt.Fprintf(w, "data: %s\n\n", raw)
			flusher.Flush()
			fmt.Fprintf(w, "data: {\"jsonrpc\":\"2.0\",\"id\":\"%s\",\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"done\"}]}}\n\n", id)
			flusher.Flush()
		}
	}))
	defer upstream.Close()

	m := NewManager()
	defer m.Close()
	if err := m.Add(ctx, ServerConfig{Name: "up", Transport: Streamable, URL: upstream.URL}, upstream.Client()); err != nil {
		t.Fatal(err)
	}

	var seen []Progress
	watching := WithProgress(ctx, func(p Progress) { seen = append(seen, p) })
	result, err := m.Call(watching, "slow", nil)
	if err != nil {
		t.Fatal(err)
	}
	if llm.TextOf(result.Content) != "done" {
		t.Fatalf("result = %+v", result.Content)
	}
	if len(seen) != 1 || seen[0].Message != "working" || seen[0].Total != 3 {
		t.Fatalf("reports = %+v: what the server said it was doing did not reach the caller", seen)
	}

	if _, err := m.Call(ctx, "slow", nil); err != nil {
		t.Fatal(err)
	}
}

func bodyParams(body string) string {
	var req struct {
		Params json.RawMessage `json:"params"`
	}
	if json.Unmarshal([]byte(body), &req) != nil {
		return "{}"
	}
	return string(req.Params)
}
