// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/wirecore"
)

const defaultHTTPFrame = 8 << 20

type httpTransport struct {
	url     string
	client  *http.Client
	headers map[string]string
	frame   int

	mu      sync.Mutex
	session string
	// lost says the server no longer holds the session. Nothing more is sent
	// on this transport: sent without the session, a request is refused for
	// lacking one, which is not an answer that says to greet the server again.
	lost bool
}

var errSessionLost = errors.New("the server no longer holds the session")

func newHTTPTransport(cfg ServerConfig, client *http.Client, frame int) (*httpTransport, error) {
	if cfg.URL == "" {
		return nil, errors.New("mcp: no url")
	}
	if err := credentialsOverTLS(cfg.URL, cfg.Headers); err != nil {
		return nil, err
	}
	if client == nil {
		client = http.DefaultClient
	}
	client = sameOriginOnly(client, cfg.URL)
	if frame <= 0 {
		frame = defaultHTTPFrame
	}
	return &httpTransport{url: cfg.URL, client: client, headers: cfg.Headers, frame: frame}, nil
}

type refusal struct {
	status int
	body   []byte
}

func (e *refusal) Error() string {
	return fmt.Sprintf("%d %s", e.status, serverError(e.body))
}

func (t *httpTransport) post(ctx context.Context, out Outbound) (*http.Response, error) {
	t.mu.Lock()
	session, lost := t.session, t.lost
	t.mu.Unlock()
	if lost {
		return nil, gone(errSessionLost)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(out.Payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	if session != "" {
		req.Header.Set("Mcp-Session-Id", session)
	}
	version := out.Version
	if version == "" {
		version = protocolVersion
	}
	req.Header.Set("MCP-Protocol-Version", version)
	if out.Method != "" {
		req.Header.Set("Mcp-Method", out.Method)
	}
	if out.Name != "" {
		req.Header.Set("Mcp-Name", headerValue(out.Name))
	}
	for name, value := range out.Params {
		req.Header.Set(name, value)
	}
	resp, err := t.client.Do(req)
	if err != nil {
		return nil, err
	}
	if id := resp.Header.Get("Mcp-Session-Id"); id != "" {
		t.mu.Lock()
		t.session = id
		t.mu.Unlock()
	}
	if resp.StatusCode == http.StatusNotFound && session != "" {
		// A server answers this to a session it no longer holds, having
		// restarted or let it lapse, and acts on nothing sent under it.
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		t.mu.Lock()
		if t.session == session {
			t.lost = true
		}
		t.mu.Unlock()
		return nil, gone(errSessionLost)
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		return nil, &refusal{status: resp.StatusCode, body: body}
	}
	return resp, nil
}

func (t *httpTransport) Call(ctx context.Context, out Outbound) (*response, error) {
	resp, err := t.post(ctx, out)
	if err != nil {
		var refused *refusal
		if errors.As(err, &refused) {
			if answer := errorFrame(refused.body); answer != nil {
				return answer, nil
			}
		}
		return nil, err
	}
	defer resp.Body.Close()

	if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		return readEventAnswer(resp.Body, out, t.frame)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, int64(t.frame)))
	if err != nil {
		return nil, err
	}
	var answer response
	if err := json.Unmarshal(body, &answer); err != nil {
		return nil, errors.New("unreadable body: " + err.Error())
	}
	return &answer, nil
}

func errorFrame(body []byte) *response {
	var out response
	if json.Unmarshal(body, &out) != nil || out.Error == nil {
		return nil
	}
	return &out
}

func (t *httpTransport) Notify(ctx context.Context, out Outbound) error {
	resp, err := t.post(ctx, out)
	if err != nil {
		return err
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
	return resp.Body.Close()
}

func (t *httpTransport) Close() error {
	t.mu.Lock()
	session, lost := t.session, t.lost
	t.session = ""
	t.mu.Unlock()
	if session == "" || lost {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, t.url, nil)
	if err != nil {
		return nil
	}
	req.Header.Set("Mcp-Session-Id", session)
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	c := t.client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
	return resp.Body.Close()
}

func credentialsOverTLS(raw string, headers map[string]string) error {
	if len(headers) == 0 {
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("mcp: url %q: %w", raw, err)
	}
	if u.Scheme == "https" || loopback(u.Hostname()) {
		return nil
	}
	return fmt.Errorf("mcp: refusing to send headers to %s over plaintext; use https", u.Host)
}

func loopback(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func readEventAnswer(r io.Reader, call Outbound, frame int) (*response, error) {
	sc := wirecore.NewScanner(r, frame, frame)
	for sc.Scan() {
		data := bytes.TrimSpace(sc.Event().Data)
		if len(data) == 0 {
			continue
		}
		var out response
		if err := json.Unmarshal(data, &out); err != nil {
			return nil, errors.New("unreadable event: " + err.Error())
		}
		if out.isAnswer(call.ID) {
			return &out, nil
		}
		if out.Method != "" && call.Observe != nil {
			call.Observe(out.Method, out.Params)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return nil, errors.New("the stream ended before the answer arrived")
}

type sseTransport struct {
	streamURL string
	client    *http.Client
	headers   map[string]string
	frame     int

	turn    turn
	postURL string
	body    io.ReadCloser
	stop    context.CancelFunc
	scanner *wirecore.Scanner

	frames   chan []byte
	done     chan struct{}
	once     sync.Once
	closeErr error

	errMu sync.Mutex
	err   error
}

func startSSE(ctx context.Context, cfg ServerConfig, client *http.Client, frame int) (*sseTransport, error) {
	if cfg.URL == "" {
		return nil, errors.New("mcp: no url")
	}
	if err := credentialsOverTLS(cfg.URL, cfg.Headers); err != nil {
		return nil, err
	}
	if client == nil {
		client = http.DefaultClient
	}
	client = sameOriginOnly(client, cfg.URL)
	if frame <= 0 {
		frame = defaultHTTPFrame
	}
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = defaultTimeout
	}
	// The stream outlives whoever opened it, so it is not bound to their
	// context. Opening it is: a server that takes the connection and never
	// names an endpoint would otherwise hold them for ever.
	stream, stop := context.WithCancel(context.WithoutCancel(ctx))
	opening, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	watch := context.AfterFunc(opening, stop)
	t, err := openSSE(stream, cfg, client, frame)
	if !watch() {
		if t != nil {
			t.body.Close()
		}
		err = fmt.Errorf("mcp: the stream named no endpoint in time: %w", opening.Err())
	}
	if err != nil {
		stop()
		return nil, err
	}
	t.stop = stop
	go t.read()
	return t, nil
}

func openSSE(ctx context.Context, cfg ServerConfig, client *http.Client, frame int) (*sseTransport, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, cfg.URL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		resp.Body.Close()
		return nil, fmt.Errorf("%d %s", resp.StatusCode, serverError(body))
	}
	// A redirect may move the path; it may not move the origin. This request
	// carried the operator's credentials for this server and every later post
	// carries them too, so the origin they are sent to is the configured one.
	streamURL := cfg.URL
	if resp.Request != nil && resp.Request.URL != nil {
		final := resp.Request.URL.String()
		if _, err := resolveEndpoint(cfg.URL, final); err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("mcp: %s redirected off its own origin, to %s", cfg.URL, resp.Request.URL.Redacted())
		}
		streamURL = final
	}
	t := &sseTransport{
		streamURL: streamURL, client: client, headers: cfg.Headers, frame: frame,
		turn: make(turn, 1), body: resp.Body, scanner: wirecore.NewScanner(resp.Body, frame, frame),
		frames: make(chan []byte), done: make(chan struct{}),
	}
	// The endpoint event must be read before anything can be sent.
	if err := t.readEndpoint(); err != nil {
		resp.Body.Close()
		return nil, err
	}
	return t, nil
}

func (t *sseTransport) read() {
	defer close(t.frames)
	// A stream that has ended carries nothing more, and the next call opens a
	// fresh one. What ended it is kept for the call that was waiting on it.
	defer t.end()
	for t.scanner.Scan() {
		data := bytes.TrimSpace(t.scanner.Event().Data)
		if len(data) == 0 {
			continue
		}
		frame := append([]byte(nil), data...)
		// A frame is for the call that holds the turn. One that arrives while
		// no call holds it, a late answer or a log line, is for nobody, and
		// holding it for the next call would stop the stream being read: its
		// end, which is what says to open a fresh one, would go unseen.
		select {
		case t.frames <- frame:
		case t.turn <- struct{}{}:
			t.turn.give()
		case <-t.done:
			return
		}
	}
	t.setErr(t.scanner.Err())
}

func (t *sseTransport) setErr(err error) {
	t.errMu.Lock()
	defer t.errMu.Unlock()
	if t.err == nil {
		t.err = err
	}
}

func (t *sseTransport) readErr() error {
	t.errMu.Lock()
	defer t.errMu.Unlock()
	return t.err
}

func (t *sseTransport) readEndpoint() error {
	for t.scanner.Scan() {
		ev := t.scanner.Event()
		if ev.Name != "endpoint" {
			continue
		}
		target := strings.TrimSpace(string(ev.Data))
		resolved, err := resolveEndpoint(t.streamURL, target)
		if err != nil {
			return err
		}
		t.postURL = resolved
		return nil
	}
	if err := t.scanner.Err(); err != nil {
		return err
	}
	return errors.New("mcp: the stream named no endpoint")
}

// sameOriginOnly refuses a redirect that leaves the configured origin, so the
// headers meant for one server are not handed to whoever answers next. Go strips
// Authorization across domains by itself, but carries every other header.
func sameOriginOnly(base *http.Client, origin string) *http.Client {
	copied := *base
	copied.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		if _, err := resolveEndpoint(origin, req.URL.String()); err != nil {
			return fmt.Errorf("mcp: refusing to follow %s off the origin of %s", req.URL.Redacted(), origin)
		}
		return nil
	}
	return &copied
}

func resolveEndpoint(stream, target string) (string, error) {
	base, err := url.Parse(stream)
	if err != nil {
		return "", err
	}
	ref, err := url.Parse(target)
	if err != nil {
		return "", err
	}
	resolved := base.ResolveReference(ref)
	upgrade := base.Scheme == "http" && resolved.Scheme == "https" && resolved.Hostname() == base.Hostname()
	if (resolved.Scheme != base.Scheme && !upgrade) || (!upgrade && !sameHost(resolved, base)) {
		return "", fmt.Errorf("mcp: endpoint %s is not on the same origin as %s", resolved.Redacted(), base.Redacted())
	}
	return resolved.String(), nil
}

func sameHost(a, b *url.URL) bool {
	return strings.EqualFold(withPort(a), withPort(b))
}

func withPort(u *url.URL) string {
	host := u.Hostname()
	port := u.Port()
	if port == "" {
		switch u.Scheme {
		case "https":
			port = "443"
		case "http":
			port = "80"
		}
	}
	return host + ":" + port
}

func (t *sseTransport) Call(ctx context.Context, out Outbound) (*response, error) {
	if err := t.turn.take(ctx); err != nil {
		return nil, err
	}
	defer t.turn.give()
	if err := t.send(ctx, out.Payload); err != nil {
		return nil, err
	}
	id := out.ID
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case data, open := <-t.frames:
			if !open {
				if err := t.readErr(); err != nil {
					return nil, err
				}
				return nil, errors.New("the stream closed before the answer arrived")
			}
			var frame response
			if err := json.Unmarshal(data, &frame); err != nil {
				return nil, errors.New("unreadable event: " + err.Error())
			}
			if frame.isAnswer(id) {
				return &frame, nil
			}
			if frame.Method != "" && out.Observe != nil {
				out.Observe(frame.Method, frame.Params)
			}
		}
	}
}

func (t *sseTransport) Notify(ctx context.Context, out Outbound) error {
	if err := t.turn.take(ctx); err != nil {
		return err
	}
	defer t.turn.give()
	return t.send(ctx, out.Payload)
}

func (t *sseTransport) send(ctx context.Context, payload []byte) error {
	select {
	case <-t.done:
		return gone(t.readErr())
	default:
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, t.postURL, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		req.Header.Set(k, v)
	}
	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
	if resp.StatusCode >= 400 {
		refused := fmt.Errorf("%d %s", resp.StatusCode, http.StatusText(resp.StatusCode))
		if resp.StatusCode == http.StatusNotFound {
			// The server no longer knows the endpoint its stream named, having
			// restarted or dropped the session, and acts on nothing sent to it.
			// It can say so before the end of the stream has been read, and a
			// stream that stays open would have every later call sent here too.
			t.setErr(refused)
			t.end()
			return gone(refused)
		}
		return refused
	}
	return nil
}

func (t *sseTransport) end() {
	t.once.Do(func() {
		close(t.done)
		t.stop()
		t.closeErr = t.body.Close()
	})
}

func (t *sseTransport) Close() error {
	t.end()
	return t.closeErr
}

func serverError(body []byte) string {
	if m, _ := httpcore.Upstream(body); m != "" {
		return m
	}
	return "the tool server refused the request"
}
