// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/llm"
)

// A name no server answers to is the caller having asked for something that is
// not there, which is a different answer from a tool that ran and failed.
var errNoSuchTool = errors.New("no server offers a tool by that name")

type TransportKind string

const (
	Stdio      TransportKind = "stdio"
	Streamable TransportKind = "http"
	SSE        TransportKind = "sse"
)

type ServerConfig struct {
	Name      string
	Transport TransportKind
	Command   string
	Args      []string
	Dir       string
	Env       []string
	URL       string
	Headers   map[string]string
	Timeout   time.Duration
	MaxFrame  int
}

func (c ServerConfig) kind() TransportKind {
	switch {
	case c.Transport != "":
		return c.Transport
	case c.Command != "":
		return Stdio
	default:
		return Streamable
	}
}

func Connect(ctx context.Context, cfg ServerConfig, client *http.Client) (*Client, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("mcp: server has no name")
	}
	var (
		t   Transport
		err error
	)
	opts := Options{Timeout: cfg.Timeout}
	switch cfg.kind() {
	case Stdio:
		t, err = startStdio(ctx, cfg, cfg.MaxFrame)
	case SSE:
		t, err = startSSE(ctx, cfg, client, cfg.MaxFrame)
		opts.Handshake = true
	default:
		t, err = newHTTPTransport(cfg, client, cfg.MaxFrame)
		opts.Mirrors = true
	}
	if err != nil {
		return nil, fmt.Errorf("mcp %s: %w", cfg.Name, err)
	}
	c := newClient(cfg.Name, t, opts)
	if err := c.connect(ctx); err != nil {
		_ = t.Close()
		return nil, err
	}
	return c, nil
}

const DefaultRefresh = 30 * time.Second

type source interface {
	Name() string
	Info() ServerInfo
	Cached() []llm.Tool
	Raw(name string) json.RawMessage
	Refused() []string
	Truncated() bool
	Tools(ctx context.Context) ([]llm.Tool, error)
	Call(ctx context.Context, name string, args json.RawMessage) (*Result, error)
	Close() error
}

type Manager struct {
	mu        sync.RWMutex
	sources   []source
	index     map[string]indexed
	Separator string

	stop     chan struct{}
	stopOnce sync.Once

	subs    subscribers
	streams map[*subscriber]struct{}
	tools   string
}

// indexed is a tool as the index offers it: the source that serves it, and the
// name that source knows it by, which differs from the offered name only where
// two sources claimed the same one.
type indexed struct {
	src  source
	tool string
}

func NewManager() *Manager {
	return &Manager{index: map[string]indexed{}, Separator: "__", stop: make(chan struct{})}
}

func (m *Manager) Watch(ctx context.Context, every time.Duration, log func(error)) {
	if every <= 0 {
		every = DefaultRefresh
	}
	go func() {
		ticker := time.NewTicker(every)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-m.stop:
				return
			case <-ticker.C:
				within, cancel := context.WithTimeout(ctx, every)
				err := m.Refresh(within)
				cancel()
				if err != nil && log != nil {
					log(err)
				}
			}
		}
	}()
}

func (m *Manager) Add(ctx context.Context, cfg ServerConfig, client *http.Client) error {
	if err := m.free(cfg.Name); err != nil {
		return err
	}
	c, err := Connect(ctx, cfg, client)
	if err != nil {
		return err
	}
	if _, err := c.Tools(ctx); err != nil {
		_ = c.Close()
		return err
	}
	m.mu.Lock()
	// Checked again under the lock: two Adds racing on the same name both
	// passed the first check while neither had connected yet.
	for _, existing := range m.sources {
		if existing.Name() == cfg.Name {
			m.mu.Unlock()
			_ = c.Close()
			return fmt.Errorf("mcp: %q is already the name of a tool source", cfg.Name)
		}
	}
	m.sources = append(m.sources, c)
	m.mu.Unlock()
	m.reindex()
	return nil
}

func (m *Manager) free(name string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, c := range m.sources {
		if c.Name() == name {
			return fmt.Errorf("mcp: %q is already the name of a tool source", name)
		}
	}
	return nil
}

func (m *Manager) reindex() {
	if m.rebuild() {
		m.announceTools()
	}
}

func (m *Manager) rebuild() (changed bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	counts := map[string]int{}
	for _, c := range m.sources {
		for _, name := range offered(c) {
			counts[name]++
		}
	}
	known := make(map[string]bool, len(m.sources))
	for _, c := range m.sources {
		known[c.Name()] = true
	}
	m.index = map[string]indexed{}
	for _, c := range m.sources {
		for _, tool := range offered(c) {
			name := tool
			if counts[tool] > 1 || squats(tool, c.Name(), m.Separator, known) {
				name = c.Name() + m.Separator + tool
			}
			// A tool whose own name is spelled like another's qualified one
			// would otherwise take that name from it, and which of them wins
			// would come down to the order they were configured in. Neither is
			// dropped: the one that arrives second is qualified again, so both
			// stay callable and the name says which server it came from. The
			// clash can be with a name the same server already holds. The bound
			// is belt and braces — every round adds a prefix, so it ends on its
			// own unless a caller has emptied both the name and the separator.
			for tries := 0; tries <= len(m.index); tries++ {
				if _, clash := m.index[name]; !clash {
					break
				}
				name = c.Name() + m.Separator + name
			}
			if _, clash := m.index[name]; clash {
				continue
			}
			m.index[name] = indexed{src: c, tool: tool}
		}
	}
	if mark := fingerprint(m.index); mark != m.tools {
		m.tools = mark
		return true
	}
	return false
}

// squats reports a name that reads as another server's qualified one. Nothing
// stops a server calling its tool "github__search", and left alone that name
// would sit exactly where the github server's own tool belongs. A server's own
// qualified spelling is its to use.
func squats(tool, server, separator string, known map[string]bool) bool {
	if separator == "" {
		return false
	}
	head, _, ok := strings.Cut(tool, separator)
	return ok && head != server && known[head]
}

// offered names what a source serves, each once. A server is free to list one
// name twice; the second copy is the same tool, and giving it a name of its own
// would cost a name per copy.
func offered(c source) []string {
	cached := c.Cached()
	seen := make(map[string]bool, len(cached))
	out := make([]string, 0, len(cached))
	for _, t := range cached {
		if seen[t.Name] {
			continue
		}
		seen[t.Name] = true
		out = append(out, t.Name)
	}
	return out
}

func (m *Manager) Tools() []llm.Tool { return m.list(false) }

func (m *Manager) Proxied() []llm.Tool { return m.list(true) }

func (m *Manager) list(serversOnly bool) []llm.Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	// Each source is asked for its catalogue once. Asking per entry walks the
	// whole of it for every tool in it, which is quadratic in what a server
	// chose to send, under the lock a refresh also wants.
	byName := make(map[source]map[string]llm.Tool, len(m.sources))
	var out []llm.Tool
	for name, e := range m.index {
		if _, own := e.src.(*local); own && serversOnly {
			continue
		}
		cached, ok := byName[e.src]
		if !ok {
			all := e.src.Cached()
			cached = make(map[string]llm.Tool, len(all))
			for _, t := range all {
				if _, seen := cached[t.Name]; !seen {
					cached[t.Name] = t
				}
			}
			byName[e.src] = cached
		}
		if t, found := cached[e.tool]; found {
			t.Name = name
			out = append(out, t)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (m *Manager) Refresh(ctx context.Context) error {
	m.mu.RLock()
	sources := append([]source(nil), m.sources...)
	m.mu.RUnlock()
	var first error
	for _, c := range sources {
		if _, err := c.Tools(ctx); err != nil && first == nil {
			first = err
		}
	}
	m.reindex()
	return first
}

func (m *Manager) Call(ctx context.Context, name string, args json.RawMessage) (*Result, error) {
	m.mu.RLock()
	e, ok := m.index[name]
	m.mu.RUnlock()
	if !ok {
		// The name came from the caller, so it is said back the way any other
		// name goose will not carry is said.
		said, _ := quoted(name)
		return nil, fmt.Errorf("mcp: %w: %s", errNoSuchTool, said)
	}
	return e.src.Call(ctx, e.tool, args)
}

type ServerStatus struct {
	Name      string
	Info      ServerInfo
	Tools     int
	Refused   []string
	Truncated bool
}

func (m *Manager) Servers() []ServerStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]ServerStatus, 0, len(m.sources))
	for _, c := range m.sources {
		out = append(out, ServerStatus{
			Name: c.Name(), Info: c.Info(), Tools: len(c.Cached()),
			Refused: c.Refused(), Truncated: c.Truncated(),
		})
	}
	return out
}

func (m *Manager) Close() error {
	m.stopOnce.Do(func() { close(m.stop) })
	m.mu.Lock()
	sources := m.sources
	m.sources = nil
	m.index = map[string]indexed{}
	m.mu.Unlock()
	var first error
	for _, c := range sources {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
