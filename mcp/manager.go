// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/llm"
)

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
	Tools(ctx context.Context) ([]llm.Tool, error)
	Call(ctx context.Context, name string, args json.RawMessage) (*Result, error)
	Close() error
}

type Manager struct {
	mu        sync.RWMutex
	sources   []source
	index     map[string]source
	Separator string

	stop     chan struct{}
	stopOnce sync.Once

	subs    subscribers
	streams map[*subscriber]struct{}
	tools   string
}

func NewManager() *Manager {
	return &Manager{index: map[string]source{}, Separator: "__", stop: make(chan struct{})}
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
		for _, t := range c.Cached() {
			counts[t.Name]++
		}
	}
	m.index = map[string]source{}
	for _, c := range m.sources {
		for _, t := range c.Cached() {
			name := t.Name
			if counts[t.Name] > 1 {
				name = c.Name() + m.Separator + t.Name
			}
			m.index[name] = c
		}
	}
	if mark := fingerprint(m.index); mark != m.tools {
		m.tools = mark
		return true
	}
	return false
}

func (m *Manager) Tools() []llm.Tool { return m.list(false) }

func (m *Manager) Proxied() []llm.Tool { return m.list(true) }

func (m *Manager) list(serversOnly bool) []llm.Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []llm.Tool
	for name, c := range m.index {
		if _, own := c.(*local); own && serversOnly {
			continue
		}
		for _, t := range c.Cached() {
			if name == t.Name || name == c.Name()+m.Separator+t.Name {
				tool := t
				tool.Name = name
				out = append(out, tool)
				break
			}
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
	c, ok := m.index[name]
	m.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("mcp: no server offers a tool named %q", name)
	}
	target := strings.TrimPrefix(name, c.Name()+m.Separator)
	return c.Call(ctx, target, args)
}

type ServerStatus struct {
	Name    string
	Info    ServerInfo
	Tools   int
	Refused []string
}

func (m *Manager) Servers() []ServerStatus {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]ServerStatus, 0, len(m.sources))
	for _, c := range m.sources {
		out = append(out, ServerStatus{
			Name: c.Name(), Info: c.Info(), Tools: len(c.Cached()), Refused: c.Refused(),
		})
	}
	return out
}

func (m *Manager) Close() error {
	m.stopOnce.Do(func() { close(m.stop) })
	m.mu.Lock()
	sources := m.sources
	m.sources = nil
	m.index = map[string]source{}
	m.mu.Unlock()
	var first error
	for _, c := range sources {
		if err := c.Close(); err != nil && first == nil {
			first = err
		}
	}
	return first
}
