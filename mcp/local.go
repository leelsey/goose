// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/leelsey/goose/llm"
)

type Tool struct {
	Definition llm.Tool
	Call       func(ctx context.Context, args json.RawMessage) (*Result, error)
}

type local struct {
	name  string
	tools []llm.Tool
	calls map[string]func(context.Context, json.RawMessage) (*Result, error)
}

func (l *local) Name() string { return l.name }

func (l *local) Info() ServerInfo {
	return ServerInfo{Name: l.name, Version: Version()}
}

func (l *local) Cached() []llm.Tool { return append([]llm.Tool(nil), l.tools...) }

func (l *local) Raw(string) json.RawMessage { return nil }

func (l *local) Refused() []string { return nil }

func (l *local) Truncated() bool { return false }

func (l *local) Tools(context.Context) ([]llm.Tool, error) { return l.Cached(), nil }

func (l *local) Call(ctx context.Context, name string, args json.RawMessage) (*Result, error) {
	fn, ok := l.calls[name]
	if !ok {
		return nil, fmt.Errorf("mcp: %s offers no tool named %q", l.name, name)
	}
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	return fn(ctx, args)
}

func (l *local) Close() error { return nil }

func (m *Manager) Serve(name string, tools ...Tool) error {
	if name == "" {
		return fmt.Errorf("mcp: served tools have no name")
	}
	if err := m.free(name); err != nil {
		return err
	}
	src := &local{name: name, calls: map[string]func(context.Context, json.RawMessage) (*Result, error){}}
	for _, t := range tools {
		switch {
		case t.Definition.Name == "":
			return fmt.Errorf("mcp %s: a tool with no name is one nothing can call", name)
		case t.Call == nil:
			return fmt.Errorf("mcp %s: tool %q has nothing to run", name, t.Definition.Name)
		case src.calls[t.Definition.Name] != nil:
			return fmt.Errorf("mcp %s: tool %q is defined twice", name, t.Definition.Name)
		}
		// A schema goose cannot mirror is refused here, where whoever wrote it
		// can see why, rather than on every call that reaches it. A remote
		// server's tools are held to the same rule when they are listed.
		if len(t.Definition.Schema) > 0 {
			if _, err := annotated(t.Definition.Schema); err != nil {
				return fmt.Errorf("mcp %s: tool %q: %w", name, t.Definition.Name, err)
			}
		}
		def := t.Definition
		if len(def.Schema) == 0 {
			def.Schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		src.tools = append(src.tools, def)
		src.calls[def.Name] = t.Call
	}
	m.mu.Lock()
	// Checked again under the lock: two calls racing on the same name both
	// passed the first check while neither had appended yet.
	for _, existing := range m.sources {
		if existing.Name() == name {
			m.mu.Unlock()
			return fmt.Errorf("mcp: %q is already the name of a tool source", name)
		}
	}
	m.sources = append(m.sources, src)
	m.mu.Unlock()
	m.reindex()
	return nil
}
