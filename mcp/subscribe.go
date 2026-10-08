// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"sort"
	"strings"
	"sync"
)

const metaSubscription = "io.modelcontextprotocol/subscriptionId"

type filter struct {
	Tools        bool     `json:"toolsListChanged"`
	Prompts      bool     `json:"promptsListChanged"`
	Resources    bool     `json:"resourcesListChanged"`
	ResourceURIs []string `json:"resourceSubscriptions"`
}

func (f filter) agreed() map[string]any {
	out := map[string]any{}
	if f.Tools {
		out["toolsListChanged"] = true
	}
	return out
}

type subscriber struct {
	id     any
	want   filter
	frames chan map[string]any
}

func (m *Manager) subscribe(id any, want filter) *subscriber {
	s := &subscriber{id: id, want: want, frames: make(chan map[string]any, 8)}
	m.subs.Lock()
	if m.streams == nil {
		m.streams = map[*subscriber]struct{}{}
	}
	m.streams[s] = struct{}{}
	m.subs.Unlock()
	return s
}

func (m *Manager) unsubscribe(s *subscriber) {
	m.subs.Lock()
	delete(m.streams, s)
	m.subs.Unlock()
}

func (m *Manager) announceTools() {
	m.subs.Lock()
	streams := make([]*subscriber, 0, len(m.streams))
	for s := range m.streams {
		if s.want.Tools {
			streams = append(streams, s)
		}
	}
	m.subs.Unlock()
	for _, s := range streams {
		frame := map[string]any{
			"jsonrpc": jsonrpcVersion,
			"method":  "notifications/tools/list_changed",
			"params":  map[string]any{"_meta": map[string]any{metaSubscription: s.id}},
		}
		select {
		case s.frames <- frame:
		default:
		}
	}
}

// A subscription is named by the id of the request that opened it, spelled as
// the client spelled it.
func acknowledged(id any, want filter) map[string]any {
	return map[string]any{
		"jsonrpc": jsonrpcVersion,
		"method":  "notifications/subscriptions/acknowledged",
		"params": map[string]any{
			"_meta":         map[string]any{metaSubscription: id},
			"notifications": want.agreed(),
		},
	}
}

func closed(id any) map[string]any {
	return map[string]any{
		"resultType": "complete",
		"_meta": map[string]any{
			metaSubscription: id,
			metaServerInfo:   implementation(),
		},
	}
}

func (m *Manager) stream(ctx context.Context, id any, want filter, send func(map[string]any)) {
	s := m.subscribe(id, want)
	defer m.unsubscribe(s)
	send(acknowledged(id, want))
	for {
		select {
		case <-ctx.Done():
			return
		case frame := <-s.frames:
			send(frame)
		}
	}
}

func fingerprint(index map[string]indexed) string {
	names := make([]string, 0, len(index))
	for name := range index {
		names = append(names, name)
	}
	sort.Strings(names)
	return strings.Join(names, "\x00")
}

type subscribers struct {
	sync.Mutex
}
