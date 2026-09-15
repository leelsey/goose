// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
)

func (m *Manager) dispatch(ctx context.Context, req *request, seen mirrored) (any, *rpcError) {
	if agreement, ok := readMeta(req.Params); ok {
		return m.modern(ctx, req, agreement, seen)
	}
	return m.legacy(ctx, req.Method, req.Params, !seen.Present)
}

func (m *Manager) modern(ctx context.Context, req *request, agreement meta, seen mirrored) (any, *rpcError) {
	params, rerr := checkModern(req, agreement, seen)
	if rerr != nil {
		return nil, rerr
	}

	switch req.Method {
	case "server/discover":
		return answer(map[string]any{
			"supportedVersions": modernSpoken,
			"capabilities":      m.capabilities(true),
		}), nil

	case "tools/list":
		return answer(map[string]any{"tools": m.toolList()}), nil

	case "tools/call":
		expected, err := paramHeaders(m.schemaOf(params.Name), params.Arguments)
		if err == nil {
			if bad := seen.checkParams(expected); bad != nil {
				return nil, bad
			}
		}
		result, rerr := m.callTool(ctx, params.Name, params.Arguments)
		if rerr != nil {
			return nil, rerr
		}
		return answer(result), nil

	case "ping":
		return answer(map[string]any{}), nil

	default:
		return nil, &rpcError{Code: -32601, Message: "unknown method " + req.Method, status: 404}
	}
}

type modernParams struct {
	Name          string          `json:"name"`
	Arguments     json.RawMessage `json:"arguments"`
	Cursor        string          `json:"cursor"`
	Notifications filter          `json:"notifications"`
}

func checkModern(req *request, agreement meta, seen mirrored) (modernParams, *rpcError) {
	var params modernParams
	if len(agreement.Caps) == 0 {
		return params, &rpcError{
			Code: -32602, Message: "io.modelcontextprotocol/clientCapabilities is required on every request",
			status: 400,
		}
	}
	if !speaksModern(agreement.Version) {
		return params, unsupportedVersion(agreement.Version)
	}
	if len(req.Params) > 0 {
		if err := json.Unmarshal(req.Params, &params); err != nil {
			return params, &rpcError{Code: -32602, Message: "the parameters could not be read", status: 400}
		}
	}
	named := ""
	if req.Method == "tools/call" {
		named = params.Name
	}
	if err := seen.check(agreement.Version, req.Method, named); err != nil {
		return params, err
	}
	return params, nil
}

func (m *Manager) listens(req *request, seen mirrored) (filter, *rpcError, bool) {
	agreement, ok := readMeta(req.Params)
	if !ok || req.Method != "subscriptions/listen" || req.ID == nil {
		return filter{}, nil, false
	}
	params, rerr := checkModern(req, agreement, seen)
	if rerr != nil {
		return filter{}, rerr, true
	}
	return params.Notifications, nil, true
}

func (m *Manager) legacy(ctx context.Context, method string, params json.RawMessage, canNotify bool) (any, *rpcError) {
	switch method {
	case "initialize":
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		_ = json.Unmarshal(params, &p)
		return map[string]any{
			"protocolVersion": agreed(p.ProtocolVersion),
			"capabilities":    m.capabilities(canNotify),
			"serverInfo":      implementation(),
		}, nil

	case "tools/list":
		return map[string]any{"tools": m.toolList()}, nil

	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(params, &p); err != nil {
			return nil, &rpcError{Code: -32602, Message: "the parameters could not be read"}
		}
		result, err := m.callTool(ctx, p.Name, p.Arguments)
		if err != nil {
			return nil, err
		}
		return result, nil

	case "ping":
		return map[string]any{}, nil

	default:
		return nil, &rpcError{Code: -32601, Message: "unknown method " + method}
	}
}

func (m *Manager) capabilities(canNotify bool) map[string]any {
	tools := map[string]any{}
	if canNotify {
		tools["listChanged"] = true
	}
	return map[string]any{"tools": tools}
}

func (m *Manager) schemaOf(name string) json.RawMessage {
	m.mu.RLock()
	c, ok := m.index[name]
	m.mu.RUnlock()
	if !ok {
		return nil
	}
	for _, t := range c.Cached() {
		if t.Name == name || c.Name()+m.Separator+t.Name == name {
			return t.Schema
		}
	}
	return nil
}

func (m *Manager) toolList() []map[string]any {
	tools := m.Tools()
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if entry := m.rawTool(t.Name); entry != nil {
			out = append(out, entry)
			continue
		}
		schema := t.Schema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		entry := map[string]any{
			"name": t.Name, "description": t.Description, "inputSchema": schema,
		}
		if len(t.OutputSchema) > 0 {
			entry["outputSchema"] = t.OutputSchema
		}
		out = append(out, entry)
	}
	return out
}

func (m *Manager) rawTool(name string) map[string]any {
	m.mu.RLock()
	c, ok := m.index[name]
	m.mu.RUnlock()
	if !ok {
		return nil
	}
	raw := c.Raw(strings.TrimPrefix(name, c.Name()+m.Separator))
	if len(raw) == 0 {
		return nil
	}
	var entry map[string]any
	if json.Unmarshal(raw, &entry) != nil {
		return nil
	}
	entry["name"] = name
	if schema, ok := entry["inputSchema"]; !ok || schema == nil {
		entry["inputSchema"] = json.RawMessage(`{"type":"object","properties":{}}`)
	}
	return entry
}

func (m *Manager) callTool(ctx context.Context, name string, args json.RawMessage) (map[string]any, *rpcError) {
	result, err := m.Call(ctx, name, args)
	if err != nil {
		return nil, &rpcError{Code: -32602, Message: err.Error()}
	}
	out := map[string]any{
		"content": encodeContent(result.Content),
		"isError": result.IsError,
	}
	if len(result.Structured) > 0 {
		out["structuredContent"] = result.Structured
	}
	return out, nil
}

var spoken = []string{"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}

func agreed(asked string) string {
	if slices.Contains(spoken, asked) {
		return asked
	}
	return protocolVersion
}

const grace = 2 * time.Second

func (m *Manager) ServeStdio(ctx context.Context, in io.Reader, out io.Writer) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var (
		mu       sync.Mutex
		writeErr error
	)
	send := func(frame map[string]any) {
		mu.Lock()
		defer mu.Unlock()
		if writeErr != nil || ctx.Err() != nil {
			return
		}
		if err := json.NewEncoder(out).Encode(frame); err != nil {
			writeErr = err
			cancel()
		}
	}
	failed := func() error {
		mu.Lock()
		defer mu.Unlock()
		return writeErr
	}

	var readErr error
	frames := make(chan []byte)
	go func() {
		defer close(frames)
		lines := wirecore.NewLines(in, maxRPCBody)
		for lines.Scan() {
			frame := append([]byte(nil), lines.Bytes()...)
			select {
			case frames <- frame:
			case <-ctx.Done():
				return
			}
		}
		readErr = lines.Err()
	}()

	var (
		armLegacy sync.Once
		unarm     = func() {}
	)
	defer func() { unarm() }()
	pushLegacy := func() {
		armLegacy.Do(func() {
			s := m.subscribe("", filter{Tools: true})
			unarm = func() { m.unsubscribe(s) }
			go func() {
				for {
					select {
					case <-ctx.Done():
						return
					case frame := <-s.frames:
						if params, ok := frame["params"].(map[string]any); ok {
							delete(params, "_meta")
						}
						send(frame)
					}
				}
			}()
		})
	}

	inFlight := &running{calls: map[string]context.CancelFunc{}}
	var wg sync.WaitGroup
	defer func() { cancel() }()
	for {
		select {
		case <-ctx.Done():
			return failed()
		case frame, open := <-frames:
			if !open {
				settled := make(chan struct{})
				go func() { wg.Wait(); close(settled) }()
				select {
				case <-settled:
				case <-time.After(grace):
					cancel()
				}
				if err := failed(); err != nil {
					return err
				}
				return readErr
			}
			var req request
			if err := json.Unmarshal(frame, &req); err != nil {
				send(map[string]any{
					"jsonrpc": jsonrpcVersion, "id": nil,
					"error": map[string]any{"code": -32700, "message": "the request is not valid JSON"},
				})
				continue
			}
			if req.ID == nil {
				if req.Method == "notifications/cancelled" {
					inFlight.cancel(cancelledID(req.Params))
				}
				continue
			}
			wg.Add(1)
			go func(req request) {
				defer wg.Done()
				answer := map[string]any{"jsonrpc": jsonrpcVersion, "id": req.ID}
				id := fmt.Sprint(req.ID)
				within, done := inFlight.start(ctx, id)
				defer done()
				if want, rerr, ok := m.listens(&req, mirrored{}); ok {
					if rerr != nil {
						answer["error"] = map[string]any{"code": rerr.Code, "message": rerr.Message}
						send(answer)
						return
					}
					m.stream(within, id, want, send)
					if ctx.Err() == nil {
						answer["result"] = closed(id)
						send(answer)
					}
					return
				}
				if _, modern := readMeta(req.Params); !modern && req.Method == "initialize" {
					pushLegacy()
				}
				if token := requestToken(req.Params); token != nil && req.Method == "tools/call" {
					within = WithProgress(within, func(p Progress) {
						send(progressFrame(token, p))
					})
				}
				result, rerr := m.dispatch(within, &req, mirrored{})
				if within.Err() != nil && ctx.Err() == nil {
					return
				}
				if rerr != nil {
					refusal := map[string]any{"code": rerr.Code, "message": rerr.Message}
					if len(rerr.Data) > 0 {
						refusal["data"] = rerr.Data
					}
					answer["error"] = refusal
				} else {
					answer["result"] = result
				}
				send(answer)
			}(req)
		}
	}
}

type running struct {
	mu    sync.Mutex
	calls map[string]context.CancelFunc
}

func (r *running) start(ctx context.Context, id string) (context.Context, func()) {
	within, cancel := context.WithCancel(ctx)
	r.mu.Lock()
	r.calls[id] = cancel
	r.mu.Unlock()
	return within, func() {
		r.mu.Lock()
		delete(r.calls, id)
		r.mu.Unlock()
		cancel()
	}
}

func (r *running) cancel(id string) {
	if id == "" {
		return
	}
	r.mu.Lock()
	cancel := r.calls[id]
	r.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func cancelledID(params json.RawMessage) string {
	if len(params) == 0 {
		return ""
	}
	var body struct {
		RequestID json.RawMessage `json:"requestId"`
	}
	if json.Unmarshal(params, &body) != nil || len(body.RequestID) == 0 {
		return ""
	}
	var id any
	if json.Unmarshal(body.RequestID, &id) != nil {
		return ""
	}
	return fmt.Sprint(id)
}
