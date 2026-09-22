// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package mcp speaks the Model Context Protocol to tool servers.
package mcp

import (
	"encoding/json"
	"fmt"
	"strconv"
	"sync/atomic"
)

const (
	protocolVersion = "2025-11-25"
	jsonrpcVersion  = "2.0"
)

var version atomic.Pointer[string]

func SetVersion(v string) { version.Store(&v) }

func Version() string {
	if v := version.Load(); v != nil && *v != "" {
		return *v
	}
	return "unknown"
}

func implementation() map[string]any {
	return map[string]any{"name": "goose", "version": Version()}
}

type request struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type response struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
	status  int
}

func (e *rpcError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return fmt.Sprintf("mcp: %d %s", e.Code, e.Message)
}

func (r *response) isAnswer(id string) bool {
	if r.Method != "" || len(r.ID) == 0 {
		return false
	}
	return idText(r.ID) == id
}

// idText spells a JSON-RPC id the one way. A server is free to send back as a
// number what goose sent as text, and fmt would write a large one in exponent
// form, which matches nothing.
func idText(raw json.RawMessage) string {
	var got any
	if json.Unmarshal(raw, &got) != nil {
		return ""
	}
	switch v := got.(type) {
	case string:
		return v
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

func newRequest(id string, method string, params any) ([]byte, error) {
	req := request{JSONRPC: jsonrpcVersion, Method: method}
	if id != "" {
		req.ID = id
	}
	if params != nil {
		raw, err := json.Marshal(params)
		if err != nil {
			return nil, err
		}
		req.Params = raw
	}
	return json.Marshal(req)
}
