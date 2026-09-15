// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package mcp speaks the Model Context Protocol to tool servers.
package mcp

import (
	"encoding/json"
	"fmt"
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
	var got any
	if json.Unmarshal(r.ID, &got) != nil {
		return false
	}
	return fmt.Sprint(got) == id
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
