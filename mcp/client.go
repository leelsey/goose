// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/leelsey/goose/llm"
)

type Outbound struct {
	ID      string
	Method  string
	Name    string
	Version string
	Payload []byte
	Params  map[string]string
	Observe func(method string, params json.RawMessage)
}

func inputSchema(definition json.RawMessage) json.RawMessage {
	if len(definition) == 0 {
		return nil
	}
	var tool struct {
		InputSchema json.RawMessage `json:"inputSchema"`
	}
	if json.Unmarshal(definition, &tool) != nil {
		return nil
	}
	return tool.InputSchema
}

type Transport interface {
	Call(ctx context.Context, out Outbound) (*response, error)
	Notify(ctx context.Context, out Outbound) error
	Close() error
}

type ServerInfo struct {
	Name            string
	Version         string
	ProtocolVersion string
	Capabilities    map[string]json.RawMessage
	Instructions    string
}

type Client struct {
	name          string
	transport     Transport
	timeout       time.Duration
	probe         time.Duration
	handshakeOnly bool
	mirrors       bool

	seq     atomic.Int64
	mu      sync.Mutex
	info    ServerInfo
	modern  bool
	tools   []llm.Tool
	raw     map[string]json.RawMessage
	refused []string
}

type Options struct {
	Timeout   time.Duration
	Probe     time.Duration
	Handshake bool
	Mirrors   bool
}

func newClient(name string, t Transport, o Options) *Client {
	if o.Timeout <= 0 {
		o.Timeout = 30 * time.Second
	}
	if o.Probe <= 0 {
		o.Probe = 5 * time.Second
	}
	if o.Probe > o.Timeout {
		o.Probe = o.Timeout
	}
	return &Client{
		name: name, transport: t, timeout: o.Timeout, probe: o.Probe,
		handshakeOnly: o.Handshake, mirrors: o.Mirrors,
	}
}

func (c *Client) Name() string { return c.name }

func (c *Client) Info() ServerInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.info
}

func (c *Client) Close() error { return c.transport.Close() }

func (c *Client) call(ctx context.Context, method string, params map[string]any, out any) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	c.mu.Lock()
	modern, version := c.modern, c.info.ProtocolVersion
	c.mu.Unlock()
	if version == "" {
		version = protocolVersion
	}
	id := strconv.FormatInt(c.seq.Add(1), 10)
	meta := map[string]any{}
	if modern {
		meta = requestMeta(version)
	}
	if Watched(ctx) && method == "tools/call" {
		meta["progressToken"] = id
	}
	if len(meta) > 0 {
		if params == nil {
			params = map[string]any{}
		}
		params["_meta"] = meta
	}

	payload, err := newRequest(id, method, params)
	if err != nil {
		return err
	}
	name, _ := params["name"].(string)
	send := Outbound{ID: id, Method: method, Name: name, Version: version, Payload: payload}
	if report, ok := ctx.Value(progressKey{}).(func(Progress)); ok && method == "tools/call" {
		send.Observe = func(frameMethod string, frameParams json.RawMessage) {
			if frameMethod != "notifications/progress" {
				return
			}
			if p, ok := readProgress(frameParams, id); ok {
				report(p)
			}
		}
	}
	if modern && c.mirrors && method == "tools/call" {
		args, _ := params["arguments"].(json.RawMessage)
		c.mu.Lock()
		schema := c.raw[name]
		c.mu.Unlock()
		if headers, err := paramHeaders(inputSchema(schema), args); err == nil {
			send.Params = headers
		}
	}
	resp, err := c.transport.Call(ctx, send)
	if err != nil {
		return fmt.Errorf("mcp %s: %s: %w", c.name, method, err)
	}
	if resp.Error != nil {
		return fmt.Errorf("mcp %s: %s: %w", c.name, method, resp.Error)
	}
	if out == nil || len(resp.Result) == 0 {
		return nil
	}
	if err := json.Unmarshal(resp.Result, out); err != nil {
		return fmt.Errorf("mcp %s: %s: malformed result: %w", c.name, method, err)
	}
	return nil
}

var errIncompatible = errors.New("mcp: the server and goose have no protocol revision in common")

func (c *Client) connect(ctx context.Context) error {
	if c.handshakeOnly {
		return c.initialise(ctx)
	}
	err := c.discover(ctx)
	switch {
	case err == nil:
		return nil
	case errors.Is(err, errIncompatible):
		return err
	}
	c.mu.Lock()
	c.modern, c.info = false, ServerInfo{}
	c.mu.Unlock()
	return c.initialise(ctx)
}

func (c *Client) discover(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, c.probe)
	defer cancel()

	asked := modernVersion
	for attempt := 0; attempt < 2; attempt++ {
		c.mu.Lock()
		c.modern, c.info.ProtocolVersion = true, asked
		c.mu.Unlock()

		var result struct {
			SupportedVersions []string                   `json:"supportedVersions"`
			Capabilities      map[string]json.RawMessage `json:"capabilities"`
			Instructions      string                     `json:"instructions"`
			Meta              struct {
				ServerInfo struct {
					Name    string `json:"name"`
					Version string `json:"version"`
				} `json:"io.modelcontextprotocol/serverInfo"`
			} `json:"_meta"`
		}
		err := c.call(ctx, "server/discover", map[string]any{}, &result)
		if err != nil {
			var refusal *rpcError
			if !errors.As(err, &refusal) || refusal.Code != codeUnsupportedVersion {
				return err
			}
			var offered struct {
				Supported []string `json:"supported"`
			}
			_ = json.Unmarshal(refusal.Data, &offered)
			version, era, ok := choose(offered.Supported)
			switch {
			case !ok:
				return errIncompatible
			case era == eraLegacy:
				return fmt.Errorf("mcp %s: the server speaks the handshake era", c.name)
			case version == asked:
				return errIncompatible
			}
			asked = version
			continue
		}

		if len(result.SupportedVersions) == 0 {
			return fmt.Errorf("mcp %s: the discovery probe was answered by something that named no protocol version", c.name)
		}
		version, era, ok := choose(result.SupportedVersions)
		switch {
		case !ok:
			return errIncompatible
		case era == eraLegacy:
			return fmt.Errorf("mcp %s: the server prefers the handshake era", c.name)
		}
		c.mu.Lock()
		c.info = ServerInfo{
			Name:            result.Meta.ServerInfo.Name,
			Version:         result.Meta.ServerInfo.Version,
			ProtocolVersion: version,
			Capabilities:    result.Capabilities,
			Instructions:    result.Instructions,
		}
		c.mu.Unlock()
		return nil
	}
	return errIncompatible
}

const (
	eraModern = "modern"
	eraLegacy = "legacy"
)

func choose(offered []string) (version, era string, ok bool) {
	for _, mine := range modernSpoken {
		for _, theirs := range offered {
			if mine == theirs {
				return mine, eraModern, true
			}
		}
	}
	for i := len(spoken) - 1; i >= 0; i-- {
		for _, theirs := range offered {
			if spoken[i] == theirs {
				return spoken[i], eraLegacy, true
			}
		}
	}
	return "", "", false
}

func (c *Client) initialise(ctx context.Context) error {
	var result struct {
		ProtocolVersion string                     `json:"protocolVersion"`
		Capabilities    map[string]json.RawMessage `json:"capabilities"`
		ServerInfo      struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"serverInfo"`
		Instructions string `json:"instructions"`
	}
	params := map[string]any{
		"protocolVersion": protocolVersion,
		"capabilities":    map[string]any{"tools": map[string]any{}},
		"clientInfo":      implementation(),
	}
	if err := c.call(ctx, "initialize", params, &result); err != nil {
		return err
	}
	c.mu.Lock()
	c.info = ServerInfo{
		Name:            result.ServerInfo.Name,
		Version:         result.ServerInfo.Version,
		ProtocolVersion: result.ProtocolVersion,
		Capabilities:    result.Capabilities,
		Instructions:    result.Instructions,
	}
	c.mu.Unlock()

	payload, err := newRequest("", "notifications/initialized", map[string]any{})
	if err != nil {
		return err
	}
	return c.transport.Notify(ctx, Outbound{
		Method: "notifications/initialized", Version: result.ProtocolVersion, Payload: payload,
	})
}

func (c *Client) Tools(ctx context.Context) ([]llm.Tool, error) {
	var out []llm.Tool
	var refused []string
	raw := map[string]json.RawMessage{}
	cursor := ""
	for pages := 0; pages < 100; pages++ {
		var result struct {
			Tools      []json.RawMessage `json:"tools"`
			NextCursor string            `json:"nextCursor"`
		}
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		if err := c.call(ctx, "tools/list", params, &result); err != nil {
			return nil, err
		}
		for _, entry := range result.Tools {
			var t struct {
				Name         string          `json:"name"`
				Description  string          `json:"description"`
				InputSchema  json.RawMessage `json:"inputSchema"`
				OutputSchema json.RawMessage `json:"outputSchema"`
			}
			if json.Unmarshal(entry, &t) != nil || t.Name == "" {
				continue
			}
			schema := t.InputSchema
			if len(schema) == 0 || string(schema) == "null" {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			if c.mirrors {
				if _, err := annotated(schema); err != nil {
					refused = append(refused, t.Name)
					continue
				}
			}
			raw[t.Name] = append(json.RawMessage(nil), entry...)
			out = append(out, llm.Tool{
				Name: t.Name, Description: t.Description, Schema: schema,
				OutputSchema: t.OutputSchema,
			})
		}
		if result.NextCursor == "" {
			break
		}
		cursor = result.NextCursor
	}
	c.mu.Lock()
	c.tools, c.raw, c.refused = out, raw, refused
	c.mu.Unlock()
	return out, nil
}

func (c *Client) Cached() []llm.Tool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]llm.Tool(nil), c.tools...)
}

func (c *Client) Raw(name string) json.RawMessage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.raw[name]
}

func (c *Client) Refused() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.refused...)
}

type Result struct {
	Content    []llm.Part
	IsError    bool
	Structured json.RawMessage
}

func (c *Client) Call(ctx context.Context, name string, args json.RawMessage) (*Result, error) {
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	var result struct {
		Content           []json.RawMessage `json:"content"`
		IsError           bool              `json:"isError"`
		StructuredContent json.RawMessage   `json:"structuredContent"`
		ResultType        string            `json:"resultType"`
	}
	params := map[string]any{"name": name, "arguments": json.RawMessage(args)}
	if err := c.call(ctx, "tools/call", params, &result); err != nil {
		return nil, err
	}
	if result.ResultType == "input_required" {
		return nil, fmt.Errorf("mcp %s: tool %q needs input from the client that goose cannot supply", c.name, name)
	}
	return &Result{
		Content:    decodeContent(result.Content),
		IsError:    result.IsError,
		Structured: result.StructuredContent,
	}, nil
}

func decodeContent(blocks []json.RawMessage) []llm.Part {
	var out []llm.Part
	for _, raw := range blocks {
		var block struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Data     string `json:"data"`
			MIMEType string `json:"mimeType"`
			Resource *struct {
				URI      string `json:"uri"`
				MIMEType string `json:"mimeType"`
				Text     string `json:"text"`
				Blob     string `json:"blob"`
			} `json:"resource"`
		}
		if json.Unmarshal(raw, &block) != nil {
			out = append(out, &llm.Raw{Vendor: "mcp", Type: "unknown", JSON: raw})
			continue
		}
		switch block.Type {
		case "text":
			out = append(out, &llm.Text{Text: block.Text})
		case "image", "audio":
			data, err := base64.StdEncoding.DecodeString(block.Data)
			if err != nil {
				out = append(out, &llm.Raw{Vendor: "mcp", Type: block.Type, JSON: raw})
				continue
			}
			out = append(out, &llm.Image{MIME: block.MIMEType, Data: data})
		case "resource":
			if block.Resource != nil && block.Resource.Text != "" {
				out = append(out, &llm.Text{Text: block.Resource.Text})
				continue
			}
			out = append(out, &llm.Raw{Vendor: "mcp", Type: "resource", JSON: raw})
		default:
			out = append(out, &llm.Raw{Vendor: "mcp", Type: block.Type, JSON: raw})
		}
	}
	return out
}
