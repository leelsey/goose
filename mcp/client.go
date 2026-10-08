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
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

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

// errGone is a transport that can carry nothing more: the child exited, the
// stream ended, or the server no longer holds the session. What was asked of it
// was never acted on, so it can be asked again of the server dialled afresh.
var errGone = errors.New("the connection to the tool server has ended")

func gone(cause error) error {
	if cause == nil {
		return errGone
	}
	return fmt.Errorf("%w: %w", errGone, cause)
}

var errClosed = errors.New("the client is closed")

// turn lets one call at a time use a transport whose answers all arrive on one
// channel, and lets a call waiting for its turn give up when its caller does.
type turn chan struct{}

func (t turn) take(ctx context.Context) error {
	select {
	case t <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	}
	// The turn can come up in the same instant the caller gives up, and the
	// request sent then runs a tool for a caller already told it failed.
	if err := ctx.Err(); err != nil {
		<-t
		return err
	}
	return nil
}

func (t turn) give() { <-t }

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

	// dial opens the transport again once the one in use has ended, and the
	// transport is replaced under mu when it does. Without it a server that
	// goes away stays away.
	dial    func(context.Context) (Transport, error)
	dialing turn
	closed  bool

	seq     atomic.Int64
	mu      sync.Mutex
	info    ServerInfo
	modern  bool
	tools   []llm.Tool
	raw     map[string]json.RawMessage
	refused []string
	// truncated says this server offers more than goose will hold, so the
	// catalogue it serves is part of one.
	truncated bool
}

type Options struct {
	Timeout   time.Duration
	Probe     time.Duration
	Handshake bool
	Mirrors   bool
}

const defaultTimeout = 30 * time.Second

func newClient(name string, t Transport, o Options) *Client {
	if o.Timeout <= 0 {
		o.Timeout = defaultTimeout
	}
	if o.Probe <= 0 {
		o.Probe = 5 * time.Second
	}
	if o.Probe > o.Timeout {
		o.Probe = o.Timeout
	}
	return &Client{
		name: name, transport: t, timeout: o.Timeout, probe: o.Probe,
		handshakeOnly: o.Handshake, mirrors: o.Mirrors, dialing: make(turn, 1),
	}
}

func (c *Client) Name() string { return c.name }

func (c *Client) Info() ServerInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.info
}

func (c *Client) Close() error {
	c.mu.Lock()
	c.closed = true
	t := c.transport
	c.mu.Unlock()
	return t.Close()
}

func (c *Client) current() Transport {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.transport
}

func (c *Client) call(ctx context.Context, method string, params map[string]any, out any) error {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	t := c.current()
	err := c.exchange(ctx, t, method, params, out)
	if !errors.Is(err, errGone) || c.dial == nil {
		return err
	}
	// Nothing sent on a transport that has ended was acted on, so the request
	// goes again to the server dialled afresh. A server that restarted or let
	// its session lapse is not one to give up on for the life of the process.
	if err := c.redial(ctx, t); err != nil {
		return err
	}
	// What the first attempt stamped on the request was for the server it
	// went to, and the one dialled afresh may speak another revision.
	delete(params, "_meta")
	return c.exchange(ctx, c.current(), method, params, out)
}

// redial replaces a transport that has ended with a fresh one, handshake and
// all. Calls that found the same one ended wait for whichever of them dials.
func (c *Client) redial(ctx context.Context, was Transport) error {
	if err := c.dialing.take(ctx); err != nil {
		return fmt.Errorf("mcp %s: %w", c.name, err)
	}
	defer c.dialing.give()
	c.mu.Lock()
	closed, now := c.closed, c.transport
	c.mu.Unlock()
	switch {
	case closed:
		return fmt.Errorf("mcp %s: %w", c.name, errClosed)
	case now != was:
		return nil
	}
	t, err := c.dial(ctx)
	if err != nil {
		return fmt.Errorf("mcp %s: %w", c.name, err)
	}
	fresh := newClient(c.name, t, Options{
		Timeout: c.timeout, Probe: c.probe, Handshake: c.handshakeOnly, Mirrors: c.mirrors,
	})
	if err := fresh.connect(ctx); err != nil {
		_ = t.Close()
		return err
	}
	info, modern := fresh.Info(), fresh.modern
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		_ = t.Close()
		return fmt.Errorf("mcp %s: %w", c.name, errClosed)
	}
	c.transport, c.info, c.modern = t, info, modern
	c.mu.Unlock()
	// What it replaces has already ended, and closing it only reaps what is
	// left: no reason to keep the call that dialled waiting on it.
	go func() { _ = was.Close() }()
	return nil
}

func (c *Client) exchange(ctx context.Context, t Transport, method string, params map[string]any, out any) error {
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
		headers, err := paramHeaders(inputSchema(schema), args)
		if err != nil {
			return fmt.Errorf("mcp %s: %s: %w", c.name, method, err)
		}
		send.Params = headers
	}
	resp, err := t.Call(ctx, send)
	if err != nil {
		return fmt.Errorf("mcp %s: %s: %w", c.name, method, err)
	}
	if resp.Error != nil {
		// What a server puts in a message is its own text and reaches a model's
		// context through a tool error, so it is kept to one line. The refusal
		// itself still travels, because version negotiation reads the code and
		// the data off it.
		refusal := *resp.Error
		refusal.Message = tame(refusal.Message)
		return fmt.Errorf("mcp %s: %s: %w", c.name, method, &refusal)
	}
	// JSON-RPC answers with a result or an error. Reading a frame that carries
	// neither as success turns a tool that never ran into one that ran and said
	// nothing, which is what the model would then be told.
	if len(resp.Result) == 0 {
		return fmt.Errorf("mcp %s: %s: the answer carried neither a result nor an error", c.name, method)
	}
	if out == nil {
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
			Name:            tame(result.Meta.ServerInfo.Name),
			Version:         tame(result.Meta.ServerInfo.Version),
			ProtocolVersion: version,
			Capabilities:    result.Capabilities,
			Instructions:    tame(result.Instructions),
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
	// The version the server picked rides every later request as a header, so it
	// has to be something a header can carry. A revision goose has not heard of
	// is not a reason to refuse a server that otherwise works.
	if !headerSafe(result.ProtocolVersion) {
		return fmt.Errorf("mcp %s: the server answered with a protocol revision a header cannot carry", c.name)
	}
	c.mu.Lock()
	c.info = ServerInfo{
		Name:            tame(result.ServerInfo.Name),
		Version:         tame(result.ServerInfo.Version),
		ProtocolVersion: result.ProtocolVersion,
		Capabilities:    result.Capabilities,
		Instructions:    tame(result.Instructions),
	}
	c.mu.Unlock()

	payload, err := newRequest("", "notifications/initialized", map[string]any{})
	if err != nil {
		return err
	}
	return c.current().Notify(ctx, Outbound{
		Method: "notifications/initialized", Version: result.ProtocolVersion, Payload: payload,
	})
}

// The whole catalogue is walked for every request and held for as long as the
// server is configured, so how much of it there is cannot be the server's to
// choose. This is far past any catalogue a model could be offered in one call.
// A name goose will not carry is held too, and on a ceiling of its own: both
// are bounded, and neither can be spent to leave no room for the other. What is
// read to decide between them is bounded separately again, high enough that a
// catalogue listing each of its tools several times over is still read whole.
const (
	maxTools     = 2048
	maxRefused   = 256
	maxEntries   = 4 * maxTools
	maxWeight    = 32 << 20
	maxToolPages = 100
)

// headerSafe reports a value this proxy can put in a header and read back.
func headerSafe(version string) bool {
	if len(version) > maxSaid {
		return false
	}
	for _, r := range version {
		if r <= 0x20 || r >= 0x7f {
			return false
		}
	}
	return true
}

// A server chooses its own name, its version and its tool names, and goose
// prints all three in a status table. Left alone, one of them can move the
// cursor and rewrite the line above it.
const maxSaid = 128

// A line that passes somebody else's words on carries goose's own account of
// what failed as well, so it is held to a ceiling of its own rather than the
// one a single name or version is held to.
const maxAccount = 1024

func tame(s string) string { return tameTo(s, maxSaid) }

// tameTo keeps what can be printed on one line and measures what is left, not
// what arrived: a byte that is not text at all becomes a rune on the way
// through, so a line cut before that would still be three times the ceiling.
// C0 and DEL are not the only characters that move a cursor or make a name
// read as something it is not, so what is kept is what Go itself will print
// unescaped, which leaves out the C1 controls, the bidi overrides and the
// other invisibles along with them.
func tameTo(s string, max int) string {
	var b strings.Builder
	for _, r := range s {
		if !strconv.IsPrint(r) || r == utf8.RuneError {
			continue
		}
		if b.Len()+utf8.RuneLen(r) > max {
			break
		}
		b.WriteRune(r)
	}
	return b.String()
}

// A tool name is an identifier a model has to reproduce exactly, so one too
// long to carry is refused rather than cut: two names cut to the same length
// would be one name, and the one that survived would be a name its own server
// does not answer to.
func usableName(name string) string {
	if len(name) > maxSaid {
		return ""
	}
	// Cutting or stripping would leave a name the server does not answer to,
	// which is worse than not offering the tool at all. The rule is the one
	// tameTo goes by: a name has to be printable as it stands, and a name
	// carrying bytes that are not text is not a name a model could reproduce.
	for _, r := range name {
		if !strconv.IsPrint(r) || r == utf8.RuneError {
			return ""
		}
	}
	return name
}

// quoted says a name goose would not carry, in a form that cannot be mistaken
// for one it did: what made the name unusable reads as an escape rather than
// moving the cursor, and stays in, so the name is not reported as the name of
// some other tool that happens to be left once it is taken out. A name too
// long to print whole is measured rather than cut, for the same reason a name
// too long to carry is refused rather than cut, and the second return says
// which of the two came back.
func quoted(name string) (string, bool) {
	// The measure is what it costs to say, not what the name was: a short name
	// of nothing but escapes is four times itself written out, and a line of
	// those is the flood this ceiling is here to stop.
	if q := strconv.Quote(name); len(q) <= maxSaid {
		return q, true
	}
	return fmt.Sprintf("a name of %d bytes", len(name)), false
}

// unreachable tells a server that is not there from one that answered, even
// with a refusal, and from one that was only slow: a server busy with a long
// call is still there, and its tools are not taken away for it.
func unreachable(err error) bool {
	var answered *rpcError
	return !errors.As(err, &answered) &&
		!errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled)
}

func (c *Client) Tools(ctx context.Context) ([]llm.Tool, error) {
	var out []llm.Tool
	var refused []string
	raw := map[string]json.RawMessage{}
	cursor := ""
	truncated, full := false, false
	examined, weight := 0, 0
	seen := map[string]bool{}
	dropped := map[string]bool{}
	// refuse records a name goose will not carry, whatever it was about the
	// tool that decided that: a name it cannot hold and a schema it cannot
	// mirror are both the server's choice, and how many of either there are is
	// no more the server's than how many tools. Held on a ceiling of its own,
	// so a run of them cannot spend the catalogue saying so and leave no room
	// for the tools behind it, and checked before anything is kept, so what is
	// remembered to tell one refusal from another is bounded by that number
	// too.
	refuse := func(name string) {
		if len(refused) >= maxRefused {
			truncated = true
			return
		}
		offered, exact := quoted(name)
		if exact {
			if dropped[offered] {
				return
			}
			dropped[offered] = true
		}
		refused = append(refused, offered)
	}
	for pages := 0; pages < maxToolPages; pages++ {
		var result struct {
			Tools      []json.RawMessage `json:"tools"`
			NextCursor string            `json:"nextCursor"`
		}
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		if err := c.call(ctx, "tools/list", params, &result); err != nil {
			if unreachable(err) {
				// A server that is not there serves none of its tools, and
				// offering them sends every call to one that fails.
				c.mu.Lock()
				c.tools, c.raw, c.refused, c.truncated = nil, nil, nil, false
				c.mu.Unlock()
			}
			return nil, err
		}
		for _, entry := range result.Tools {
			// A catalogue is free to list one tool more than once, so reading
			// more entries than goose will hold is ordinary. Reading several
			// times as many is a server that is not going to stop, and the work
			// of reading them is its to choose otherwise.
			if examined++; examined > maxEntries {
				truncated, full = true, true
				break
			}
			var t struct {
				Name         string          `json:"name"`
				Description  string          `json:"description"`
				InputSchema  json.RawMessage `json:"inputSchema"`
				OutputSchema json.RawMessage `json:"outputSchema"`
			}
			if json.Unmarshal(entry, &t) != nil || t.Name == "" {
				continue
			}
			name := usableName(t.Name)
			// A name offered twice is one tool. Keeping both copies shows the
			// first to the model while the second decides what goes on the
			// wire, so the definition on display would not be the one in use.
			// The two sets are kept apart because the two names are judged
			// differently: one that cannot be carried must not stand in front
			// of one that can just because they read alike once the characters
			// that made it unusable are taken out.
			if name == "" {
				// Cutting it would leave a name the server does not answer to,
				// so it is not offered — and saying so is the whole point.
				refuse(t.Name)
				continue
			}
			if seen[name] {
				continue
			}
			schema := t.InputSchema
			if len(schema) == 0 || string(schema) == "null" {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			// Only a transport goose mirrors over will be asked to mirror, and
			// a tool called as it arrives is callable whatever its annotations
			// say. Judging every tool by a rule only some of them are held to
			// takes working tools off the ones that are not.
			if c.mirrors {
				if _, err := annotated(schema); err != nil {
					refuse(t.Name)
					continue
				}
			}
			// The name is held from here, and this is where there has to be room
			// for it. A copy of a tool already held needs none, and a tool goose
			// will not serve must not stand in the room kept for one it will.
			// How many entries there are is the server's to choose and bounded
			// above; how large they are is its to choose too, and two thousand
			// tools can be sent as a gigabyte of them.
			if len(seen) >= maxTools || weight+len(entry) > maxWeight {
				truncated, full = true, true
				break
			}
			weight += len(entry)
			seen[name] = true
			t.Name = name
			raw[t.Name] = append(json.RawMessage(nil), entry...)
			out = append(out, llm.Tool{
				Name: t.Name, Description: t.Description, Schema: schema,
				OutputSchema: t.OutputSchema,
			})
		}
		// Only the tools being full ends the walk. Running out of room to say
		// what was refused is not the end of the catalogue, and the pages
		// behind it are still read.
		if full {
			break
		}
		if result.NextCursor == "" {
			break
		}
		if pages == maxToolPages-1 {
			truncated = true
		}
		cursor = result.NextCursor
	}
	// The rest is not served, and saying so is what keeps this from being a
	// silent truncation. Refusing the server outright would take a working one
	// out of the proxy, and the proxy with it.
	c.mu.Lock()
	c.truncated = truncated
	c.mu.Unlock()
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

// Truncated reports a catalogue longer than goose will hold, so that a status
// table can say the list it shows is part of one.
func (c *Client) Truncated() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.truncated
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
				// The text is what the model reads. Sending the raw block with
				// it would deliver the same words twice and, on a wire with no
				// shape for a foreign block, fail the whole call under a
				// strict policy.
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
