// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package agentcli drives a local agent CLI as an upstream.
package agentcli

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

type Kind string

const (
	Claude      Kind = "claude"
	Codex       Kind = "codex"
	Antigravity Kind = "antigravity"
	Copilot     Kind = "copilot"
)

// Every call forks a whole agent CLI, which costs hundreds of megabytes while
// it runs, so what bounds this is the host's memory rather than its cores.
// A request past the limit is shed, which is what the route's next link is for.
// A zero MaxConcurrent takes this default; a negative one removes the gate.
const DefaultMaxConcurrent = 4

type SystemMode string

const (
	SystemFile   SystemMode = "file"
	SystemArg    SystemMode = "arg"
	SystemAppend SystemMode = "append"
	SystemPrompt SystemMode = "prompt"
	SystemOff    SystemMode = "off"
)

func (m SystemMode) known() bool {
	switch m {
	case SystemFile, SystemArg, SystemAppend, SystemPrompt, SystemOff:
		return true
	}
	return false
}

type Options struct {
	ID            string
	Kind          Kind
	Binary        string
	Model         string
	Dir           string
	Env           []string
	SystemMode    SystemMode
	Tools         string
	MaxTurns      int
	Sandbox       string
	MaxConcurrent int
	Timeout       time.Duration
	Grace         time.Duration
	TempDir       string
	Now           func() time.Time
}

type Provider struct {
	opts Options
	env  llm.Env
	gate chan struct{}

	mu       sync.Mutex
	live     map[*spawn.Process]struct{}
	sessions map[string]string
	order    []string
	vault    *spawn.Vault
	closed   bool
}

func (p *Provider) systemFile(content string) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return "", llm.Errorf(llm.KindInternal, "%s: provider is closed", p.ID())
	}
	if p.vault == nil {
		v, err := spawn.OpenVault(p.opts.TempDir, "goose-agentcli-")
		if err != nil {
			return "", err
		}
		p.vault = v
	}
	return p.vault.File("system", content)
}

var defaultEnvAllowlist = []string{
	"HOME", "PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TMPDIR",
	"XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
	"APPDATA", "LOCALAPPDATA", "USERPROFILE", "SystemRoot", "TEMP", "TMP",
	"http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
	"NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
}

func DefaultEnvAllowlist() []string {
	return append([]string(nil), defaultEnvAllowlist...)
}

func New(o Options) (*Provider, error) {
	if o.Kind == "" {
		return nil, llm.Errorf(llm.KindInvalidRequest, "agentcli: kind is required")
	}
	if _, ok := kinds[o.Kind]; !ok {
		return nil, llm.Errorf(llm.KindInvalidRequest, "agentcli: unknown kind %q", o.Kind)
	}
	if o.ID == "" {
		o.ID = string(o.Kind)
	}
	if o.Binary == "" {
		o.Binary = kinds[o.Kind].binary
	}
	if o.SystemMode == "" {
		o.SystemMode = kinds[o.Kind].systemMode
	}
	if !o.SystemMode.known() {
		return nil, llm.Errorf(llm.KindInvalidRequest,
			"agentcli: unknown system mode %q (known: file, arg, append, prompt, off)", o.SystemMode)
	}
	if o.Sandbox != "" && !kinds[o.Kind].sandbox {
		return nil, llm.Errorf(llm.KindInvalidRequest,
			"agentcli: the %s binary takes no sandbox setting, so %q cannot be enforced", o.Kind, o.Sandbox)
	}
	if o.Timeout <= 0 {
		o.Timeout = 5 * time.Minute
	}
	if o.MaxConcurrent == 0 {
		o.MaxConcurrent = DefaultMaxConcurrent
	}
	p := &Provider{opts: o, live: map[*spawn.Process]struct{}{}}
	if o.MaxConcurrent > 0 {
		p.gate = make(chan struct{}, o.MaxConcurrent)
	}
	return p, nil
}

func (p *Provider) ID() string { return p.opts.ID }

func (p *Provider) Bind(env llm.Env) error { p.env = env; return nil }

func (p *Provider) Capabilities() llm.Capabilities {
	k := kinds[p.opts.Kind]
	return llm.Capabilities{
		Streaming: true,
		Usage: llm.UsageShape{
			StreamUsage: true,
			Cost:        p.opts.Kind == Claude,
			MultiTurn:   true,
		},
		Tools:        false,
		ToolResults:  true,
		SystemPrompt: p.opts.SystemMode != SystemOff,
		Sessions:     k.sessions,
		Thinking:     true,
		Candidates:   1,
		Ceiling:      llm.CeilingNone,
	}
}

func (p *Provider) Acquire(ctx context.Context) (func(), error) {
	if p.gate == nil {
		return func() {}, nil
	}
	select {
	case p.gate <- struct{}{}:
		var once sync.Once
		return func() { once.Do(func() { <-p.gate }) }, nil
	case <-ctx.Done():
		return func() {}, &llm.Error{Kind: llm.KindCancelled, Provider: p.ID(), Message: ctx.Err().Error()}
	default:
		return func() {}, &llm.Error{
			Kind: llm.KindSaturated, Provider: p.ID(), Retryable: true,
			Message: fmt.Sprintf("%s: all %d slots are busy", p.ID(), p.opts.MaxConcurrent),
		}
	}
}

func (p *Provider) Close(context.Context) error {
	p.mu.Lock()
	live := make([]*spawn.Process, 0, len(p.live))
	for proc := range p.live {
		live = append(live, proc)
	}
	p.live = map[*spawn.Process]struct{}{}
	p.mu.Unlock()
	for _, proc := range live {
		_ = proc.Close()
	}
	p.mu.Lock()
	vault := p.vault
	p.vault, p.closed = nil, true
	p.mu.Unlock()
	if vault != nil {
		return vault.Close()
	}
	return nil
}

func (p *Provider) track(proc *spawn.Process) {
	p.mu.Lock()
	p.live[proc] = struct{}{}
	p.mu.Unlock()
}

func (p *Provider) untrack(proc *spawn.Process) {
	p.mu.Lock()
	delete(p.live, proc)
	p.mu.Unlock()
}

func (p *Provider) now() time.Time {
	switch {
	case p.opts.Now != nil:
		return p.opts.Now()
	case p.env.Now != nil:
		return p.env.Now()
	default:
		return time.Now()
	}
}

func (p *Provider) model(a *llm.Attempt) string {
	switch {
	case a.Model != "":
		return a.Model
	case p.opts.Model != "":
		return p.opts.Model
	case a.Request != nil:
		return a.Request.Model
	default:
		return ""
	}
}

func (p *Provider) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	s, err := p.Stream(ctx, a)
	if err != nil {
		return nil, err
	}
	resp, ferr := wirecore.Fold(s, wirecore.FoldOptions{
		MaxBufferBytes: a.Limits.MaxBufferBytes, Recorder: a.Recorder,
	})
	if resp != nil {
		resp.Provider = p.ID()
		resp.Requested = a.Request.Model
	}
	return resp, ferr
}

func flatten(req *llm.Request, provider string) (system, prompt string, diags llm.Diagnostics) {
	sys, msgs := req.SystemAndMessages()
	system = llm.TextOf(sys)
	diags = append(diags, llm.Flattened(sys, provider,
		"this backend takes a system prompt as text, so what was not text did not travel")...)

	var b strings.Builder
	for _, m := range msgs {
		text := llm.TextOf(m.Content)
		if refusal := llm.RefusalOf(m.Content); refusal != "" {
			text = strings.TrimSpace(text + "\n" + refusal)
		}
		for _, part := range m.Content {
			if tr, ok := part.(*llm.ToolResult); ok {
				text = strings.TrimSpace(text + "\n" + llm.TextOf(tr.Content))
				diags = append(diags, llm.Flattened(tr.Content, provider,
					"this backend takes a tool result as text, so what was not text did not travel")...)
			}
		}
		diags = append(diags, llm.Flattened(carriedAsText(m.Content), provider,
			"this backend takes a conversation as text, so this block did not travel")...)
		if text == "" {
			diags = append(diags, llm.DroppedMessage(m.Role, provider,
				"this turn carried nothing this backend can read as text"))
			continue
		}
		switch m.Role {
		case llm.RoleSystem, llm.RoleDeveloper:
			b.WriteString("System: ")
		case llm.RoleAssistant:
			b.WriteString("Assistant: ")
		case llm.RoleTool:
			b.WriteString("Tool result: ")
		default:
			b.WriteString("User: ")
		}
		b.WriteString(text)
		b.WriteString("\n\n")
	}
	return system, strings.TrimSpace(b.String()), diags
}

func carriedAsText(parts []llm.Part) []llm.Part {
	out := make([]llm.Part, 0, len(parts))
	for _, p := range parts {
		switch p.Kind() {
		case llm.KindText, llm.KindRefusal, llm.KindToolResult:
			continue
		}
		out = append(out, p)
	}
	return out
}
