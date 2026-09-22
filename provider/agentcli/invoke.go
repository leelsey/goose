// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"strconv"
	"strings"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

type invocation struct {
	Model      string
	Prompt     string
	System     string
	SystemFile string
	Session    string
	Owner      string
	Stream     bool
}

type kindSpec struct {
	binary         string
	systemMode     SystemMode
	sessions       bool
	sandbox        bool
	restrictsTools bool
	args           func(p *Provider, in invocation) []string
	parse          func(p *Provider, line []byte, st *state) ([]*llm.Frame, error)
}

var kinds = map[Kind]kindSpec{
	Claude: {
		binary: "claude", systemMode: SystemFile, sessions: true,
		restrictsTools: true, args: claudeArgs, parse: claudeParse,
	},
	Codex: {
		binary: "codex", systemMode: SystemPrompt, sessions: true,
		sandbox: true, args: codexArgs, parse: codexParse,
	},
	Antigravity: {
		binary: "gemini", systemMode: SystemPrompt, sessions: false,
		args: antigravityArgs, parse: antigravityParse,
	},
	Copilot: {
		binary: "copilot", systemMode: SystemPrompt, sessions: true,
		restrictsTools: true, args: copilotArgs, parse: copilotParse,
	},
}

func (p *Provider) prepare(a *llm.Attempt) (invocation, llm.Diagnostics, error) {
	var diags llm.Diagnostics
	system, prompt, fdiags := flatten(a.Request, p.ID())
	diags = append(diags, fdiags...)
	in := invocation{
		Model:   p.model(a),
		Prompt:  prompt,
		System:  system,
		Session: a.Session,
		Stream:  true,
	}
	if prompt == "" {
		return in, diags, llm.Errorf(llm.KindInvalidRequest, "no prompt survived translation")
	}
	diags = append(diags, llm.UnsentSampling(a.Request, p.ID())...)
	if p.toolsUnrestricted() {
		diags = append(diags, llm.Diagnostic{
			Code: llm.PolicyAgentUnrestricted, Class: llm.ClassPolicy, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(),
			Detail: "this binary runs its own tools on the host for this call; goose cannot restrict them",
		})
	}
	if !kinds[p.opts.Kind].sessions {
		in.Session = ""
	}
	if in.Session != "" {
		by := owner(a.Credential)
		switch {
		case !safeSession(in.Session):
			in.Session = ""
			diags = append(diags, llm.Diagnostic{
				Code: llm.CapSessionElsewhere, Class: llm.ClassPolicy, Stage: llm.StageRequest,
				Severity: llm.SeverityLoss, Provider: p.ID(), Field: "session",
				Detail: "this conversation handle is not one this binary could have issued, so a new conversation was started",
			})
		case !p.resumable(in.Session, by):
			in.Session = ""
			diags = append(diags, llm.Diagnostic{
				Code: llm.CapSessionElsewhere, Class: llm.ClassPolicy, Stage: llm.StageRequest,
				Severity: llm.SeverityLoss, Provider: p.ID(), Field: "session",
				Detail: "this conversation was begun by another caller, so a new one was started rather than taking up theirs",
			})
		default:
			in.Owner = by
		}
	}
	if in.Session == "" {
		in.Owner = owner(a.Credential)
	}

	mode := p.opts.SystemMode
	switch {
	case system == "" || mode == SystemOff:
		in.System = ""
	case (mode == SystemFile || mode == SystemArg) && p.opts.Kind == Claude:
		path, err := p.systemFile(system)
		if err != nil {
			return in, diags, llm.Errorf(llm.KindInternal, "system prompt file: %v", err)
		}
		in.SystemFile = path
	case mode == SystemAppend && p.opts.Kind == Claude:
		diags = append(diags, llm.Diagnostic{
			Code: llm.PolicyPromptOnArgv, Class: llm.ClassPolicy, Stage: llm.StageRequest,
			Severity: llm.SeverityLoss, Provider: p.ID(), Field: "system",
			Detail: "append mode passes the system prompt on the command line, where any local user can read it",
		})
	case mode == SystemFile, mode == SystemArg, mode == SystemAppend:
		if p.opts.Kind != Claude {
			in.Prompt = "System: " + system + "\n\n" + in.Prompt
			in.System = ""
			diags = append(diags, systemFolded(p.ID()))
		}
	default:
		in.Prompt = "System: " + system + "\n\n" + in.Prompt
		in.System = ""
		if p.opts.Kind != Claude {
			diags = append(diags, systemFolded(p.ID()))
		}
	}

	return in, diags, nil
}

func (p *Provider) toolsUnrestricted() bool {
	if !kinds[p.opts.Kind].restrictsTools {
		return true
	}
	switch p.opts.Kind {
	case Claude:
		return strings.EqualFold(p.opts.Tools, "default")
	case Copilot:
		return strings.EqualFold(p.opts.Tools, "all")
	}
	return false
}

func systemFolded(id string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropSystemPrompt, Class: llm.ClassCapability, Stage: llm.StageRequest,
		Severity: llm.SeverityInfo, Provider: id,
		Detail: "this binary has no system channel, so the prompt was folded into the conversation and is billed as fresh input every call",
	}
}

func (p *Provider) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	in, diags, err := p.prepare(a)
	if err != nil {
		return nil, err
	}
	spec := kinds[p.opts.Kind]
	args := spec.args(p, in)

	if max := a.Limits.MaxRequestBytes; max > 0 && int64(len(in.Prompt)+len(in.System)) > max {
		return nil, &llm.Error{
			Kind: llm.KindInvalidRequest, Code: llm.PolicyBodyTooLarge, Provider: p.ID(),
			Message: "the request is larger than this route allows",
			Sent:    false, Retryable: true,
		}
	}

	runCtx, release := context.WithTimeout(ctx, p.opts.Timeout)

	proc, err := spawn.Start(runCtx, spawn.Options{
		Binary: p.opts.Binary, Args: args, Dir: p.opts.Dir, Env: p.opts.Env,
		Stdin: []byte(in.Prompt), Grace: p.opts.Grace,
	})
	if err != nil {
		release()
		return nil, &llm.Error{Kind: llm.KindTransport, Provider: p.ID(), Message: err.Error(), Err: err}
	}
	p.track(proc)

	st := &state{owner: in.Owner}
	pr := &parser{
		ctx: runCtx, provider: p.ID(), proc: proc, state: st, parse: spec.parse, owner: p,
		lines:   wirecore.NewLines(httpcore.Capped(proc.Stdout(), a.Limits.MaxResponseBytes, p.ID()), a.Limits.MaxFrameBytes),
		release: release,
	}
	stream := llm.NewStream(
		llm.StreamMeta{Provider: p.ID(), Model: in.Model, Requested: a.Request.Model, Created: p.now()},
		pr.next,
		pr.close,
	)
	for _, d := range diags {
		stream.Note(d)
		a.Note(d)
	}
	return stream, nil
}

func claudeArgs(p *Provider, in invocation) []string {
	args := []string{"-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages"}
	if in.Model != "" {
		args = append(args, "--model", in.Model)
	}
	if in.Session != "" {
		args = append(args, "--resume", in.Session)
	}
	switch {
	case in.SystemFile != "":
		args = append(args, "--system-prompt-file", in.SystemFile)
	case in.System != "" && p.opts.SystemMode == SystemAppend:
		args = append(args, "--append-system-prompt", in.System)
	case in.System != "":
		args = append(args, "--system-prompt", in.System)
	}
	if p.opts.MaxTurns > 0 {
		args = append(args, "--max-turns", strconv.Itoa(p.opts.MaxTurns))
	}
	if !strings.EqualFold(p.opts.Tools, "default") {
		args = append(args, "--tools", p.opts.Tools)
	}
	return args
}

func codexArgs(p *Provider, in invocation) []string {
	args := []string{"exec", "--json", "--skip-git-repo-check"}
	sandbox := p.opts.Sandbox
	if sandbox == "" {
		sandbox = "read-only"
	}
	args = append(args, "--sandbox", sandbox)
	if in.Model != "" {
		args = append(args, "--model", in.Model)
	}
	if in.Session != "" {
		return append(args, "resume", in.Session, "--", "-")
	}
	return append(args, "--", "-")
}

func antigravityArgs(p *Provider, in invocation) []string {
	args := []string{"-p", "-", "--output-format", "stream-json"}
	if in.Model != "" {
		args = append(args, "--model", in.Model)
	}
	return args
}

func copilotArgs(p *Provider, in invocation) []string {
	args := []string{"-p", "-", "--output-format", "json", "--no-ask-user"}
	if strings.EqualFold(p.opts.Tools, "all") {
		args = append(args, "--allow-all-tools")
	}
	if in.Model != "" {
		args = append(args, "--model", in.Model)
	}
	if in.Session != "" {
		args = append(args, "--resume="+in.Session)
	}
	return args
}
