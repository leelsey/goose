// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

type state struct {
	session     string
	owner       string
	id          string
	toolIdx     int
	openTool    bool
	finished    bool
	turnOpen    bool
	sawContent  bool
	lastWasTool bool
}

// began records a session the binary has just issued, against the caller it was
// issued to, so that nobody else can take it up later.
func (st *state) began(p *Provider, id string) {
	if id == "" || id == st.session {
		return
	}
	st.session = id
	p.remember(id, st.owner)
}

// toolFrame mints the next tool-call index and opens a frame carrying it.
func (st *state) toolFrame(id, name string, args json.RawMessage) []*llm.Frame {
	idx := st.toolIdx
	st.toolIdx++
	st.lastWasTool = true
	return []*llm.Frame{{Delta: []llm.Part{&llm.ToolUse{
		ID: id, Name: name, Args: args, Index: &idx,
	}}}}
}

func (st *state) ending() llm.StopReason {
	if st.finished {
		return llm.StopNone
	}
	st.finished = true
	if st.lastWasTool {
		return llm.StopToolUse
	}
	return llm.StopEnd
}

type parser struct {
	ctx      context.Context
	provider string
	owner    *Provider
	proc     *spawn.Process
	lines    *wirecore.Lines
	state    *state
	parse    func(p *Provider, line []byte, st *state) ([]*llm.Frame, error)

	release func()

	queue  []*llm.Frame
	done   bool
	closed bool
}

func (pr *parser) next() (*llm.Frame, error) {
	for {
		if len(pr.queue) > 0 {
			f := pr.queue[0]
			pr.queue = pr.queue[1:]
			return f, nil
		}
		if pr.done {
			return nil, nil
		}
		if !pr.lines.Scan() {
			pr.done = true
			if err := pr.lines.Err(); err != nil {
				if typed := llm.AsError(err); typed != nil && typed.Kind != llm.KindInternal {
					return nil, typed
				}
				return nil, httpcore.ReadError(pr.ctx, pr.provider, err)
			}
			if err := pr.exit(); err != nil {
				return nil, err
			}
			if stop := pr.state.ending(); stop != llm.StopNone {
				return &llm.Frame{Stop: stop, TurnEnd: true}, nil
			}
			return nil, nil
		}
		line := pr.lines.Bytes()
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		frames, err := pr.parse(pr.owner, line, pr.state)
		if err != nil {
			pr.done = true
			return nil, err
		}
		pr.queue = append(pr.queue, frames...)
	}
}

func (pr *parser) exit() error {
	err := pr.proc.Wait()
	pr.owner.untrack(pr.proc)
	if err == nil {
		return nil
	}
	detail := pr.proc.Stderr()
	if detail == "" {
		detail = err.Error()
	}
	kind := llm.KindUpstream
	if strings.Contains(strings.ToLower(detail), "not found") || pr.proc.ExitCode() == 127 {
		kind = llm.KindNotFound
	}
	return &llm.Error{
		Kind: kind, Provider: pr.provider, Sent: true,
		Message: pr.provider + ": the agent binary exited " + exitText(pr.proc.ExitCode()),
		Body:    truncate(detail, 2000), Err: err,
	}
}

func (pr *parser) close() error {
	if pr.closed {
		return nil
	}
	pr.closed = true
	if pr.release != nil {
		pr.release()
	}
	pr.owner.untrack(pr.proc)
	return pr.proc.Close()
}

func exitText(code int) string {
	if code < 0 {
		return "without a status"
	}
	return "with status " + strconv.Itoa(code)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func unmarshal(line []byte, v any, provider string) error {
	if err := json.Unmarshal(line, v); err != nil {
		return &llm.Error{
			Kind: llm.KindUpstream, Provider: provider, Sent: true, Code: llm.WireFrameMalformed,
			Message: "unreadable output line: " + err.Error(), Err: err,
		}
	}
	return nil
}
