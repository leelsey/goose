// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"

	"github.com/leelsey/goose/llm"
)

// A session id is a handle to a conversation that is still on this host, with
// its history and its working directory. It reaches goose as a string in a
// request body, so two things are asked of it before it is acted on.
//
// First, it must look like a handle. It goes into the argument list of a child
// process, and for codex it goes there as a bare positional before the "--"
// terminator, where a value beginning with a dash is in the flag-parsing region
// and could be read as one.
//
// Second, it must belong to whoever is presenting it. Sessions are remembered
// against the credential they were issued to, and one presented by anybody else
// starts a new conversation instead of resuming that one. Where callers carry
// no credential of their own — a proxy with a single key, which is every caller
// proving the same thing — they are one caller as far as this can tell, and the
// check has nothing to separate them by.
const maxSessions = 1024

func safeSession(id string) bool {
	if id == "" || len(id) > 256 || id[0] == '-' {
		return false
	}
	for i := range len(id) {
		c := id[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-', c == '_', c == '.', c == ':':
		default:
			return false
		}
	}
	return true
}

func owner(c llm.Credential) string {
	if c.Empty() {
		return ""
	}
	h := sha256.New()
	h.Write([]byte(c.Key))
	names := make([]string, 0, len(c.Headers))
	for k := range c.Headers {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		h.Write([]byte("\x00" + k + "\x00" + c.Headers[k]))
	}
	return hex.EncodeToString(h.Sum(nil)[:8])
}

// remember records which caller a session was issued to. Past the cap the
// oldest is dropped rather than the whole table: emptying it would let anyone
// who can open a thousand conversations forget who owns the rest.
func (p *Provider) remember(session, by string) {
	if session == "" {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.sessions == nil {
		p.sessions = map[string]string{}
	}
	if _, held := p.sessions[session]; !held {
		for len(p.order) >= maxSessions {
			delete(p.sessions, p.order[0])
			p.order = p.order[1:]
		}
		p.order = append(p.order, session)
	}
	p.sessions[session] = by
}

// resumable reports whether this caller may take up that session.
//
// A handle nobody remembers issuing is refused to a caller that has a
// credential of its own, because between two callers this proxy can tell apart
// there is no reason to hand one the other's conversation on its say-so; the
// cost is a resumption lost after a restart, which is a new conversation rather
// than a wrong one. Where callers carry no credential they are one caller here,
// so there is nothing to protect and an unknown handle is taken at its word.
func (p *Provider) resumable(session, by string) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	issued, known := p.sessions[session]
	if !known {
		return by == ""
	}
	return issued == by
}
