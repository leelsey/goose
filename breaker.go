// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"sync"
	"time"

	"github.com/leelsey/goose/llm"
)

type Breaker struct {
	Failures int
	Cooldown time.Duration
}

const (
	DefaultBreakerFailures = 5
	DefaultBreakerCooldown = 30 * time.Second
)

type breakers struct {
	policy Breaker
	now    func() time.Time

	mu     sync.Mutex
	state  map[string]*breakerState
	probes probe
}

// A probe names the one call a breaker lets through after its cooldown, and
// only that call gives the slot back. Another, let through while the breaker
// was still closed, would otherwise free it by finishing while the probe runs
// and admit a second beside it. Zero is no probe.
type probe uint64

type breakerState struct {
	failures int
	openedAt time.Time
	probe    probe
}

func newBreakers(p Breaker, now func() time.Time) *breakers {
	if p.Failures <= 0 {
		return nil
	}
	if p.Cooldown <= 0 {
		p.Cooldown = DefaultBreakerCooldown
	}
	if now == nil {
		now = time.Now
	}
	return &breakers{policy: p, now: now, state: map[string]*breakerState{}}
}

// An allowed caller given a probe holds the slot until succeeded, failed or
// abandoned.
func (b *breakers) allow(provider string) (probe, bool) {
	if b == nil {
		return 0, true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.state[provider]
	if s == nil || s.failures < b.policy.Failures {
		return 0, true
	}
	if b.now().Sub(s.openedAt) < b.policy.Cooldown {
		return 0, false
	}
	if s.probe != 0 {
		return 0, false
	}
	b.probes++
	s.probe = b.probes
	return s.probe, true
}

func (b *breakers) abandoned(provider string, p probe) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if s := b.state[provider]; s != nil {
		s.release(p)
	}
}

func (s *breakerState) release(p probe) {
	if p != 0 && s.probe == p {
		s.probe = 0
	}
}

func (b *breakers) succeeded(provider string) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.state, provider)
}

func (b *breakers) failed(provider string, p probe) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.state[provider]
	if s == nil {
		s = &breakerState{}
		b.state[provider] = s
	}
	s.release(p)
	s.failures++
	if s.failures >= b.policy.Failures {
		s.openedAt = b.now()
	}
}

func breakerOpen(provider string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.PolicyBreakerOpen, Class: llm.ClassPolicy, Stage: llm.StagePlan,
		Severity: llm.SeverityInfo, Provider: provider,
		Detail: "this backend has failed its last several calls, so the attempt was not made",
	}
}
