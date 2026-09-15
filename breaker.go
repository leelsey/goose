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

	mu    sync.Mutex
	state map[string]*breakerState
}

type breakerState struct {
	failures int
	openedAt time.Time
	probing  bool
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

// An allowed caller holds the probe slot until succeeded, failed or abandoned.
func (b *breakers) allow(provider string) bool {
	if b == nil {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	s := b.state[provider]
	if s == nil || s.failures < b.policy.Failures {
		return true
	}
	if b.now().Sub(s.openedAt) < b.policy.Cooldown {
		return false
	}
	if s.probing {
		return false
	}
	s.probing = true
	return true
}

func (b *breakers) abandoned(provider string) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if s := b.state[provider]; s != nil {
		s.probing = false
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

func (b *breakers) failed(provider string) {
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
	s.probing = false
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
