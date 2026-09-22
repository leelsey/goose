// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
)

// A cachePoint block marks everything before it as cacheable. Converse takes
// four of them per request.
func cachePoint() map[string]any {
	return map[string]any{"cachePoint": map[string]any{"type": "default"}}
}

type breakpoints struct{ left int }

func (b *breakpoints) take() bool {
	if b.left <= 0 {
		return false
	}
	b.left--
	return true
}

// caching decides how many cache points this request may spend, and says why
// where it may spend none.
func (p *Provider) caching(a *llm.Attempt, tools []llm.Tool, diags *llm.Diagnostics) *breakpoints {
	wanted, long := cacheHints(a.Request, tools)
	if !wanted {
		return &breakpoints{}
	}
	if !cacheable(p.model(a)) {
		*diags = append(*diags, llm.Diagnostic{
			Code: llm.DropCacheHint, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(),
			Detail: "this model does not serve prompt caching, so the breakpoints were not sent",
		})
		return &breakpoints{}
	}
	if long {
		*diags = append(*diags, llm.Diagnostic{
			Code: llm.DropCacheHint, Class: llm.ClassCapability, Stage: llm.StageRequest,
			Severity: llm.SeverityInfo, Provider: p.ID(), Field: "cache.ttl",
			Detail: "a cache point here keeps the backend's own window, so the longer one asked for was not sent",
		})
	}
	return &breakpoints{left: maxBreakpoints}
}

// cacheHints reports whether the request asks to cache anything, and whether any
// of those asks names a window longer than the one a cache point keeps.
func cacheHints(req *llm.Request, tools []llm.Tool) (wanted, long bool) {
	walk := func(h *llm.CacheHint) {
		if h == nil {
			return
		}
		wanted = true
		if h.TTL >= time.Hour {
			long = true
		}
	}
	walk(req.SystemCache)
	for _, m := range req.Messages {
		walk(m.Cache)
	}
	for _, t := range tools {
		walk(t.Cache)
	}
	return wanted, long
}

// Converse accepts a cachePoint only on the models that serve prompt caching;
// on any other model the block is a request error, so this names them rather
// than guessing. A model missing from the list loses the cache, which is what
// happened before caching was sent at all; a model wrongly on it loses the
// whole request.
var caches = []string{
	"claude-3-7-sonnet",
	"claude-3-5-haiku",
	"claude-sonnet-4",
	"claude-opus-4",
	"claude-haiku-4",
	"nova-micro",
	"nova-lite",
	"nova-pro",
	"nova-premier",
}

func cacheable(model string) bool {
	id := strings.ToLower(model)
	for _, name := range caches {
		if strings.Contains(id, name) {
			return true
		}
	}
	return false
}
