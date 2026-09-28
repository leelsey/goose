// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"slices"
	"sort"
	"strings"

	"github.com/leelsey/goose/llm"
)

func Extras(body []byte, known map[string]bool, a *llm.Attempt, wires ...string) map[string]json.RawMessage {
	var top map[string]json.RawMessage
	if json.Unmarshal(body, &top) != nil {
		return nil
	}
	var out map[string]json.RawMessage
	var names []string
	for k, v := range top {
		if known[k] {
			continue
		}
		names = append(names, k)
		if out == nil {
			out = map[string]json.RawMessage{}
		}
		out[k] = v
	}
	if out == nil {
		return nil
	}
	if a == nil || a.Format == "" || slices.Contains(wires, a.Format) {
		return out
	}
	sort.Strings(names)
	a.Note(llm.Diagnostic{
		Code: llm.DropExtraField, Class: llm.ClassCapability, Stage: llm.StageResponse,
		Severity: llm.SeverityLoss, Count: len(names), Field: strings.Join(names, ","),
		Detail: "the backend answered with fields this wire does not model, and the client speaks another, so they were not passed on",
	})
	return nil
}

func Unmodelled(body []byte, known map[string]bool, field string, a *llm.Attempt) llm.Diagnostics {
	var obj map[string]json.RawMessage
	if json.Unmarshal(body, &obj) != nil {
		return nil
	}
	var names []string
	for k, v := range obj {
		// A field that is null carries nothing, so leaving it behind loses
		// nothing — and OpenAI puts a null logprobs on every choice of every
		// answer that did not ask for them.
		if !known[k] && string(v) != "null" {
			names = append(names, k)
		}
	}
	if len(names) == 0 {
		return nil
	}
	sort.Strings(names)
	d := llm.Diagnostic{
		Code: llm.DropExtraField, Class: llm.ClassContent, Stage: llm.StageResponse,
		Severity: llm.SeverityInfo, Count: len(names), Field: field + ":" + strings.Join(names, ","),
		Detail: "the backend answered with fields beside the answer that this model has nowhere to carry",
	}
	if a != nil {
		d.Provider, d.Model = a.Provider, a.Model
	}
	return llm.Diagnostics{d}
}

// Beside says, once a stream, which fields a backend puts beside its answer.
// They are a fact about the backend rather than about any one chunk, and said on
// every chunk they are a copy in the ledger row for every token of the answer.
type Beside struct{ said bool }

func (b *Beside) Note(body []byte, known map[string]bool, field string, a *llm.Attempt) llm.Diagnostics {
	if b.said {
		return nil
	}
	d := Unmodelled(body, known, field, a)
	if len(d) > 0 {
		b.said = true
	}
	return d
}
