// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"time"

	"github.com/leelsey/goose/llm"
)

func BookDiscards(a *llm.Attempt) Attempted {
	if a == nil || a.Book == nil {
		return nil
	}
	return func(e *llm.Error, elapsed time.Duration) {
		rec := llm.Attempted{
			Provider: a.Provider, Model: a.Model, Route: a.Route, Requested: requested(a),
			Index: a.Index, Sent: true, Outcome: llm.OutcomeDiscarded,
			Elapsed: elapsed.Milliseconds(),
		}
		if e != nil {
			rec.Err, rec.Status = e.Message, e.Status
		}
		a.BookExtra(rec)
	}
}

func requested(a *llm.Attempt) string {
	if a.Request == nil {
		return ""
	}
	return a.Request.Model
}
