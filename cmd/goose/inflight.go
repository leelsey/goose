// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/leelsey/goose/mcp"
)

// A call cut off at shutdown has already been sent and billed, and it books its
// row only once its cancellation has unwound. That takes moments; the wait is
// bounded so that a call which never unwinds cannot hold the process open.
const settleFor = 5 * time.Second

// inflight counts the calls still running, so that the usage ledger is closed
// and the summary printed only after they have booked what they spent.
type inflight struct {
	// closing turns away a call that arrives once settle has begun: a
	// WaitGroup may not be added to from zero while it is waited on, and the
	// call's row would land in a ledger that is about to close.
	mu      sync.Mutex
	closing bool
	open    int
	wg      sync.WaitGroup
}

func (f *inflight) enter() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closing {
		return false
	}
	f.open++
	f.wg.Add(1)
	return true
}

func (f *inflight) leave() {
	f.mu.Lock()
	f.open--
	f.mu.Unlock()
	f.wg.Done()
}

func (f *inflight) running() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.open
}

func (f *inflight) settle(within time.Duration, logger *slog.Logger) {
	f.mu.Lock()
	f.closing = true
	f.mu.Unlock()

	done := make(chan struct{})
	go func() {
		f.wg.Wait()
		close(done)
	}()
	timer := time.NewTimer(within)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
		logger.Warn("calls were still running at exit, so what they spent may be missing from the usage log",
			"waited", within)
	}
}

func (f *inflight) handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !f.enter() {
			http.Error(w, "goose is shutting down", http.StatusServiceUnavailable)
			return
		}
		defer f.leave()
		next.ServeHTTP(w, r)
	})
}

func (f *inflight) tools(list []mcp.Tool) []mcp.Tool {
	out := make([]mcp.Tool, len(list))
	for i, t := range list {
		call := t.Call
		t.Call = func(ctx context.Context, args json.RawMessage) (*mcp.Result, error) {
			if !f.enter() {
				return nil, errors.New("goose is shutting down")
			}
			defer f.leave()
			return call(ctx, args)
		}
		out[i] = t
	}
	return out
}

// Shutdown lets the requests in flight finish within the budget, and Close
// then cuts the rest off, at once if again delivers. Neither waits for a
// cut-off handler to unwind, so settle does, before run closes the ledger
// behind it. A stream open for the whole budget is otherwise a pause nobody
// explained, an MCP client's idle listen stream among them.
func shutdown(server *http.Server, calls *inflight, budget time.Duration, logger *slog.Logger, again <-chan os.Signal) {
	ctx, cancel := context.WithTimeout(context.Background(), budget)
	defer cancel()
	if n := calls.running(); n > 0 {
		logger.Info("stopping: waiting for open requests and streams to finish; interrupt again to cut them off",
			"open", n, "up_to", budget)
	}
	go func() {
		select {
		case <-again:
			cancel()
		case <-ctx.Done():
		}
	}()
	if err := server.Shutdown(ctx); err != nil {
		_ = server.Close()
	}
	calls.settle(settleFor, logger)
}
