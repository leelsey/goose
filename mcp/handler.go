// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

func (m *Manager) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /", m.rpc)
	return sameOrigin(atRoot(mux))
}

func atRoot(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "" {
			next.ServeHTTP(w, r)
			return
		}
		r2 := *r
		u := *r.URL
		u.Path = "/"
		r2.URL = &u
		next.ServeHTTP(w, &r2)
	})
}

const maxRPCBody = 8 << 20

func sameOrigin(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Origin") != "" {
			writeRPC(w, http.StatusForbidden, map[string]any{
				"jsonrpc": jsonrpcVersion, "id": nil,
				"error": map[string]any{"code": -32600, "message": "this endpoint does not serve browser origins"},
			})
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (m *Manager) rpc(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRPCBody+1))
	if err != nil {
		writeRPCError(w, nil, &rpcError{Code: -32700, Message: "the request body could not be read", status: http.StatusBadRequest})
		return
	}
	if len(body) > maxRPCBody {
		writeRPCError(w, nil, &rpcError{
			Code: -32600, Message: "the request is larger than this endpoint accepts",
			status: http.StatusRequestEntityTooLarge,
		})
		return
	}
	var req request
	if err := json.Unmarshal(body, &req); err != nil {
		writeRPCError(w, nil, &rpcError{Code: -32700, Message: "the request is not valid JSON", status: http.StatusBadRequest})
		return
	}

	if req.ID == nil {
		w.WriteHeader(http.StatusAccepted)
		return
	}

	seen := mirrored{
		Present: true,
		Version: r.Header.Get("MCP-Protocol-Version"),
		Method:  r.Header.Get("Mcp-Method"),
		Name:    r.Header.Get("Mcp-Name"),
		Params:  map[string]string{},
	}
	for name, values := range r.Header {
		if strings.HasPrefix(name, "Mcp-Param-") && len(values) > 0 {
			seen.Params[name] = values[0]
		}
	}
	if want, rerr, ok := m.listens(&req, seen); ok {
		if rerr != nil {
			writeRPCError(w, req.ID, rerr)
			return
		}
		m.streamOver(w, r, req.ID, want)
		return
	}
	if token := requestToken(req.Params); token != nil && req.Method == "tools/call" && req.ID != nil {
		if agreement, modern := readMeta(req.Params); modern {
			if _, rerr := checkModern(&req, agreement, seen); rerr != nil {
				writeRPCError(w, req.ID, rerr)
				return
			}
		}
		m.answerStreamed(w, r, &req, seen, token)
		return
	}
	result, rerr := m.dispatch(r.Context(), &req, seen)
	if rerr != nil {
		writeRPCError(w, req.ID, rerr)
		return
	}
	writeRPCResult(w, req.ID, result)
}

func (m *Manager) answerStreamed(w http.ResponseWriter, r *http.Request, req *request, seen mirrored, token any) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		result, rerr := m.dispatch(r.Context(), req, seen)
		if rerr != nil {
			writeRPCError(w, req.ID, rerr)
			return
		}
		writeRPCResult(w, req.ID, result)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	event := func(frame map[string]any) bool {
		raw, err := json.Marshal(frame)
		if err != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", raw); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	reports := make(chan map[string]any, 8)
	ctx := WithProgress(r.Context(), func(p Progress) {
		select {
		case reports <- progressFrame(token, p):
		default:
		}
	})
	type outcome struct {
		result any
		err    *rpcError
	}
	finished := make(chan outcome, 1)
	go func() {
		result, rerr := m.dispatch(ctx, req, seen)
		finished <- outcome{result, rerr}
	}()

	for {
		select {
		case <-r.Context().Done():
			return
		case frame := <-reports:
			if !event(frame) {
				return
			}
		case done := <-finished:
			for {
				select {
				case frame := <-reports:
					if !event(frame) {
						return
					}
					continue
				default:
				}
				break
			}
			answer := map[string]any{"jsonrpc": jsonrpcVersion, "id": req.ID}
			if done.err != nil {
				refusal := map[string]any{"code": done.err.Code, "message": done.err.Message}
				if len(done.err.Data) > 0 {
					refusal["data"] = done.err.Data
				}
				answer["error"] = refusal
			} else {
				answer["result"] = done.result
			}
			event(answer)
			return
		}
	}
}

const heartbeat = 25 * time.Second

func (m *Manager) streamOver(w http.ResponseWriter, r *http.Request, id any, want filter) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeRPCError(w, id, &rpcError{
			Code: -32603, Message: "this server cannot stream", status: http.StatusInternalServerError,
		})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	event := func(frame map[string]any) bool {
		raw, err := json.Marshal(frame)
		if err != nil {
			return false
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", raw); err != nil {
			return false
		}
		flusher.Flush()
		return true
	}

	name := fmt.Sprint(id)
	s := m.subscribe(name, want)
	defer m.unsubscribe(s)
	if !event(acknowledged(name, want)) {
		return
	}
	ticker := time.NewTicker(heartbeat)
	defer ticker.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-m.stop:
			event(map[string]any{"jsonrpc": jsonrpcVersion, "id": id, "result": closed(name)})
			return
		case frame := <-s.frames:
			if !event(frame) {
				return
			}
		case <-ticker.C:
			if _, err := io.WriteString(w, ":\n\n"); err != nil {
				return
			}
			flusher.Flush()
		}
	}
}

func writeRPCResult(w http.ResponseWriter, id any, result any) {
	raw, err := json.Marshal(result)
	if err != nil {
		writeRPCError(w, id, &rpcError{Code: -32603, Message: "the result could not be encoded"})
		return
	}
	writeRPC(w, http.StatusOK, map[string]any{
		"jsonrpc": jsonrpcVersion, "id": id, "result": json.RawMessage(raw),
	})
}

func writeRPCError(w http.ResponseWriter, id any, err *rpcError) {
	body := map[string]any{"code": err.Code, "message": err.Message}
	if len(err.Data) > 0 {
		body["data"] = err.Data
	}
	status := err.status
	if status == 0 {
		status = http.StatusOK
	}
	writeRPC(w, status, map[string]any{"jsonrpc": jsonrpcVersion, "id": id, "error": body})
}

func writeRPC(w http.ResponseWriter, status int, frame map[string]any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(frame)
}
