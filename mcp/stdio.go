// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"sync"

	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/internal/wirecore"
)

const defaultStdioFrame = 16 << 20

var errNotReading = errors.New("the server stopped reading its input")

type stdioTransport struct {
	turn turn
	proc *spawn.Process

	frames chan []byte
	// done says the transport can carry nothing more, and closed that the
	// child behind it has been shut down.
	done     chan struct{}
	once     sync.Once
	closed   chan struct{}
	closeErr error

	errMu sync.Mutex
	err   error
}

func startStdio(ctx context.Context, cfg ServerConfig, maxFrame int) (*stdioTransport, error) {
	if maxFrame <= 0 {
		maxFrame = defaultStdioFrame
	}
	proc, err := spawn.StartPiped(context.WithoutCancel(ctx), spawn.Options{
		Binary: cfg.Command, Args: cfg.Args, Dir: cfg.Dir, Env: cfg.Env,
	})
	if err != nil {
		return nil, err
	}
	t := &stdioTransport{
		turn:   make(turn, 1),
		proc:   proc,
		frames: make(chan []byte),
		done:   make(chan struct{}),
		closed: make(chan struct{}),
	}
	go t.read(wirecore.NewLines(proc.Stdout(), maxFrame))
	return t, nil
}

func (t *stdioTransport) read(lines *wirecore.Lines) {
	defer close(t.frames)
	// Once nothing reads the child's output it is shut down rather than left
	// to fill a pipe nobody empties, and the next call dials a fresh one. What
	// stopped the reading is kept for the call that was waiting on it.
	defer t.end()
	for lines.Scan() {
		frame := append([]byte(nil), lines.Bytes()...)
		select {
		case t.frames <- frame:
		case <-t.done:
			return
		}
	}
	t.setErr(lines.Err())
}

func (t *stdioTransport) setErr(err error) {
	t.errMu.Lock()
	defer t.errMu.Unlock()
	if t.err == nil {
		t.err = err
	}
}

func (t *stdioTransport) readErr() error {
	t.errMu.Lock()
	defer t.errMu.Unlock()
	return t.err
}

func (t *stdioTransport) Call(ctx context.Context, out Outbound) (*response, error) {
	if err := t.turn.take(ctx); err != nil {
		return nil, err
	}
	defer t.turn.give()
	if err := t.write(ctx, out.Payload); err != nil {
		return nil, err
	}
	id := out.ID
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case frame, open := <-t.frames:
			if !open {
				if err := t.readErr(); err != nil {
					return nil, err
				}
				return nil, errors.New("server closed its output")
			}
			var resp response
			if err := json.Unmarshal(frame, &resp); err != nil {
				// Servers write to their own stdout, and a log line or a
				// batched answer is not this call failing. Keep reading for
				// the answer that was asked for.
				continue
			}
			if resp.isAnswer(id) {
				return &resp, nil
			}
			if resp.Method != "" && out.Observe != nil {
				out.Observe(resp.Method, resp.Params)
			}
		}
	}
}

func (t *stdioTransport) Notify(ctx context.Context, out Outbound) error {
	if err := t.turn.take(ctx); err != nil {
		return err
	}
	defer t.turn.give()
	return t.write(ctx, out.Payload)
}

func (t *stdioTransport) write(ctx context.Context, payload []byte) error {
	select {
	case <-t.done:
		return gone(t.readErr())
	default:
	}
	wrote := make(chan error, 1)
	go func() {
		_, err := t.proc.Write(append(payload, '\n'))
		wrote <- err
	}()
	select {
	case err := <-wrote:
		return t.wrote(err)
	case <-ctx.Done():
	}
	select {
	case err := <-wrote:
		return t.wrote(err)
	default:
	}
	// A child that has stopped reading holds the write for as long as it
	// likes, and part of the frame may already be in the pipe, where whatever
	// is written next would be read as the rest of it. It is treated as gone:
	// shutting it down releases the write, and the next call dials afresh.
	t.setErr(errNotReading)
	t.end()
	return ctx.Err()
}

func (t *stdioTransport) wrote(err error) error {
	if err == nil {
		return nil
	}
	// A child that has closed its input has exited or is about to, and what
	// was written to it was never read.
	t.setErr(err)
	t.end()
	return gone(err)
}

func (t *stdioTransport) end() {
	t.once.Do(func() {
		close(t.done)
		go func() {
			t.closeErr = t.proc.Close()
			close(t.closed)
		}()
	})
}

func (t *stdioTransport) Close() error {
	t.end()
	<-t.closed
	return t.closeErr
}
