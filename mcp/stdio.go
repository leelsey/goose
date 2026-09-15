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

type stdioTransport struct {
	mu   sync.Mutex
	proc *spawn.Process

	frames chan []byte
	done   chan struct{}
	once   sync.Once

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
		proc:   proc,
		frames: make(chan []byte),
		done:   make(chan struct{}),
	}
	go t.read(wirecore.NewLines(proc.Stdout(), maxFrame))
	return t, nil
}

func (t *stdioTransport) read(lines *wirecore.Lines) {
	defer close(t.frames)
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
	t.mu.Lock()
	defer t.mu.Unlock()
	if err := t.write(out.Payload); err != nil {
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
				return nil, errors.New("unreadable frame: " + err.Error())
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
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.write(out.Payload)
}

func (t *stdioTransport) write(payload []byte) error {
	if _, err := t.proc.Write(append(payload, '\n')); err != nil {
		return err
	}
	return nil
}

func (t *stdioTransport) Close() error {
	t.once.Do(func() { close(t.done) })
	return t.proc.Close()
}
