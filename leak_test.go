// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"runtime"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

func TestNoGoroutinesAreLeftBehind(t *testing.T) {
	settle(t)
	before := runtime.NumGoroutine()

	for range 200 {
		p := &stub{id: "one", frames: []*llm.Frame{
			{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
			{Stop: llm.StopEnd},
			{Usage: &llm.Usage{Prompt: 1, Completion: 1, Reported: true}, TurnEnd: true},
		}}
		c, err := New(WithProvider(p))
		if err != nil {
			t.Fatal(err)
		}
		c.abandon = 20 * time.Millisecond
		if _, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err != nil {
			t.Fatal(err)
		}

		s, err := c.Stream(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}})
		if err != nil {
			t.Fatal(err)
		}
		for range s.Frames() {
		}
		_ = s.Close()

		ctx, cancel := context.WithCancel(context.Background())
		s2, err := c.Stream(ctx, &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}})
		if err != nil {
			t.Fatal(err)
		}
		cancel()
		_ = s2.Close()

		ctx3, cancel3 := context.WithCancel(context.Background())
		if _, err := c.Stream(ctx3, &llm.Request{Model: "m", Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}); err != nil {
			t.Fatal(err)
		}
		cancel3()
	}

	settle(t)
	after := runtime.NumGoroutine()
	if after > before+8 {
		buf := make([]byte, 1<<20)
		n := runtime.Stack(buf, true)
		t.Fatalf("goroutines %d -> %d after 800 calls\n%s", before, after, buf[:n])
	}
}

func settle(t *testing.T) {
	t.Helper()
	for range 60 {
		runtime.GC()
		time.Sleep(20 * time.Millisecond)
	}
}
