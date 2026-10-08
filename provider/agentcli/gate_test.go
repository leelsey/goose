// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

// A call waiting for a slot holds whatever admitted it, a proxy's own slot
// among them, so an unbounded queue here would turn away calls to every other
// backend. Past the bound a call is shed for the next link or a 503.
func TestAFullQueueSheds(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", func(o *Options) { o.MaxConcurrent = 1 })
	release, err := p.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	waited := make(chan error, 1)
	go func() {
		next, err := p.Acquire(context.Background())
		if err == nil {
			next()
		}
		waited <- err
	}()
	for start := time.Now(); len(p.queue) == 0; time.Sleep(time.Millisecond) {
		if time.Since(start) > 10*time.Second {
			t.Fatal("the second call never queued")
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = p.Acquire(ctx)
	if e := llm.AsError(err); e == nil || e.Kind != llm.KindSaturated || e.Sent {
		t.Fatalf("err = %v: the queue holds as many as the gate by default, and the third call is one too many", err)
	}

	release()
	if err := <-waited; err != nil {
		t.Fatalf("the call within the bound was not given the slot that came free: %v", err)
	}
}

func TestANegativeQueueShedsAtOnce(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", func(o *Options) { o.MaxConcurrent, o.MaxQueued = 1, -1 })
	release, err := p.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err = p.Acquire(ctx)
	if e := llm.AsError(err); e == nil || e.Kind != llm.KindSaturated {
		t.Fatalf("err = %v: an operator who asked for no queue asked for the call to go elsewhere", err)
	}
}

// A context that has already ended by the time Acquire is called is not owed
// a slot even when the gate is empty: the call it belongs to is not going to
// be sent, so taking the slot only withholds it from a call that could use it.
func TestADoneContextIsRefusedEvenWithAFreeSlot(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", func(o *Options) { o.MaxConcurrent = 1 })
	ctx, cancel := context.WithTimeout(context.Background(), 0)
	cancel()
	<-ctx.Done()
	release, err := p.Acquire(ctx)
	release()
	e := llm.AsError(err)
	if e == nil || e.Kind != llm.KindTimeout {
		t.Fatalf("err = %v: a call whose deadline had already passed was given a slot", err)
	}

	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	release, err = p.Acquire(ctx)
	release()
	e = llm.AsError(err)
	if e == nil || e.Kind != llm.KindCancelled {
		t.Fatalf("err = %v: a call whose caller had already gone was given a slot", err)
	}
}
