// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"testing"
	"time"
)

func TestAWaiterIsNotHeldByAnotherCallersRefresh(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{})
	e := &expiring{
		now: time.Now,
		fetch: func(ctx context.Context) (Creds, time.Time, error) {
			close(started)
			<-release
			return Creds{AccessKeyID: "AK", SecretAccessKey: "SK"}, time.Now().Add(time.Hour), nil
		},
	}

	go func() { _, _ = e.Get(context.Background()) }()
	<-started

	// A second caller with its own short deadline must not be parked on the
	// mutex until the first exchange finishes.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	if _, err := e.Get(ctx); err == nil {
		t.Fatal("this caller's deadline passed, so it must give up rather than wait on the refresh")
	}
	if took := time.Since(start); took > 2*time.Second {
		t.Fatalf("waited %v: a waiter blocked on a mutex cannot honour its own deadline", took)
	}
	close(release)
}
