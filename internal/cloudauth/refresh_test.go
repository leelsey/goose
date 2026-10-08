// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
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

func TestTheCallerThatStartsARefreshKeepsItsOwnDeadline(t *testing.T) {
	release := make(chan struct{})
	var once sync.Once
	free := func() { once.Do(func() { close(release) }) }
	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked.Add(1)
		<-release
		fmt.Fprintf(w, `{"access_token":"tok","expires_in":3600,"AccessKeyId":"AK","SecretAccessKey":"SK","Expiration":%q}`,
			time.Now().Add(time.Hour).Format(time.RFC3339))
	}))
	defer srv.Close()
	defer free()
	// Before the fix the first call returned only once the exchange did; this
	// lets it, so the test fails on the time taken rather than hanging.
	defer time.AfterFunc(3*time.Second, free).Stop()

	key, err := ParseServiceAccountKey(keyJSON(t, srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	account, err := ServiceAccountSource(key, srv.Client(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	container := ContainerCredentials(srv.Client(), "", srv.URL, "", "", nil)
	metadata := MetadataSource(srv.Client(), srv.URL, nil)
	sources := map[string]func(context.Context) error{
		"aws container":          func(ctx context.Context) error { _, err := container(ctx); return err },
		"google metadata":        func(ctx context.Context) error { _, err := metadata.Token(ctx); return err },
		"google service account": func(ctx context.Context) error { _, err := account.Token(ctx); return err },
	}

	for name, get := range sources {
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		start := time.Now()
		err := get(ctx)
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("%s: err = %v: this caller's deadline passed while the exchange it started went on", name, err)
		}
		if took := time.Since(start); took > time.Second {
			t.Fatalf("%s: returned after %v: the caller that starts a refresh is held for the whole exchange", name, took)
		}
	}

	free()
	for name, get := range sources {
		if err := get(context.Background()); err != nil {
			t.Fatalf("%s: %v: the exchange a departed caller started still finishes for the rest", name, err)
		}
	}
	if n := asked.Load(); n != int64(len(sources)) {
		t.Fatalf("exchanges = %d, want %d: the one already under way is waited on, not repeated", n, len(sources))
	}
}

func TestARefreshThatPanicsFailsOnlyItsCaller(t *testing.T) {
	var calls atomic.Int64
	e := &expiring{
		now: time.Now,
		fetch: func(context.Context) (Creds, time.Time, error) {
			if calls.Add(1) == 1 {
				panic("broken transport")
			}
			return Creds{AccessKeyID: "AK", SecretAccessKey: "SK"}, time.Now().Add(time.Hour), nil
		},
	}
	if _, err := e.Get(context.Background()); err == nil {
		t.Fatal("a refresh that panicked produced no error")
	}
	if creds, err := e.Get(context.Background()); err != nil || creds.AccessKeyID != "AK" {
		t.Fatalf("creds = %+v, err = %v: a panicked refresh left the next caller nothing to do", creds, err)
	}
}
