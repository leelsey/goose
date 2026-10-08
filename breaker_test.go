// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/usage"
)

func TestABreakerSkipsALinkThatKeepsFailing(t *testing.T) {
	clock := time.Now()
	dead := &stub{id: "dead", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "dead", Message: "down", Sent: true, Retryable: true,
	}}
	alive := &stub{id: "alive"}
	mem := usage.NewMemory()
	c, err := New(
		WithProvider(dead), WithProvider(alive),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "dead"}, {Provider: "alive"},
		}}),
		WithBreaker(Breaker{Failures: 3, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
		WithUsage(mem),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	for range 3 {
		if _, err := c.Chat(context.Background(), req()); err != nil {
			t.Fatal(err)
		}
	}
	if dead.calls != 3 {
		t.Fatalf("dead link tried %d times before the breaker opened", dead.calls)
	}

	for range 5 {
		resp, err := c.Chat(context.Background(), req())
		if err != nil {
			t.Fatal(err)
		}
		if resp.Provider != "alive" {
			t.Fatalf("served by %q", resp.Provider)
		}
	}
	if dead.calls != 3 {
		t.Fatalf("dead link tried %d times: an open breaker skips it", dead.calls)
	}

	for _, r := range mem.Records() {
		if r.Provider == "dead" && r.Index >= 0 && r.Sent && r.Outcome == llm.OutcomeOK {
			t.Fatalf("a skipped link was booked as served: %+v", r)
		}
	}

	clock = clock.Add(2 * time.Minute)
	if _, err := c.Chat(context.Background(), req()); err != nil {
		t.Fatal(err)
	}
	if dead.calls != 4 {
		t.Fatalf("dead link tried %d times: the cooldown lets one through", dead.calls)
	}
}

func TestABreakerClosesOnceALinkAnswers(t *testing.T) {
	clock := time.Now()
	flaky := &flakyStub{stub: stub{id: "flaky"}, failFor: 3}
	c, err := New(
		WithProvider(flaky),
		WithBreaker(Breaker{Failures: 3, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}}
	for range 3 {
		_, _ = c.Chat(context.Background(), req)
	}
	if _, err := c.Chat(context.Background(), req); err == nil {
		t.Fatal("an open breaker skips the only link there is")
	}
	clock = clock.Add(2 * time.Minute)
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatalf("the link recovered and was still refused: %v", err)
	}
	if _, err := c.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
}

type flakyStub struct {
	stub
	failFor int
}

func (f *flakyStub) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	f.calls++
	if f.calls <= f.failFor {
		return nil, &llm.Error{Kind: llm.KindOverloaded, Provider: f.id, Sent: true, Retryable: true, Message: "down"}
	}
	return &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}},
		Stop:    llm.StopEnd,
	}}}, nil
}

func TestABreakerRefusesANegativeCount(t *testing.T) {
	if _, err := New(WithProvider(&stub{id: "p"}), WithBreaker(Breaker{Failures: -1})); err == nil {
		t.Fatal("accepted a negative failure count")
	}
}

func TestAProbeRefusedBeforeItIsSentGivesItsSlotBack(t *testing.T) {
	clock := time.Now()
	b := newBreakers(Breaker{Failures: 2, Cooldown: time.Minute}, func() time.Time { return clock })

	b.failed("p", 0)
	b.failed("p", 0)
	if _, ok := b.allow("p"); ok {
		t.Fatal("an open breaker let a request through inside the cooldown")
	}

	clock = clock.Add(2 * time.Minute)
	pr, ok := b.allow("p")
	if !ok {
		t.Fatal("the cooldown passed and no probe was let through")
	}
	b.abandoned("p", pr)

	if _, ok := b.allow("p"); !ok {
		t.Fatal("the slot was never given back, so the link is skipped for good")
	}
}

func TestASkippedLinkIsReportedToTheCaller(t *testing.T) {
	clock := time.Now()
	dead := &stub{id: "dead", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "dead", Message: "down", Sent: true, Retryable: true,
	}}
	alive := &stub{id: "alive", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "ok"}}},
		{Stop: llm.StopEnd, TurnEnd: true},
	}}
	c, err := New(
		WithProvider(dead), WithProvider(alive),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "dead"}, {Provider: "alive"},
		}}),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	if _, err := c.Chat(context.Background(), req()); err != nil {
		t.Fatal(err)
	}

	resp, err := c.Chat(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if codes := resp.Diagnostics.Codes(); !slices.Contains(codes, llm.PolicyBreakerOpen) {
		t.Fatalf("the response says nothing about the skipped link: %v", codes)
	}

	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	for range s.Frames() {
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if codes := s.Diagnostics().Codes(); !slices.Contains(codes, llm.PolicyBreakerOpen) {
		t.Fatalf("the stream says nothing about the skipped link: %v", codes)
	}
}

func TestAWhollySkippedChainSaysWhyOnTheError(t *testing.T) {
	clock := time.Now()
	dead := &stub{id: "dead", err: &llm.Error{
		Kind: llm.KindOverloaded, Provider: "dead", Message: "down", Sent: true, Retryable: true,
	}}
	c, err := New(
		WithProvider(dead),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { return clock }),
	)
	if err != nil {
		t.Fatal(err)
	}
	req := func() *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}}
	}
	if _, err := c.Chat(context.Background(), req()); err == nil {
		t.Fatal("the dead link answered")
	}
	_, err = c.Chat(context.Background(), req())
	e := llm.AsError(err)
	if e == nil || !slices.Contains(llm.Diagnostics(e.Diagnostics).Codes(), llm.PolicyBreakerOpen) {
		t.Fatalf("the refusal says nothing about the skipped link: %v", err)
	}
	if _, err := c.Stream(context.Background(), req()); !slices.Contains(
		llm.Diagnostics(llm.AsError(err).Diagnostics).Codes(), llm.PolicyBreakerOpen) {
		t.Fatalf("the streamed refusal says nothing about the skipped link: %v", err)
	}
}

// scripted fails each call with the next of its failures, then answers.
type scripted struct {
	stub
	errs []error
	caps llm.Capabilities
}

func (s *scripted) Capabilities() llm.Capabilities { return s.caps }

func (s *scripted) Complete(_ context.Context, a *llm.Attempt) (*llm.Response, error) {
	s.calls++
	if s.calls <= len(s.errs) {
		return nil, s.errs[s.calls-1]
	}
	return &llm.Response{Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}},
		Stop:    llm.StopEnd,
	}}}, nil
}

// A failure the request or the caller brought with it would be the same on a
// healthy backend, and the breaker is shared by every caller: counting it would
// let one caller take a backend out of the chain for everyone else.
func TestAFailureTheCallerBroughtIsNotTheBackends(t *testing.T) {
	for _, tc := range []struct {
		what string
		err  *llm.Error
		key  bool
	}{
		{"a request the backend could not read", &llm.Error{
			Kind: llm.KindInvalidRequest, Status: 400, Sent: true, Message: "prompt is too long"}, false},
		{"a request refused before it was sent", &llm.Error{
			Kind: llm.KindInvalidRequest, Code: llm.PolicyBodyTooLarge, Message: "too large", Retryable: true}, false},
		{"a capability this backend lacks", &llm.Error{
			Kind: llm.KindUnsupported, Message: "this deployment reads no images"}, false},
		{"a loss the caller would not take", &llm.Error{
			Kind: llm.KindLoss, Message: "translation would lose content"}, false},
		{"the caller's own key refused", &llm.Error{
			Kind: llm.KindAuth, Status: 401, Sent: true, Message: "invalid key"}, true},
		{"the caller's own key out of allowance", &llm.Error{
			Kind: llm.KindRateLimit, Status: 429, Sent: true, Message: "slow down"}, true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			clock := time.Now()
			p := &scripted{stub: stub{id: "one"}, errs: []error{tc.err, tc.err}, caps: llm.Capabilities{ClientKey: tc.key}}
			c, err := New(
				WithProvider(p),
				WithBreaker(Breaker{Failures: 2, Cooldown: time.Minute}),
				WithClock(func() time.Time { return clock }),
			)
			if err != nil {
				t.Fatal(err)
			}
			var opts []CallOption
			if tc.key {
				opts = append(opts, WithCredential(llm.Credential{Key: "sk-callers-own"}))
			}
			for range 2 {
				if _, err := c.Chat(context.Background(), req(), opts...); err == nil {
					t.Fatal("the backend refused")
				}
			}
			if _, err := c.Chat(context.Background(), req()); err != nil {
				t.Fatalf("err = %v: a healthy backend was taken out of the chain by what one caller sent", err)
			}
		})
	}
}

// The same refusals answered for the operator's own key are the backend's, even
// beside a caller's key the backend does not send in its place, and a backend
// that cannot be reached is down whether or not anything left.
func TestABackendsOwnFailuresStillCount(t *testing.T) {
	refused := &llm.Error{Kind: llm.KindAuth, Status: 401, Sent: true, Message: "invalid key"}
	for _, tc := range []struct {
		what string
		err  *llm.Error
		key  bool
	}{
		{"the operator's key refused", refused, false},
		{"the operator's key refused beside the caller's", refused, true},
		{"the operator's key out of allowance beside the caller's", &llm.Error{
			Kind: llm.KindRateLimit, Status: 429, Sent: true, Message: "slow down"}, true},
		{"a connection refused", &llm.Error{Kind: llm.KindTransport, Message: "connection refused", Retryable: true}, false},
	} {
		t.Run(tc.what, func(t *testing.T) {
			clock := time.Now()
			p := &scripted{stub: stub{id: "one"}, errs: []error{tc.err, tc.err}}
			c, err := New(
				WithProvider(p),
				WithBreaker(Breaker{Failures: 2, Cooldown: time.Minute}),
				WithClock(func() time.Time { return clock }),
			)
			if err != nil {
				t.Fatal(err)
			}
			var opts []CallOption
			if tc.key {
				opts = append(opts, WithCredential(llm.Credential{Key: "sk-placeholder"}))
			}
			for range 2 {
				_, _ = c.Chat(context.Background(), req(), opts...)
			}
			if _, err := c.Chat(context.Background(), req()); err == nil || p.calls != 2 {
				t.Fatalf("err = %v, calls = %d: a backend failing on its own account is skipped", err, p.calls)
			}
		})
	}
}

// An open proxy hands on whatever key a client presented, and a backend with a
// key of its own sends that instead, so a refusal there is the backend's
// whatever the caller brought. Only a backend that sends the caller's key in
// place of the operator's is answering for the caller.
func TestARefusedKeyIsTheCallersOnlyWhenTheCallersWasSent(t *testing.T) {
	for _, tc := range []struct {
		what    string
		opts    openai.Options
		counted bool
	}{
		{"the operator's key sent", openai.Options{APIKey: "operator-key"}, true},
		{"the operator's key sent where the caller's may be", openai.Options{APIKey: "operator-key", AllowClientKey: true}, true},
		{"the caller's key sent", openai.Options{AllowClientKey: true}, false},
	} {
		t.Run(tc.what, func(t *testing.T) {
			var hits atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				hits.Add(1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusUnauthorized)
				_, _ = io.WriteString(w, `{"error":{"message":"invalid key"}}`)
			}))
			defer srv.Close()
			o := tc.opts
			o.ID, o.BaseURL = "up", srv.URL
			p, err := openai.New(o)
			if err != nil {
				t.Fatal(err)
			}
			c, err := New(WithProvider(p), WithBreaker(Breaker{Failures: 2, Cooldown: time.Minute}))
			if err != nil {
				t.Fatal(err)
			}
			for range 3 {
				if _, err := c.Chat(context.Background(), req(), WithCredential(llm.Credential{Key: "sk-placeholder"})); err == nil {
					t.Fatal("the backend refused the key")
				}
			}
			want := int32(3)
			if tc.counted {
				want = 2
			}
			if got := hits.Load(); got != want {
				t.Fatalf("backend reached %d times, want %d", got, want)
			}
		})
	}
}

// A model the caller named and the backend does not have is the caller's
// mistake, which the next caller for a model it does have should not pay for.
// One the operator named is the backend's to serve, and its absence counts.
func TestAModelTheBackendLacksCountsOnlyWhenTheOperatorNamedIt(t *testing.T) {
	missing := &llm.Error{Kind: llm.KindNotFound, Status: 404, Sent: true, Message: "model: claude-typo"}
	for _, tc := range []struct {
		what    string
		chain   []route.Target
		counted bool
	}{
		{"the caller's model", []route.Target{{Provider: "one"}}, false},
		{"the operator's model", []route.Target{{Provider: "one", Model: "claude-pinned"}}, true},
	} {
		t.Run(tc.what, func(t *testing.T) {
			p := &scripted{stub: stub{id: "one"}, errs: []error{missing, missing}}
			c, err := New(
				WithProvider(p),
				WithRoutes(route.Rule{Match: route.Always(), Chain: tc.chain}),
				WithBreaker(Breaker{Failures: 2, Cooldown: time.Minute}),
			)
			if err != nil {
				t.Fatal(err)
			}
			for range 2 {
				if _, err := c.Chat(context.Background(), req()); err == nil {
					t.Fatal("the backend has no such model")
				}
			}
			_, err = c.Chat(context.Background(), req())
			if tc.counted && (err == nil || p.calls != 2) {
				t.Fatalf("err = %v, calls = %d: a model the operator named and the backend lacks is the backend's failing", err, p.calls)
			}
			if !tc.counted && err != nil {
				t.Fatalf("err = %v: one caller's unknown model took the backend out for everyone", err)
			}
		})
	}
}

// A request too large for one backend is the caller's to send, and would be
// refused as surely on a healthy one; the next link has a ceiling of its own.
func TestARequestTooLargeForOneBackendIsNotItsFailure(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		_, _ = io.WriteString(w, `{"type":"error","error":{"type":"request_too_large","message":"Request exceeds the maximum allowed number of bytes."}}`)
	}))
	defer srv.Close()
	claude, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	second := &stub{id: "second"}
	c, err := New(
		WithProvider(claude), WithProvider(second),
		WithRoutes(route.Rule{Match: route.Always(), Chain: []route.Target{{Provider: "claude"}, {Provider: "second"}}}),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
	)
	if err != nil {
		t.Fatal(err)
	}
	for round := range 2 {
		resp, err := c.Chat(context.Background(), req())
		if err != nil || resp.Provider != "second" {
			t.Fatalf("round %d: err = %v: the next link, with a ceiling of its own, was not tried", round, err)
		}
	}
	if n := hits.Load(); n != 2 {
		t.Fatalf("claude was tried %d times: one caller's oversized request took it out for everyone", n)
	}
}

// A backend with no key of its own sends only what the caller brought. When
// that is nothing — none presented, or one withheld from a link it was not
// minted for — the refusal is the caller's, and a caller with a key of its own
// is still served.
func TestAKeylessCallerOnABackendWithNoKeyIsNotItsFailure(t *testing.T) {
	refused := &llm.Error{Kind: llm.KindAuth, Status: 401, Sent: true, Message: "x-api-key header is required"}
	p := &scripted{stub: stub{id: "one"}, errs: []error{refused, refused}, caps: llm.Capabilities{ClientKey: true}}
	c, err := New(WithProvider(p), WithBreaker(Breaker{Failures: 2, Cooldown: time.Minute}))
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := c.Chat(context.Background(), req()); err == nil {
			t.Fatal("the backend refused a call with no key")
		}
	}
	if _, err := c.Chat(context.Background(), req(), WithCredential(llm.Credential{Key: "sk-callers-own"})); err != nil {
		t.Fatalf("err = %v: callers without a key took the backend out for one with a key", err)
	}
}

// held fails a call asking it to at once, and holds any other until its caller
// leaves or it is let go.
type held struct {
	id      string
	mu      sync.Mutex
	calls   int
	let     chan struct{}
	arrived chan struct{}
}

func (h *held) ID() string                     { return h.id }
func (h *held) Capabilities() llm.Capabilities { return llm.Capabilities{Streaming: true} }

func (h *held) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	h.mu.Lock()
	h.calls++
	h.mu.Unlock()
	if llm.TextOf(a.Request.Messages[0].Content) == "fail" {
		return nil, &llm.Error{Kind: llm.KindOverloaded, Provider: h.id, Sent: true, Retryable: true, Message: "down"}
	}
	h.arrived <- struct{}{}
	select {
	case <-ctx.Done():
		return nil, &llm.Error{Kind: llm.KindCancelled, Provider: h.id, Sent: true, Message: ctx.Err().Error()}
	case <-h.let:
		return &llm.Response{Choices: []llm.Choice{{Stop: llm.StopEnd}}}, nil
	}
}

func (h *held) Stream(context.Context, *llm.Attempt) (*llm.Stream, error) {
	return nil, llm.Errorf(llm.KindInternal, "not this way")
}

func (h *held) count() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.calls
}

// A recovering backend is sent one probe. A call let through before the breaker
// opened, finishing while that probe runs, does not hand its slot to a second.
func TestOnlyTheProbeGivesItsSlotBack(t *testing.T) {
	var mu sync.Mutex
	now := time.Now()
	p := &held{id: "one", let: make(chan struct{}), arrived: make(chan struct{}, 4)}
	c, err := New(
		WithProvider(p),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { mu.Lock(); defer mu.Unlock(); return now }),
	)
	if err != nil {
		t.Fatal(err)
	}
	say := func(text string) *llm.Request {
		return &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: text}}},
		}}
	}

	early, leave := context.WithCancel(context.Background())
	defer leave()
	earlyDone := make(chan struct{})
	go func() { defer close(earlyDone); _, _ = c.Chat(early, say("hold")) }()
	<-p.arrived

	if _, err := c.Chat(context.Background(), say("fail")); err == nil {
		t.Fatal("the backend failed")
	}
	mu.Lock()
	now = now.Add(2 * time.Minute)
	mu.Unlock()

	probeDone := make(chan struct{})
	go func() { defer close(probeDone); _, _ = c.Chat(context.Background(), say("hold")) }()
	<-p.arrived

	leave()
	<-earlyDone

	before := p.count()
	if _, err := c.Chat(context.Background(), say("fail")); err == nil {
		t.Fatal("the probe is still out, so this call is not served")
	}
	if got := p.count(); got != before {
		t.Fatalf("calls = %d, want %d: a second probe was let through while the first was running", got, before)
	}
	close(p.let)
	<-probeDone
}

// A backend can fail after its first frame of content as surely as before it,
// and a stream that did has failed. The breaker hears of it when the stream
// ends, not when the stream was handed over as though it had already worked.
func TestAStreamThatFailsPartWayCountsAgainstItsBackend(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "event: message_start\n"+
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"usage":{"input_tokens":12,"output_tokens":1}}}`+"\n\n"+
			"event: content_block_start\n"+`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`+"\n\n"+
			"event: content_block_delta\n"+`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"half"}}`+"\n\n"+
			"event: error\n"+`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`+"\n\n")
	}))
	defer srv.Close()
	claude, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: srv.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	second := &stub{id: "second", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "whole"}}},
		{Stop: llm.StopEnd},
	}}
	c, err := New(
		WithProvider(claude), WithProvider(second),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "claude"}, {Provider: "second"},
		}}),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
	)
	if err != nil {
		t.Fatal(err)
	}
	read := func() (string, *llm.Stream) {
		s, err := c.Stream(context.Background(), req())
		if err != nil {
			t.Fatal(err)
		}
		var text string
		for f := range s.Frames() {
			text += llm.TextOf(f.Delta)
		}
		_ = s.Close()
		return text, s
	}
	if text, s := read(); text != "half" || s.Err() == nil {
		t.Fatalf("text = %q, err = %v: the backend fell over part way", text, s.Err())
	}
	if text, s := read(); text != "whole" || s.Meta().Provider != "second" || hits.Load() != 1 {
		t.Fatalf("text = %q from %q, claude reached %d times: its failure part way was never counted",
			text, s.Meta().Provider, hits.Load())
	}
}

// A recovering backend's one probe may be a stream, and it holds the slot until
// it ends: no second call is let through beside it while it runs, and once it
// ends cleanly the backend is back. One abandoned without being closed gives
// the slot back once its caller has gone, or the backend is never tried again.
func TestAStreamedProbeHoldsItsSlotUntilItEnds(t *testing.T) {
	var mu sync.Mutex
	now := time.Now()
	later := func() {
		mu.Lock()
		now = now.Add(2 * time.Minute)
		mu.Unlock()
	}
	p := &stub{id: "one", frames: []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "ok"}}},
		{Stop: llm.StopEnd},
	}}
	c, err := New(
		WithProvider(p),
		WithBreaker(Breaker{Failures: 1, Cooldown: time.Minute}),
		WithClock(func() time.Time { mu.Lock(); defer mu.Unlock(); return now }),
	)
	if err != nil {
		t.Fatal(err)
	}
	c.abandon = 20 * time.Millisecond

	c.breaker.failed("one", 0)
	later()
	s, err := c.Stream(context.Background(), req())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.breaker.allow("one"); ok {
		t.Fatal("a second call was let through beside the probe while it ran")
	}
	for range s.Frames() {
	}
	_ = s.Close()
	if _, ok := c.breaker.allow("one"); !ok {
		t.Fatal("the probe ended cleanly and the backend is still out")
	}

	c.breaker.failed("one", 0)
	later()
	ctx, cancel := context.WithCancel(context.Background())
	if _, err := c.Stream(ctx, req()); err != nil {
		t.Fatal(err)
	}
	cancel()
	for deadline := time.Now().Add(2 * time.Second); ; time.Sleep(5 * time.Millisecond) {
		if _, ok := c.breaker.allow("one"); ok {
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("a probe abandoned without being closed kept its slot, so this backend is out for good")
		}
	}
}
