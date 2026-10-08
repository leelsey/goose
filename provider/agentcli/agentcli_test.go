// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

func fake(t *testing.T, name string) string {
	t.Helper()
	path, err := filepath.Abs(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return path
}

func provider(t *testing.T, kind Kind, script string, mutate func(*Options)) *Provider {
	t.Helper()
	o := Options{Kind: kind, Binary: fake(t, script), Env: os.Environ(), Grace: time.Second}
	if mutate != nil {
		mutate(&o)
	}
	p, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close(context.Background()) })
	return p
}

func attempt(req *llm.Request) *llm.Attempt {
	return &llm.Attempt{Request: req, Model: "test-model", Recorder: llm.NopRecorder{}}
}

func hello() *llm.Request {
	return &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}
}

func TestClaudeTurnsAddAndFramesMerge(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	parts := resp.Choices[0].Message.Content
	th, ok := parts[0].(*llm.Thinking)
	if !ok || th.Text != "considering" || th.Signature != "sig-1" {
		t.Fatalf("thinking = %+v", parts[0])
	}
	if th.Signer != llm.FamilyAnthropic {
		t.Fatalf("thinking = %+v: Claude's signature is Anthropic's, and no backend of another family is sent it", th)
	}
	tu := llm.ToolUses(parts)
	if len(tu) != 1 || string(tu[0].Args) != `{"q":1}` {
		t.Fatalf("tool call = %+v", tu)
	}
	if !strings.Contains(llm.TextOf(parts), "the answer") {
		t.Fatalf("text = %q", llm.TextOf(parts))
	}

	u := resp.Usage
	if u.Prompt != 1012+200 {
		t.Fatalf("prompt = %d: turns add where frames merge", u.Prompt)
	}
	if u.Completion != 47 {
		t.Fatalf("completion = %d", u.Completion)
	}
	if u.CacheRead != 900 || u.CacheWrite != 100 {
		t.Fatalf("cache = %d/%d: the breakdown is announced once and must survive", u.CacheRead, u.CacheWrite)
	}
	if u.Cost == nil || u.Cost.Amount != 0.0123 {
		t.Fatalf("cost = %+v: only this binary reports a price and it must not be lost", u.Cost)
	}
	if u.Turns != 2 {
		t.Fatalf("turns = %d: the invocation ran two billed turns", u.Turns)
	}
}

func TestClaudeSessionReachesTheStream(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for range s.Frames() {
	}
	if s.Meta().Session != "sess-42" {
		t.Fatalf("session = %q: the next turn resumes with it", s.Meta().Session)
	}
}

func TestCodexUsage(t *testing.T) {
	p := provider(t, Codex, "fake-codex.sh", nil)
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Prompt != 90 || resp.Usage.Completion != 12 || resp.Usage.CacheRead != 50 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if resp.Usage.Reasoning != 4 {
		t.Fatalf("reasoning = %d", resp.Usage.Reasoning)
	}
	parts := resp.Choices[0].Message.Content
	if _, ok := parts[0].(*llm.Thinking); !ok {
		t.Fatalf("parts = %+v", parts)
	}
}

func TestFailedInvocationKeepsItsAccounting(t *testing.T) {
	p := provider(t, Claude, "fake-claude-fail.sh", nil)
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err == nil {
		t.Fatal("an invocation that reported an error must not read as success")
	}
	e := llm.AsError(err)
	if e.Status != 529 || !strings.Contains(e.Message, "overloaded") {
		t.Fatalf("error = %+v", e)
	}
	if resp == nil || resp.Usage.Prompt != 33 {
		t.Fatalf("usage = %+v: the turn ran and was billed before it was rejected", resp)
	}
}

// A context that has already ended by the time the binary is spawned is not
// the binary's fault, or the machine's: it is the timeout or cancellation the
// caller already has, and the caller reads it as one.
func TestADoneContextAtSpawnIsTimeoutOrCancelledNotTransport(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	ctx, cancel := context.WithTimeout(context.Background(), 0)
	cancel()
	<-ctx.Done()
	_, err := p.Stream(ctx, attempt(hello()))
	if e := llm.AsError(err); e == nil || e.Kind != llm.KindTimeout {
		t.Fatalf("err = %v: a deadline that had already passed before the spawn was reported as a transport failure", err)
	}

	ctx, cancel = context.WithCancel(context.Background())
	cancel()
	_, err = p.Stream(ctx, attempt(hello()))
	if e := llm.AsError(err); e == nil || e.Kind != llm.KindCancelled {
		t.Fatalf("err = %v: a caller that had already gone before the spawn was reported as a transport failure", err)
	}
}

// The other side of the same fix: a spawn that fails for a real reason, with
// a context nowhere near ending, must still read as the transport failure it
// is. Releasing the attempt's own timeout context as part of handling the
// spawn error must not make every such failure look like a cancellation.
func TestARealSpawnFailureWithALiveContextIsStillTransport(t *testing.T) {
	p := provider(t, Claude, "does-not-exist-anywhere", nil)
	_, err := p.Stream(context.Background(), attempt(hello()))
	e := llm.AsError(err)
	if e == nil || e.Kind != llm.KindTransport {
		t.Fatalf("err = %v: a binary that does not exist was reported as a cancellation", err)
	}
	if e.Sent {
		t.Fatalf("err = %v: a call that never spawned was booked as sent", err)
	}
}

func TestUnreadableLineIsNotSwallowed(t *testing.T) {
	p := provider(t, Claude, "fake-broken.sh", nil)
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err == nil {
		t.Fatal("a line that cannot be parsed is a line that was lost")
	}
	if resp == nil || !strings.Contains(llm.TextOf(resp.Choices[0].Message.Content), "partial") {
		t.Fatalf("what was already delivered must survive: %+v", resp)
	}
}

func TestSystemPromptTravelsOnItsOwnChannelForClaude(t *testing.T) {
	record := filepath.Join(t.TempDir(), "record")
	p := provider(t, Claude, "fake-echo-args.sh", func(o *Options) {
		o.Env = append(os.Environ(), "GOOSE_TEST_RECORD="+record)
		o.TempDir = t.TempDir()
	})
	req := hello()
	req.System = []llm.Part{&llm.Text{Text: "be brief"}}
	if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
		t.Fatal(err)
	}
	body, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "--system-prompt-file") {
		t.Fatalf("argv = %s: the system prompt belongs on the cached channel", body)
	}
	if strings.Contains(string(body), "System: be brief") {
		t.Fatal("the system prompt was also folded into the prompt, so it is billed twice")
	}
}

func TestSystemPromptFoldedElsewhereIsReported(t *testing.T) {
	record := filepath.Join(t.TempDir(), "record")
	p := provider(t, Codex, "fake-echo-args.sh", func(o *Options) {
		o.Env = append(os.Environ(), "GOOSE_TEST_RECORD="+record)
	})
	req := hello()
	req.System = []llm.Part{&llm.Text{Text: "be brief"}}
	var diags llm.Diagnostics
	a := attempt(req)
	a.Recorder = &diags
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(record)
	if !strings.Contains(string(body), "System: be brief") {
		t.Fatalf("argv = %s: the prompt must still reach the binary", body)
	}
	found := false
	for _, d := range diags {
		if d.Code == llm.DropSystemPrompt {
			found = true
		}
	}
	if !found {
		t.Fatalf("diagnostics = %+v: three of four binaries dropped this silently before", diags)
	}
}

func TestPromptNeverReachesArgv(t *testing.T) {
	for _, kind := range []Kind{Claude, Codex, Antigravity, Copilot} {
		for _, size := range []struct {
			name   string
			prompt string
		}{
			{"short", "SECRET-PROMPT-BODY"},
			{"long", strings.Repeat("x", 100<<10)},
		} {
			t.Run(string(kind)+"/"+size.name, func(t *testing.T) {
				record := filepath.Join(t.TempDir(), "record")
				p := provider(t, kind, "fake-echo-args.sh", func(o *Options) {
					o.Env = append(os.Environ(), "GOOSE_TEST_RECORD="+record)
					o.TempDir = t.TempDir()
				})
				req := hello()
				req.Messages[0].Content = []llm.Part{&llm.Text{Text: size.prompt}}
				if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
					t.Fatal(err)
				}
				body, err := os.ReadFile(record)
				if err != nil {
					t.Fatal(err)
				}
				args := strings.SplitN(string(body), "\n", 2)[0]
				if strings.Contains(args, size.prompt[:min(len(size.prompt), 64)]) {
					t.Fatalf("argv = %.200s: the prompt must never be an argument", args)
				}
				if !strings.Contains(string(body), "STDIN\tUser: ") {
					t.Fatalf("record = %.200s: the prompt must arrive on stdin instead", body)
				}
			})
		}
	}
}

func TestEnvironmentIsScrubbed(t *testing.T) {
	record := filepath.Join(t.TempDir(), "record")
	env := append(os.Environ(), "GOOSE_PROXY_KEY=secret-key", "GOOSE_TEST_RECORD="+record)
	p := provider(t, Claude, "fake-echo-args.sh", func(o *Options) {
		allow := append(DefaultEnvAllowlist(), "GOOSE_TEST_RECORD")
		o.Env = scrub(env, allow)
	})
	if _, err := p.Complete(context.Background(), attempt(hello())); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(record)
	if strings.Contains(string(body), "secret-key") {
		t.Fatal("a spawned agent received this proxy's own key")
	}
}

func TestConcurrencyGateQueuesRatherThanSheds(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", func(o *Options) { o.MaxConcurrent = 1 })
	release, err := p.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	got := make(chan error, 1)
	go func() {
		next, err := p.Acquire(context.Background())
		if err == nil {
			next()
		}
		got <- err
	}()
	select {
	case err := <-got:
		t.Fatalf("err = %v: a call past the limit waits for a slot, since for a subscription backend there is often no next link to shed it to", err)
	case <-time.After(100 * time.Millisecond):
	}
	release()
	select {
	case err := <-got:
		if err != nil {
			t.Fatalf("the queued call was not given the slot that came free: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the slot was not returned")
	}
	if _, err := p.Acquire(context.Background()); err != nil {
		t.Fatalf("the slot was not returned: %v", err)
	}
}

func TestAQueuedCallEndsWithItsContext(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", func(o *Options) { o.MaxConcurrent = 1 })
	release, err := p.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer release()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := p.Acquire(ctx); err == nil {
		t.Fatal("a call whose deadline passed in the queue was given a slot that was never free")
	} else if e := llm.AsError(err); e.Kind != llm.KindTimeout || e.Sent {
		t.Fatalf("err = %+v: the caller is still there, its deadline is what ran out", e)
	}

	gone, hangUp := context.WithCancel(context.Background())
	time.AfterFunc(50*time.Millisecond, hangUp)
	if _, err := p.Acquire(gone); err == nil {
		t.Fatal("a caller that has gone was given a slot that was never free")
	} else if e := llm.AsError(err); e.Kind != llm.KindCancelled || e.Sent {
		t.Fatalf("err = %+v", e)
	}
}

func TestCancellationEndsTheChild(t *testing.T) {
	p := provider(t, Claude, "fake-slow.sh", nil)
	ctx, cancel := context.WithCancel(context.Background())
	s, err := p.Stream(ctx, attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	cancel()
	done := make(chan struct{})
	go func() {
		_, _ = wirecore.Fold(s, wirecore.FoldOptions{})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("a cancelled invocation left its child running")
	}
}

func TestUnknownKindIsRefused(t *testing.T) {
	if _, err := New(Options{Kind: "nonsense"}); err == nil {
		t.Fatal("an unknown binary must be refused at construction")
	}
}

func TestUnknownSystemModeIsRefused(t *testing.T) {
	if _, err := New(Options{Kind: Claude, Binary: fake(t, "fake-claude.sh"), SystemMode: "fil"}); err == nil {
		t.Fatal("an unknown system mode must be refused at construction")
	}
	for _, mode := range []SystemMode{SystemFile, SystemArg, SystemAppend, SystemPrompt, SystemOff} {
		if _, err := New(Options{Kind: Claude, Binary: fake(t, "fake-claude.sh"), SystemMode: mode}); err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
	}
}

func scrub(environ, allow []string) []string {
	allowed := map[string]bool{}
	for _, name := range allow {
		allowed[name] = true
	}
	var out []string
	for _, kv := range environ {
		name, _, ok := strings.Cut(kv, "=")
		if ok && allowed[name] {
			out = append(out, kv)
		}
	}
	return out
}

func TestCopilotDoesNotHandOverEveryToolByDefault(t *testing.T) {
	record := filepath.Join(t.TempDir(), "record")
	p := provider(t, Copilot, "fake-echo-args.sh", func(o *Options) {
		o.Env = append(os.Environ(), "GOOSE_TEST_RECORD="+record)
	})
	if _, err := p.Complete(context.Background(), attempt(hello())); err != nil {
		t.Fatal(err)
	}
	body, _ := os.ReadFile(record)
	if strings.Contains(string(body), "--allow-all-tools") {
		t.Fatalf("argv = %.200s: every tool on the host is not a default", body)
	}

	record = filepath.Join(t.TempDir(), "record")
	opted := provider(t, Copilot, "fake-echo-args.sh", func(o *Options) {
		o.Env = append(os.Environ(), "GOOSE_TEST_RECORD="+record)
		o.Tools = "all"
	})
	if _, err := opted.Complete(context.Background(), attempt(hello())); err != nil {
		t.Fatal(err)
	}
	body, _ = os.ReadFile(record)
	if !strings.Contains(string(body), "--allow-all-tools") {
		t.Fatalf("argv = %.200s: an operator who asks for it must get it", body)
	}
}

func TestUnrestrictableAgentIsReportedOnEveryCall(t *testing.T) {
	p := provider(t, Antigravity, "fake-echo-args.sh", nil)
	var diags llm.Diagnostics
	a := attempt(hello())
	a.Recorder = &diags
	if _, err := p.Complete(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	var found bool
	for _, d := range diags {
		if d.Code == llm.PolicyAgentUnrestricted {
			found = true
		}
	}
	if !found {
		t.Fatalf("diagnostics = %+v: an agent goose cannot restrain is a fact the caller needs", diags)
	}
}

// Stream notes each prepare diagnostic onto a with the caller's loss policy
// applied, and the chain notes that same, already-applied copy onto the
// stream once it holds it (chain.go). Stream must not also note the raw
// diagnostic onto the stream itself, or the chain's copy lands beside an
// unpolicied duplicate. Regression test for r3-agentcli-2.
func TestPrepareDiagnosticsReachTheStreamOnlyOnce(t *testing.T) {
	p := provider(t, Antigravity, "fake-echo-args.sh", nil)
	var noted llm.Diagnostics
	a := attempt(hello())
	a.Recorder = &noted
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	// This is the chain's own step (chain.go), replayed here so the test does
	// not need the whole routing client to catch a duplicate that only shows
	// up once both sinks are merged, as a real call merges them.
	for _, d := range noted {
		s.Note(d)
	}
	for range s.Frames() {
	}
	n := 0
	for _, d := range s.Diagnostics() {
		if d.Code == llm.PolicyAgentUnrestricted {
			n++
		}
	}
	if n != 1 {
		t.Fatalf("policy.agent_unrestricted noted %d times on the stream, want 1", n)
	}
}

func TestSandboxIsRefusedWhereItCannotBeEnforced(t *testing.T) {
	for _, kind := range []Kind{Claude, Antigravity, Copilot} {
		if _, err := New(Options{Kind: kind, Binary: fake(t, "fake-echo-args.sh"), Sandbox: "read-only"}); err == nil {
			t.Fatalf("%s: a sandbox that cannot be enforced must be refused, not ignored", kind)
		}
	}
	if _, err := New(Options{Kind: Codex, Binary: fake(t, "fake-echo-args.sh"), Sandbox: "read-only"}); err != nil {
		t.Fatalf("codex takes a sandbox: %v", err)
	}
}

func TestToolPolicyIsReportedPerBinary(t *testing.T) {
	cases := []struct {
		kind  Kind
		tools string
		loud  bool
		why   string
	}{
		{Claude, "", false, "an empty list disables claude's tools"},
		{Claude, "default", true, "default leaves the binary's own tools live"},
		{Copilot, "", false, "copilot acts on nothing without the flag"},
		{Copilot, "all", true, "every tool on the host was handed over"},
		{Antigravity, "", true, "this binary cannot be restricted at all"},
		{Codex, "", true, "the sandbox bounds what its tools reach, not whether they run"},
	}
	for _, c := range cases {
		p := provider(t, c.kind, "fake-echo-args.sh", func(o *Options) { o.Tools = c.tools })
		var diags llm.Diagnostics
		a := attempt(hello())
		a.Recorder = &diags
		if _, err := p.Complete(context.Background(), a); err != nil {
			t.Fatal(err)
		}
		var loud bool
		for _, d := range diags {
			if d.Code == llm.PolicyAgentUnrestricted {
				loud = true
			}
		}
		if loud != c.loud {
			t.Errorf("%s tools=%q reported = %v, want %v: %s", c.kind, c.tools, loud, c.loud, c.why)
		}
	}
}

func TestProbeChecksTheBinaryWithoutRunningIt(t *testing.T) {
	present, err := New(Options{ID: "cli", Kind: Claude, Binary: fake(t, "fake-claude.sh")})
	if err != nil {
		t.Fatal(err)
	}
	if err := present.Probe(context.Background()); err != nil {
		t.Fatalf("a binary that is there is ready: %v", err)
	}

	absent, err := New(Options{ID: "cli", Kind: Claude, Binary: "/nonexistent/agent-binary"})
	if err != nil {
		t.Fatal(err)
	}
	if err := absent.Probe(context.Background()); err == nil {
		t.Fatal("a binary that is not there cannot serve")
	}
}

func TestModelsReportsTheConfiguredModelOnly(t *testing.T) {
	with, err := New(Options{ID: "cli", Kind: Claude, Binary: fake(t, "fake-claude.sh"), Model: "claude-sonnet-5"})
	if err != nil {
		t.Fatal(err)
	}
	models, err := with.Models(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(models) != 1 || models[0].ID != "claude-sonnet-5" {
		t.Fatalf("models = %+v", models)
	}

	without, err := New(Options{ID: "cli", Kind: Claude, Binary: fake(t, "fake-claude.sh")})
	if err != nil {
		t.Fatal(err)
	}
	if models, err := without.Models(context.Background()); err != nil || len(models) != 0 {
		t.Fatalf("models = %+v, err = %v: an unconfigured binary names no model", models, err)
	}
}

func TestClaudeSummaryIsReconciledNotAdded(t *testing.T) {
	p, err := New(Options{ID: "cli", Kind: Claude, Binary: fake(t, "fake-claude-summary.sh")})
	if err != nil {
		t.Fatal(err)
	}
	s, err := p.Stream(context.Background(), &llm.Attempt{
		Request: &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	for range s.Frames() {
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	u := s.Usage()
	if u.Prompt != 200 || u.Completion != 20 {
		t.Fatalf("usage = %d/%d, want 200/20: the summary was counted as a third turn", u.Prompt, u.Completion)
	}
	if u.Cost == nil || u.Cost.Amount != 0.02 {
		t.Fatalf("cost = %+v: the price is on the summary and nowhere else", u.Cost)
	}
	if u.DurationMS != 1000 {
		t.Fatalf("duration = %d", u.DurationMS)
	}
}

func TestAToolResultIsCarriedAndDeclared(t *testing.T) {
	p, err := New(Options{Kind: Claude, Binary: "claude"})
	if err != nil {
		t.Fatal(err)
	}
	if !p.Capabilities().ToolResults {
		t.Fatal("a backend that folds a tool result into the prompt takes tool results")
	}
	_, prompt, _ := flatten(&llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "what is it"}}},
		{Role: llm.RoleTool, Content: []llm.Part{&llm.ToolResult{ID: "1", Content: []llm.Part{&llm.Text{Text: "42"}}}}},
	}}, "cli")
	if !strings.Contains(prompt, "42") {
		t.Fatalf("prompt = %q", prompt)
	}
}

func TestWhatThePromptCouldNotCarryIsReported(t *testing.T) {
	_, _, diags := flatten(&llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Thinking{Text: "hmm"}}},
	}}, "cli")
	if !diags.Has(llm.DropThinking) {
		t.Errorf("diagnostics = %v", diags.Codes())
	}
	if !diags.Has(llm.DropMessage) {
		t.Errorf("diagnostics = %v: the turn vanished without a word", diags.Codes())
	}
}

func TestTheWholeTurnShapeIsRead(t *testing.T) {
	p := provider(t, Claude, "fake-claude-whole-turn.sh", nil)
	text, usage := drain(t, p)
	if text != "whole answer" {
		t.Fatalf("text = %q", text)
	}
	if usage.Prompt != 11 || usage.Completion != 4 {
		t.Fatalf("usage = %+v", usage)
	}
	if usage.Cost == nil || usage.Cost.Amount != 0.002 {
		t.Fatalf("cost = %+v", usage.Cost)
	}
}

func TestAStreamedTurnIsNotRepeatedByTheLineThatClosesIt(t *testing.T) {
	p := provider(t, Claude, "fake-claude-partial-then-whole.sh", nil)
	text, usage := drain(t, p)
	if text != "whole answer" {
		t.Fatalf("text = %q: the deltas carried the turn, so restating it whole must not add a second copy", text)
	}
	if usage.Prompt != 11 || usage.Completion != 4 {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestAnAnswerOnlyTheClosingLineCarriesStillArrives(t *testing.T) {
	p := provider(t, Claude, "fake-claude-only-result.sh", nil)
	text, usage := drain(t, p)
	if text != "late answer" {
		t.Fatalf("text = %q", text)
	}
	if usage.Prompt != 7 || usage.Completion != 3 {
		t.Fatalf("usage = %+v", usage)
	}
}

func drain(t *testing.T, p *Provider) (string, llm.Usage) {
	t.Helper()
	s, err := p.Stream(context.Background(), &llm.Attempt{
		Model: "claude-x", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "claude-x", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if tx, ok := part.(*llm.Text); ok {
				text.WriteString(tx.Text)
			}
		}
	}
	if err := s.Err(); err != nil {
		t.Fatalf("stream: %v", err)
	}
	return text.String(), s.Usage()
}

func TestARunThatEndsInToolWorkSaysSo(t *testing.T) {
	p := provider(t, Codex, "fake-codex-tools.sh", nil)
	s, err := p.Stream(context.Background(), &llm.Attempt{
		Model: "codex", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Model: "codex", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var calls []string
	stop := llm.StopNone
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if tu, ok := part.(*llm.ToolUse); ok {
				calls = append(calls, tu.Name)
			}
		}
		if f.Stop != llm.StopNone {
			stop = f.Stop
		}
	}
	if len(calls) != 1 || calls[0] != "command_execution" {
		t.Fatalf("calls = %v", calls)
	}
	if stop != llm.StopToolUse {
		t.Fatalf("stop = %q: a turn that ended in a tool call has not finished answering", stop)
	}
}

func TestTheAnthropicShapedCopilotStreamIsRead(t *testing.T) {
	p := provider(t, Copilot, "fake-copilot-anthropic.sh", nil)
	text, usage := drain(t, p)
	if text != "copilot answered" {
		t.Fatalf("text = %q", text)
	}
	if usage.Prompt != 5 || usage.Completion != 2 {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestARequestOverTheRouteBoundIsNotSpawned(t *testing.T) {
	p, err := New(Options{ID: "cli", Kind: Claude, Binary: fake(t, "fake-claude.sh")})
	if err != nil {
		t.Fatal(err)
	}
	a := &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Limits:  llm.Limits{MaxRequestBytes: 8},
		Request: &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: strings.Repeat("x", 64)}}}}},
	}
	_, err = p.Stream(context.Background(), a)
	e := llm.AsError(err)
	if e == nil || e.Code != llm.PolicyBodyTooLarge {
		t.Fatalf("err = %v: the bound is the route's, whatever the backend is", err)
	}
	if e.Sent {
		t.Fatal("nothing was spawned, so nothing was billed")
	}
}

func TestTheConfiguredTimeoutEndsTheChild(t *testing.T) {
	p := provider(t, Claude, "fake-slow.sh", func(o *Options) { o.Timeout = 300 * time.Millisecond })
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		_, _ = wirecore.Fold(s, wirecore.FoldOptions{})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the invocation outlived the timeout it was given")
	}
}

func TestAnAnswerOverTheRouteBoundIsRefused(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	a := attempt(hello())
	a.Limits = llm.Limits{MaxResponseBytes: 4}
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = wirecore.Fold(s, wirecore.FoldOptions{})
	e := llm.AsError(s.Err())
	if e == nil || e.Code != llm.PolicyBodyTooLarge {
		t.Fatalf("err = %v: an answer past the bound is refused, not served", s.Err())
	}
}

func TestTheBinarysOwnTurnCountIsKept(t *testing.T) {
	p := provider(t, Claude, "fake-claude-turns.sh", nil)
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := wirecore.Fold(s, wirecore.FoldOptions{}); err != nil {
		t.Fatal(err)
	}
	if got := s.Usage().Turns; got != 5 {
		t.Fatalf("turns = %d, want 5: the binary said how many it billed", got)
	}
}

func TestCodexToolArgumentsCarryTheCallNotItsLabel(t *testing.T) {
	raw := json.RawMessage(`{"id":"i2","type":"command_execution","name":"shell","message":"running","command":"ls -la"}`)
	got := string(codexToolArgs(raw))
	for _, gone := range []string{`"id"`, `"type"`, `"name"`, `"message"`} {
		if strings.Contains(got, gone) {
			t.Fatalf("args = %s: %s names the call, it does not describe it", got, gone)
		}
	}
	if !strings.Contains(got, `"command":"ls -la"`) {
		t.Fatalf("args = %s: what the tool was asked to do must survive", got)
	}
}

func TestARestatedTurnStillCarriesWhatTheDeltasCouldNot(t *testing.T) {
	kept := unstreamed([]llm.Part{
		&llm.Text{Text: "streamed"},
		&llm.Thinking{Text: "streamed"},
		&llm.ToolUse{ID: "t1", Name: "search"},
		&llm.Thinking{Redacted: []byte("opaque")},
		&llm.Raw{Vendor: "anthropic", Type: "server_tool_use"},
	})
	if len(kept) != 2 {
		t.Fatalf("kept = %+v: only the blocks no stream event spells out survive the restatement", kept)
	}
	if th, ok := kept[0].(*llm.Thinking); !ok || len(th.Redacted) == 0 {
		t.Fatalf("kept[0] = %+v: redacted thinking has no delta to arrive on", kept[0])
	}
	if raw, ok := kept[1].(*llm.Raw); !ok || raw.Type != "server_tool_use" {
		t.Fatalf("kept[1] = %+v: a block this parser has no event for must not be dropped", kept[1])
	}
}

func TestAServerToolsStreamedInputIsNotTheClientsCall(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	st := &state{}
	var args strings.Builder
	for _, line := range []string{
		`{"type":"stream_event","event":{"type":"message_start","message":{"id":"m"}}}`,
		`{"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"search"}}}`,
		`{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":1}"}}}`,
		`{"type":"stream_event","event":{"type":"content_block_start","index":1,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search"}}}`,
		`{"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"goose\"}"}}}`,
	} {
		frames, err := claudeParse(p, []byte(line), st)
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range frames {
			for _, tu := range llm.ToolUses(f.Delta) {
				if *tu.Index != 0 {
					t.Fatalf("call %d = %+v: a server tool's input opened a call for the client to run", *tu.Index, tu)
				}
				args.Write(tu.Args)
			}
		}
	}
	if args.String() != `{"q":1}` {
		t.Fatalf("args = %s: a server tool's input was written into the client's call", args.String())
	}
}

func TestAgentsAreGatedByDefault(t *testing.T) {
	p, err := New(Options{Kind: Claude, Binary: fake(t, "fake-claude.sh"), Env: os.Environ()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = p.Close(context.Background()) })
	if p.gate == nil {
		t.Fatal("every call forks a whole agent CLI, so an ungated provider is bounded only by the host's memory")
	}
	if cap(p.gate) != DefaultMaxConcurrent {
		t.Fatalf("gate = %d, want %d", cap(p.gate), DefaultMaxConcurrent)
	}

	off, err := New(Options{Kind: Claude, Binary: fake(t, "fake-claude.sh"), Env: os.Environ(), MaxConcurrent: -1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = off.Close(context.Background()) })
	if off.gate != nil {
		t.Fatal("a negative limit is how an operator asks for no gate at all")
	}
}

func TestASessionHandleMustLookLikeOne(t *testing.T) {
	for _, bad := range []string{"--sandbox=danger-full-access", "-x", "a b", "a;b", "a$(id)", ""} {
		if safeSession(bad) {
			t.Fatalf("%q was accepted: this reaches a child process's argument list", bad)
		}
	}
	for _, good := range []string{"sess-42", "t1", "01JABC.def:ghi", "a_b-c"} {
		if !safeSession(good) {
			t.Fatalf("%q was refused: this is what the binaries actually issue", good)
		}
	}
}

func TestAModelThatReadsAsAFlagIsRefused(t *testing.T) {
	for _, kind := range []Kind{Claude, Codex, Antigravity, Copilot} {
		t.Run(string(kind), func(t *testing.T) {
			record := filepath.Join(t.TempDir(), "record")
			p := provider(t, kind, "fake-echo-args.sh", func(o *Options) {
				o.Env = append(os.Environ(), "GOOSE_TEST_RECORD="+record)
				o.TempDir = t.TempDir()
			})
			req := hello()
			req.Model = "--yolo"
			a := attempt(req)
			a.Model = ""
			_, err := p.Complete(context.Background(), a)
			if e := llm.AsError(err); e == nil || e.Kind != llm.KindInvalidRequest || e.Sent {
				t.Fatalf("err = %v: a request's model goes into the child's argument list", err)
			}
			if _, err := os.Stat(record); !os.IsNotExist(err) {
				t.Fatal("the binary was run with a model that could read as one of its flags")
			}

			if _, err := p.Complete(context.Background(), attempt(hello())); err != nil {
				t.Fatalf("a model named as models are named must still reach the binary: %v", err)
			}
		})
	}
}

// Tools is false because none of these binaries takes a caller's tool
// definitions: they run their own, and a caller that offers tools is told they
// were dropped. Thinking is false because none of them is given a thinking
// setting, so a caller that asks for one is told rather than ignored.
func TestTheCapabilitiesAreWhatTheBinariesTake(t *testing.T) {
	for _, kind := range []Kind{Claude, Codex, Antigravity, Copilot} {
		caps := provider(t, kind, "fake-echo-args.sh", nil).Capabilities()
		if caps.Tools || caps.ToolChoice || caps.MCPServers {
			t.Errorf("%s: caps = %+v: no tool definition reaches the binary", kind, caps)
		}
		if caps.Thinking || caps.Effort {
			t.Errorf("%s: caps = %+v: no thinking setting reaches the binary", kind, caps)
		}
	}
}

func TestOneCallersSessionIsNotAnothersToResume(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	p.remember("sess-42", owner(llm.Credential{Key: "tenant-a"}))

	req := hello()
	a := attempt(req)
	a.Session = "sess-42"
	a.Credential = llm.Credential{Key: "tenant-b"}
	var diags llm.Diagnostics
	a.Recorder = &diags

	in, _, err := p.prepare(a)
	if err != nil {
		t.Fatal(err)
	}
	if in.Session != "" {
		t.Fatal("another caller's conversation was taken up, with its history and its working directory")
	}

	a2 := attempt(hello())
	a2.Session = "sess-42"
	a2.Credential = llm.Credential{Key: "tenant-a"}
	in2, _, err := p.prepare(a2)
	if err != nil {
		t.Fatal(err)
	}
	if in2.Session != "sess-42" {
		t.Fatal("the caller it was issued to must still be able to resume it")
	}
}

// Every flag goose writes is one the binary has. A CLI is free to rename or
// drop one between releases, and the first anyone would hear of it is a call
// failing in production, so the binary itself is asked where there is one.
func TestTheFlagsTheseBinariesAreGivenExist(t *testing.T) {
	for _, tc := range []struct {
		binary string
		flags  []string
	}{
		{"claude", []string{
			"--output-format", "--include-partial-messages", "--model", "--resume",
			"--system-prompt-file", "--append-system-prompt", "--system-prompt",
			"--max-turns", "--tools",
		}},
		{"copilot", []string{"--output-format", "--no-ask-user", "--allow-all-tools", "--model", "--resume"}},
	} {
		t.Run(tc.binary, func(t *testing.T) {
			path, err := exec.LookPath(tc.binary)
			if err != nil {
				t.Skipf("%s is not installed here, so its own grammar cannot be asked", tc.binary)
			}
			for _, flag := range tc.flags {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				cmd := exec.CommandContext(ctx, path, flag)
				cmd.Stdin = strings.NewReader("")
				out, _ := cmd.CombinedOutput()
				cancel()
				// A flag that exists but was given no value says so; one that
				// does not exist is refused as unknown.
				for _, unknown := range []string{"unknown option", "unexpected argument", "unknown flag", "unrecognized"} {
					if bytes.Contains(bytes.ToLower(out), []byte(unknown)) {
						t.Errorf("%s %s: %s", tc.binary, flag, bytes.TrimSpace(out))
					}
				}
			}
		})
	}
}

// The order matters: codex takes its own flags before the subcommand, and a
// flag written after `resume` is read as a positional argument and refused.
// Run against the real binary where there is one, since the grammar is its.
func TestTheCodexArgumentsParse(t *testing.T) {
	binary, err := exec.LookPath("codex")
	if err != nil {
		t.Skip("codex is not installed here, so its own grammar cannot be asked")
	}
	for _, tc := range []struct {
		name string
		in   invocation
	}{
		{"a fresh session", invocation{Model: "gpt-5"}},
		{"a resumed one", invocation{Model: "gpt-5", Session: "00000000-0000-0000-0000-000000000000"}},
		{"no model", invocation{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := codexArgs(&Provider{}, tc.in)
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, binary, args...)
			cmd.Stdin = strings.NewReader("")
			out, _ := cmd.CombinedOutput()
			// An empty prompt stops it before it reaches the network, which is
			// past the point where the arguments have already been read.
			if bytes.Contains(out, []byte("unexpected argument")) ||
				bytes.Contains(out, []byte("unrecognized subcommand")) {
				t.Fatalf("codex %s: %s", strings.Join(args, " "), out)
			}
		})
	}
}
