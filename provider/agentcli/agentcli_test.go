// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"encoding/json"
	"os"
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

func TestConcurrencyGateShedsRatherThanQueues(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", func(o *Options) { o.MaxConcurrent = 1 })
	release, err := p.Acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Acquire(context.Background()); err == nil {
		t.Fatal("a saturated subscription backend must refuse rather than pile up")
	} else if llm.AsError(err).Kind != llm.KindSaturated {
		t.Fatalf("err = %v", err)
	}
	release()
	if _, err := p.Acquire(context.Background()); err != nil {
		t.Fatalf("the slot was not returned: %v", err)
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
