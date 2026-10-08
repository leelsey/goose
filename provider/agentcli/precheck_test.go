// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"testing"

	"github.com/leelsey/goose/llm"
)

func silent() *llm.Request {
	return &llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Thinking{Text: "nothing to say"}}},
	}}
}

// What the binary is never run for is known before any slot is waited for, so
// Precheck refuses it outright rather than leaving it to queue behind calls
// that can use the slot.
func TestPrecheckRefusesWhatTheBinaryIsNeverRunFor(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	flag := attempt(hello())
	flag.Model = "--yolo"
	for _, tc := range []struct {
		what string
		a    *llm.Attempt
		lost string
	}{
		{"a model that reads as a flag", flag, ""},
		{"a request with no prompt left", attempt(silent()), llm.DropMessage},
	} {
		diags, err := p.Precheck(tc.a)
		if e := llm.AsError(err); e == nil || e.Kind != llm.KindInvalidRequest || e.Sent {
			t.Fatalf("%s: err = %v", tc.what, err)
		}
		if tc.lost != "" && !diags.Has(tc.lost) {
			t.Fatalf("%s: diagnostics = %v: the refusal says nothing of what the prompt lost", tc.what, diags.Codes())
		}
	}
	if _, err := p.Precheck(attempt(hello())); err != nil {
		t.Fatalf("err = %v: a request the binary can be run for was refused", err)
	}
}

// A request refused for having no prompt left still says what took it away,
// so the row booked for the refusal explains it.
func TestARefusalSaysWhatTheRequestLostOnTheWay(t *testing.T) {
	p := provider(t, Claude, "fake-claude.sh", nil)
	var noted llm.Diagnostics
	a := attempt(silent())
	a.Recorder = &noted
	if _, err := p.Stream(context.Background(), a); llm.AsError(err) == nil || llm.AsError(err).Kind != llm.KindInvalidRequest {
		t.Fatalf("err = %v", err)
	}
	if !noted.Has(llm.DropMessage) {
		t.Fatalf("noted = %v: the refusal left no word of the turn that left no prompt", noted.Codes())
	}
}
