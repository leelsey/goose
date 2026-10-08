// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package goose

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
)

func signing(id string, family llm.Family) *capped {
	caps := full()
	caps.Signer = family
	return &capped{id: id, caps: caps}
}

// A backend refuses a signature another family minted with a 400 that ends the
// chain, so one is stripped before the call and the loss is said. A signature
// its own family minted, or one nobody attributed, is left to the backend.
func TestASignatureAnotherFamilyMintedIsStrippedBeforeTheCall(t *testing.T) {
	p := signing("claude", llm.FamilyAnthropic)
	c, err := New(WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{ID: "rs_up", Text: "because", Signature: "gAAAA", Signer: llm.FamilyOpenAI},
			&llm.Thinking{Text: "mine", Signature: "EqQB", Signer: llm.FamilyAnthropic},
			&llm.Thinking{Text: "kept", Signature: "unattributed"},
			&llm.ToolUse{ID: "c1", Name: "f", Args: json.RawMessage(`{}`), Signature: "CiQB", Signer: llm.FamilyGoogle},
			&llm.Text{Signature: "CiQC", Signer: llm.FamilyGoogle},
		}},
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	sent := p.seen.Request.Messages[0].Content
	if len(sent) != 4 {
		t.Fatalf("content = %+v: a part that held only a foreign signature holds nothing here", sent)
	}
	if th := sent[0].(*llm.Thinking); th.Signature != "" || th.ID != "" || th.Text != "because" {
		t.Fatalf("foreign thinking = %+v: its signature and the name its backend gave it go, its text stays", th)
	}
	if th := sent[1].(*llm.Thinking); th.Signature != "EqQB" {
		t.Fatalf("own thinking = %+v: this backend's own signature is what it asks for back", th)
	}
	if th := sent[2].(*llm.Thinking); th.Signature != "unattributed" {
		t.Fatalf("unattributed thinking = %+v: nobody said it was foreign", th)
	}
	if call := sent[3].(*llm.ToolUse); call.Signature != "" {
		t.Fatalf("call = %+v: Gemini's signature is nothing Claude can verify", call)
	}
	notes := resp.Diagnostics.Find(llm.DropSignature)
	if len(notes) != 1 || notes[0].Count != 3 {
		t.Fatalf("diagnostics = %+v: three signatures were not sent", resp.Diagnostics)
	}
}

// The chain a failover builds: OpenAI served a turn and the client kept its
// reasoning. With Claude back, a caller who refuses the loss is not sent to
// Claude to be refused there; the link that minted the signature serves it.
func TestAChainHandsAForeignSignatureToTheFamilyThatMintedIt(t *testing.T) {
	claude := signing("claude", llm.FamilyAnthropic)
	gpt := signing("gpt", llm.FamilyOpenAI)
	c, err := New(
		WithProvider(claude), WithProvider(gpt),
		WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "claude"}, {Provider: "gpt"},
		}}),
		WithLossPolicy(llm.LossPolicy{Strict: true}),
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Thinking{ID: "rs_up", Text: "because", Signature: "gAAAA", Signer: llm.FamilyOpenAI},
		}},
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if claude.seen != nil {
		t.Fatal("claude was sent a signature it cannot verify")
	}
	if gpt.seen == nil {
		t.Fatal("the link that minted the signature was never tried")
	}
	if th := gpt.seen.Request.Messages[0].Content[0].(*llm.Thinking); th.Signature != "gAAAA" || th.ID != "rs_up" {
		t.Fatalf("thinking = %+v: its own backend gets it whole", th)
	}
}
