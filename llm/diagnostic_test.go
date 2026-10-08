// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"errors"
	"os"
	"regexp"
	"testing"
)

func lost() Diagnostic {
	return Diagnostic{Code: DropThinking, Class: ClassContent, Stage: StageEncode, Severity: SeverityLoss}
}

func TestStrictPolicyFailsOnContentLoss(t *testing.T) {
	err := LossPolicy{Strict: true}.Check([]Diagnostic{lost()})
	if err == nil {
		t.Fatal("strict mode must refuse to deliver a lossy translation")
	}
	e := AsError(err)
	if e.Kind != KindLoss || e.Code != DropThinking {
		t.Fatalf("error = %+v", e)
	}
}

func TestDefaultPolicyReportsWithoutFailing(t *testing.T) {
	if err := (LossPolicy{}).Check([]Diagnostic{lost()}); err != nil {
		t.Fatalf("default policy failed the call: %v", err)
	}
}

func TestAllowDowngradesAndDenyPromotes(t *testing.T) {
	p := LossPolicy{Strict: true, Allow: []string{DropThinking}}
	if err := p.Check([]Diagnostic{lost()}); err != nil {
		t.Fatalf("allowed code still fatal: %v", err)
	}
	d := Diagnostic{Code: DropImage, Class: ClassContent, Severity: SeverityInfo}
	if err := (LossPolicy{Deny: []string{DropImage}}).Check([]Diagnostic{d}); err == nil {
		t.Fatal("denied code must fail the call")
	}
}

func TestFatalDiagnosticIsAlwaysFatal(t *testing.T) {
	d := Diagnostic{Code: DropToolCall, Class: ClassContent, Severity: SeverityFatal}
	err := (LossPolicy{}).Check([]Diagnostic{d})
	if err == nil {
		t.Fatal("a site that marked a loss fatal must not be overridden by the default policy")
	}
	if !errors.Is(err, &Error{Kind: KindLoss}) {
		t.Fatalf("err = %v", err)
	}
}

func TestDiagnosticsRecorder(t *testing.T) {
	var d Diagnostics
	var r Recorder = &d
	r.Note(lost())
	r.Note(Diagnostic{Code: CapNoStreamUsage, Class: ClassAccounting, Severity: SeverityInfo})
	if len(d) != 2 || d.Worst() != SeverityLoss {
		t.Fatalf("diagnostics = %+v worst = %v", d, d.Worst())
	}
	if codes := d.Codes(); len(codes) != 2 {
		t.Fatalf("codes = %v", codes)
	}
}

func TestCapabilitiesReadZeroCandidatesAsOne(t *testing.T) {
	if n := (Capabilities{}).MaxCandidates(); n != 1 {
		t.Fatalf("candidates = %d, want 1", n)
	}
	if n := (Capabilities{Candidates: 4}).MaxCandidates(); n != 4 {
		t.Fatalf("candidates = %d, want 4", n)
	}
}

func TestNilFrameIsEmpty(t *testing.T) {
	var f *Frame
	if !f.Empty() {
		t.Fatal("a nil frame must not read as carrying content")
	}
}

func TestStrictPromotesContentAndNotCapability(t *testing.T) {
	strict := LossPolicy{Strict: true}
	content := Diagnostic{Code: DropUnknownBlock, Class: ClassContent, Severity: SeverityLoss}
	if strict.Apply(content).Severity != SeverityFatal {
		t.Fatal("a dropped content block is a loss strict refuses")
	}
	setting := Diagnostic{Code: DropToolDefinition, Class: ClassCapability, Severity: SeverityInfo}
	if strict.Apply(setting).Severity == SeverityFatal {
		t.Fatal("a setting a backend does not honour is not lost content")
	}
	if err := strict.Check([]Diagnostic{setting}); err != nil {
		t.Fatalf("check = %v", err)
	}
	if err := strict.Check([]Diagnostic{setting, content}); err == nil {
		t.Fatal("a translation that lost content is not delivered under strict")
	}
}

// Strict refuses content that was lost. A block carried through verbatim lost
// nothing, and a caller who denies the code by name still refuses it.
func TestStrictLeavesANoteThatLostNothingAlone(t *testing.T) {
	verbatim := Diagnostic{
		Code: DropUnknownBlock, Class: ClassContent, Severity: SeverityInfo,
		Detail: "content block carried verbatim",
	}
	if got := (LossPolicy{Strict: true}).Apply(verbatim).Severity; got != SeverityInfo {
		t.Fatalf("severity = %v: nothing was lost", got)
	}
	if err := (LossPolicy{Deny: []string{DropUnknownBlock}}).Check([]Diagnostic{verbatim}); err == nil {
		t.Fatal("a code the caller denies is refused whatever severity it came with")
	}
}

// A site reporting a drop at info still dropped it: thinking that could not be
// replayed without its signature was not sent at all, and strict refuses that
// as it always has.
func TestStrictStillRefusesContentReportedAtInfo(t *testing.T) {
	unsigned := Diagnostic{
		Code: DropSignature, Class: ClassContent, Stage: StageRequest, Severity: SeverityInfo,
		Detail: "thinking without its signature cannot be replayed to this backend",
	}
	if err := (LossPolicy{Strict: true}).Check([]Diagnostic{unsigned}); err == nil {
		t.Fatal("strict delivered an answer to a conversation whose thinking was left out")
	}
}

func TestEveryDeclaredCodeIsOnTheList(t *testing.T) {
	src, err := os.ReadFile("diagnostic.go")
	if err != nil {
		t.Fatal(err)
	}
	declared := regexp.MustCompile(`(?m)^\t([A-Z]\w*)\s*=\s*"((?:drop|cap|acct|wire|policy)\.[a-z_]+)"`).
		FindAllStringSubmatch(string(src), -1)
	if len(declared) < 20 {
		t.Fatalf("found %d codes; the check is not looking where they are", len(declared))
	}
	for _, d := range declared {
		if !KnownCode(d[2]) {
			t.Errorf("%s (%s) is declared and not on the list a policy is checked against", d[1], d[2])
		}
	}
	if len(Codes()) != len(declared) {
		t.Errorf("the list has %d codes and %d are declared", len(Codes()), len(declared))
	}
}
