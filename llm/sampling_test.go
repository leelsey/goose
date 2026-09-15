// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import "testing"

func TestUnsentSamplingNamesWhatDidNotTravel(t *testing.T) {
	req := &Request{
		Temperature:      Ptr(0.5),
		TopP:             Ptr(0.9),
		TopK:             Ptr(40),
		Seed:             Ptr(int64(7)),
		Stop:             []string{"x"},
		FrequencyPenalty: Ptr(0.1),
		PresencePenalty:  Ptr(0.2),
		MaxOutputTokens:  Ptr(64),
	}
	got := UnsentSampling(req, "p", FieldTemperature, FieldTopP, FieldMaxOutputTokens)
	want := map[string]bool{
		FieldTopK: true, FieldSeed: true, FieldStop: true,
		FieldFrequencyPenalty: true, FieldPresencePenalty: true,
	}
	if len(got) != len(want) {
		t.Fatalf("diagnostics = %d: %v", len(got), got)
	}
	for _, d := range got {
		if !want[d.Field] {
			t.Errorf("%s was reported as unsent and it was sent", d.Field)
		}
		if d.Provider != "p" || d.Severity != SeverityInfo || d.Class != ClassCapability {
			t.Errorf("diagnostic = %+v", d)
		}
	}
}

func TestUnsentSamplingIsSilentForWhatNobodyAsked(t *testing.T) {
	if got := UnsentSampling(&Request{}, "p"); len(got) != 0 {
		t.Fatalf("diagnostics = %v", got)
	}
}
