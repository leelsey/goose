// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/base64"
	"testing"

	"github.com/leelsey/goose/llm"
)

// A signature handed to a client comes back attributed to whoever minted it,
// whichever wire carried it, and one that already reads as goose's own is not
// taken apart on the way back.
func TestASignatureComesBackFromTheClientAsItWentOut(t *testing.T) {
	for _, tc := range []struct {
		native, signer llm.Family
		sig, sealed    string
		back           llm.Family
	}{
		{llm.FamilyAnthropic, llm.FamilyAnthropic, "EqQB", "EqQB", llm.FamilyAnthropic},
		{llm.FamilyAnthropic, "", "EqQB", "EqQB", llm.FamilyAnthropic},
		{llm.FamilyAnthropic, llm.FamilyOpenAI, "gAAAA", "goose.v1.openai.gAAAA", llm.FamilyOpenAI},
		{llm.FamilyOpenAI, llm.FamilyGoogle, "CiQB", "goose.v1.google.CiQB", llm.FamilyGoogle},
		{llm.FamilyGoogle, llm.FamilyGoogle, "goose.v1.anthropic.EqQB", "goose.v1.google.goose.v1.anthropic.EqQB", llm.FamilyGoogle},
	} {
		sealed := SealSignature(tc.native, tc.signer, tc.sig)
		if sealed != tc.sealed {
			t.Errorf("seal(%s, %s, %s) = %q, want %q", tc.native, tc.signer, tc.sig, sealed, tc.sealed)
		}
		signer, sig := OpenSignature(tc.native, sealed)
		if signer != tc.back || sig != tc.sig {
			t.Errorf("open(%s, %q) = %s %q, want %s %q", tc.native, sealed, signer, sig, tc.back, tc.sig)
		}
	}
	if signer, sig := OpenSignature(llm.FamilyGoogle, ""); signer != "" || sig != "" {
		t.Fatalf("open of nothing = %s %q: no signature has no signer", signer, sig)
	}
	if signer, sig := OpenSignature(llm.FamilyGoogle, "goose.v1..x"); signer != llm.FamilyGoogle || sig != "goose.v1..x" {
		t.Fatalf("open of a malformed prefix = %s %q: it is the wire's own, as sent", signer, sig)
	}
}

// Where the wire's field holds bytes, what goes out is base64 whole, and it
// comes back attributed in whichever base64 alphabet the client's SDK uses.
func TestASignatureInABytesFieldIsBase64ToTheClient(t *testing.T) {
	marked := base64.StdEncoding.EncodeToString([]byte("goose.v1.anthropic.x"))
	for _, tc := range []struct {
		signer llm.Family
		sig    string
	}{
		{llm.FamilyGoogle, "CiQBcQ+/"},
		{"", "CiQBcQ+/"},
		{llm.FamilyAnthropic, "EqQBCkgIARABGAIiQPla+/8="},
		{llm.FamilyOpenAI, "gAAAAABo-x_y"},
		{llm.FamilyGoogle, marked},
	} {
		sealed := SealBytesSignature(llm.FamilyGoogle, tc.signer, tc.sig)
		raw, err := base64.StdEncoding.DecodeString(sealed)
		if err != nil {
			t.Fatalf("seal(%s, %q) = %q: a bytes field is base64: %v", tc.signer, tc.sig, sealed, err)
		}
		want := tc.signer
		if want == "" {
			want = llm.FamilyGoogle
		}
		for _, back := range []string{sealed, base64.RawURLEncoding.EncodeToString(raw)} {
			if want == llm.FamilyGoogle && back != sealed {
				continue
			}
			if signer, sig := OpenBytesSignature(llm.FamilyGoogle, back); signer != want || sig != tc.sig {
				t.Errorf("open(%q) = %s %q, want %s %q", back, signer, sig, want, tc.sig)
			}
		}
	}
	if s := SealBytesSignature(llm.FamilyGoogle, llm.FamilyGoogle, "CiQB"); s != "CiQB" {
		t.Fatalf("seal of Google's own = %q: it goes out as Google issued it", s)
	}
	if signer, sig := OpenBytesSignature(llm.FamilyGoogle, "Z29vc2UudjEu!"); signer != llm.FamilyGoogle || sig != "Z29vc2UudjEu!" {
		t.Fatalf("open of a malformed mark = %s %q: it is the wire's own, as sent", signer, sig)
	}
}
