// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/base64"
	"strings"

	"github.com/leelsey/goose/llm"
)

// A client's wire has no field for who minted a signature, so one handed to a
// client on a wire of another family carries its signer inside it. Vendor
// signatures are base64, which has no dot, so the prefix cannot be mistaken for
// one of theirs.
const signaturePrefix = "goose.v1."

// sealedBytesPrefix begins every signature SealBytesSignature marks. The
// prefix is nine bytes, which base64 spells in twelve characters with no
// padding and none of the characters its URL-safe spelling swaps.
var sealedBytesPrefix = base64.StdEncoding.EncodeToString([]byte(signaturePrefix))

// Signed is the signature a part carries and the family that minted it.
func Signed(p llm.Part) (string, llm.Family) {
	switch v := p.(type) {
	case *llm.Text:
		return v.Signature, v.Signer
	case *llm.Thinking:
		return v.Signature, v.Signer
	case *llm.ToolUse:
		return v.Signature, v.Signer
	case *llm.Image:
		return v.Signature, v.Signer
	}
	return "", ""
}

// SealSignature spells a signature for a client on a wire whose own family is
// native. One minted by another family is prefixed with its signer, and so is
// one that already reads as prefixed, so that OpenSignature gives back exactly
// what went in.
func SealSignature(native, signer llm.Family, sig string) string {
	if sig == "" {
		return ""
	}
	if signer == "" {
		signer = native
	}
	if signer == native && !strings.HasPrefix(sig, signaturePrefix) {
		return sig
	}
	return signaturePrefix + string(signer) + "." + sig
}

// SealBytesSignature is SealSignature for a wire whose signature field holds
// bytes, as Gemini's thoughtSignature does. Its typed clients decode that field
// as base64 and cannot read an answer that holds anything else, so the marked
// form goes out as base64 itself.
func SealBytesSignature(native, signer llm.Family, sig string) string {
	if sig == "" {
		return ""
	}
	if signer == "" {
		signer = native
	}
	if signer == native && !strings.HasPrefix(sig, sealedBytesPrefix) {
		return sig
	}
	return base64.StdEncoding.EncodeToString([]byte(signaturePrefix + string(signer) + "." + sig))
}

// OpenSignature reads a signature a client sent on a wire whose own family is
// native. One without the prefix is that family's.
func OpenSignature(native llm.Family, s string) (llm.Family, string) {
	if s == "" {
		return "", ""
	}
	if signer, sig, ok := unseal(s); ok {
		return signer, sig
	}
	return native, s
}

// OpenBytesSignature reads what SealBytesSignature wrote. A client's SDK may
// hand the bytes back in either base64 alphabet, padded or not.
func OpenBytesSignature(native llm.Family, s string) (llm.Family, string) {
	if s == "" {
		return "", ""
	}
	if strings.HasPrefix(s, sealedBytesPrefix) {
		trimmed := strings.TrimRight(s, "=")
		for _, enc := range []*base64.Encoding{base64.RawStdEncoding, base64.RawURLEncoding} {
			if b, err := enc.DecodeString(trimmed); err == nil {
				if signer, sig, ok := unseal(string(b)); ok {
					return signer, sig
				}
			}
		}
	}
	return native, s
}

func unseal(s string) (llm.Family, string, bool) {
	rest, ok := strings.CutPrefix(s, signaturePrefix)
	if !ok {
		return "", "", false
	}
	signer, sig, ok := strings.Cut(rest, ".")
	if !ok || signer == "" || sig == "" {
		return "", "", false
	}
	return llm.Family(signer), sig, true
}
