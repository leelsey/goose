// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import (
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
)

// An attempt that was sent and then discarded is a row like any other, and an
// operator who asked for an upstream's own words to be kept out of the ledger
// asked about every row rather than only the last one.
func TestADiscardedAttemptIsRedactedLikeAnyOther(t *testing.T) {
	const secret = "MY-SECRET-PASSPHRASE"
	boom := &llm.Error{
		Kind: llm.KindRateLimit, Status: 429, Sent: true,
		Message: "blocked: user said " + secret + " in message 3",
		Body:    `{"error":{"message":"blocked: user said ` + secret + ` in message 3"}}`,
	}
	for _, redact := range []bool{false, true} {
		var booked []llm.Attempted
		a := &llm.Attempt{
			Provider: "one", Model: "m", Redact: redact,
			Book: func(rec llm.Attempted) { booked = append(booked, rec) },
		}
		BookDiscards(a)(boom, time.Second)
		if len(booked) != 1 {
			t.Fatalf("rows = %d, want the discarded attempt to be booked", len(booked))
		}
		held := strings.Contains(booked[0].Err, secret)
		switch {
		case redact && held:
			t.Errorf("a discarded row kept the upstream's own words: %q", booked[0].Err)
		case !redact && !held:
			t.Errorf("without redaction the message is what an operator diagnoses from: %q", booked[0].Err)
		}
	}
}
