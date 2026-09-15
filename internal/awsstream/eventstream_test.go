// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package awsstream

import (
	"bytes"
	"errors"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	buf.Write(Encode(map[string]string{":message-type": "event", ":event-type": "messageStart"}, []byte(`{"role":"assistant"}`)))
	buf.Write(Encode(map[string]string{":message-type": "event", ":event-type": "contentBlockDelta"}, []byte(`{"delta":{"text":"hi"}}`)))

	r := NewReader(&buf, 0)
	var types []string
	for r.Next() {
		types = append(types, r.Message().Type())
	}
	if r.Err() != nil {
		t.Fatal(r.Err())
	}
	if len(types) != 2 || types[0] != "messageStart" || types[1] != "contentBlockDelta" {
		t.Fatalf("types = %v", types)
	}
}

func TestCorruptedFrameIsAnError(t *testing.T) {
	msg := Encode(map[string]string{":event-type": "x"}, []byte(`{"a":1}`))
	msg[len(msg)-1] ^= 0xff
	r := NewReader(bytes.NewReader(msg), 0)
	if r.Next() {
		t.Fatal("a frame that fails its checksum must not be delivered")
	}
	if !errors.Is(r.Err(), ErrMessageCRC) {
		t.Fatalf("err = %v", r.Err())
	}
}

func TestTruncatedStreamIsAnError(t *testing.T) {
	msg := Encode(map[string]string{":event-type": "x"}, []byte(`{"a":1}`))
	r := NewReader(bytes.NewReader(msg[:len(msg)-3]), 0)
	if r.Next() {
		t.Fatal("half a frame is not a frame")
	}
	if r.Err() == nil {
		t.Fatal("a stream that ends mid-message must say so")
	}
}

func TestOversizedMessageIsRefused(t *testing.T) {
	msg := Encode(map[string]string{":event-type": "x"}, bytes.Repeat([]byte("y"), 4096))
	r := NewReader(bytes.NewReader(msg), 128)
	if r.Next() {
		t.Fatal("an oversized frame must not be buffered")
	}
	if !errors.Is(r.Err(), ErrTooLarge) {
		t.Fatalf("err = %v", r.Err())
	}
}

func TestExceptionMessagesAreTyped(t *testing.T) {
	msg := Encode(map[string]string{":message-type": "exception", ":exception-type": "throttlingException"}, []byte(`{"message":"slow down"}`))
	r := NewReader(bytes.NewReader(msg), 0)
	if !r.Next() {
		t.Fatal(r.Err())
	}
	if r.Message().Kind() != "exception" || r.Message().Type() != "throttlingException" {
		t.Fatalf("message = %+v", r.Message())
	}
}
