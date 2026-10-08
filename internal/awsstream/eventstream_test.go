// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package awsstream

import (
	"bytes"
	"encoding/hex"
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

// The empty message from the test vectors AWS publishes for this framing
// (aws-c-event-stream's empty_message): a twelve-byte prelude and its checksum,
// no headers, no payload, and the message checksum.
func TestAWSReferenceFrameIsRead(t *testing.T) {
	vector, _ := hex.DecodeString("00000010" + "00000000" + "05c248eb" + "7d98c8ff")
	r := NewReader(bytes.NewReader(vector), 0)
	if !r.Next() {
		t.Fatalf("the reference frame was refused: %v", r.Err())
	}
	if m := r.Message(); len(m.Headers) != 0 || len(m.Payload) != 0 {
		t.Fatalf("message = %+v", m)
	}
	if r.Next() || r.Err() != nil {
		t.Fatalf("one frame, then a clean end: %v", r.Err())
	}
	if got := Encode(nil, nil); !bytes.Equal(got, vector) {
		t.Fatalf("encoded = %x, want %x: the tests would agree with a reader AWS does not", got, vector)
	}

	castagnoli, _ := hex.DecodeString("00000010" + "00000000" + "030ad359" + "6d16a789")
	r = NewReader(bytes.NewReader(castagnoli), 0)
	if r.Next() {
		t.Fatal("a frame checksummed with CRC-32C was accepted")
	}
	if !errors.Is(r.Err(), ErrPreludeCRC) {
		t.Fatalf("err = %v", r.Err())
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
