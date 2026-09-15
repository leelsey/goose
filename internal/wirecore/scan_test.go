// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"errors"
	"strings"
	"testing"
)

func events(t *testing.T, in string, maxLine, maxEvent int) ([]Event, error) {
	t.Helper()
	s := NewScanner(strings.NewReader(in), maxLine, maxEvent)
	var out []Event
	for s.Scan() {
		e := s.Event()
		out = append(out, Event{Name: e.Name, ID: e.ID, Data: append([]byte(nil), e.Data...)})
	}
	return out, s.Err()
}

func TestScanNamedEvents(t *testing.T) {
	got, err := events(t, "event: message_start\ndata: {\"a\":1}\n\nevent: message_stop\ndata: {}\n\n", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Name != "message_start" || string(got[0].Data) != `{"a":1}` {
		t.Fatalf("events = %+v", got)
	}
	if got[1].Name != "message_stop" {
		t.Fatalf("second event = %+v", got[1])
	}
}

func TestScanAcceptsDataWithoutSpace(t *testing.T) {
	got, err := events(t, "data:{\"x\":1}\n\n", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || string(got[0].Data) != `{"x":1}` {
		t.Fatalf("events = %+v", got)
	}
}

func TestScanDeliversFinalEventWithoutBlankLine(t *testing.T) {
	got, err := events(t, "data: {\"last\":true}\n", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || string(got[0].Data) != `{"last":true}` {
		t.Fatalf("events = %+v: the last answer of a stream must not be discarded", got)
	}
}

func TestScanJoinsMultilineData(t *testing.T) {
	got, err := events(t, "data: one\ndata: two\n\n", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if string(got[0].Data) != "one\ntwo" {
		t.Fatalf("data = %q", got[0].Data)
	}
}

func TestScanIgnoresComments(t *testing.T) {
	got, err := events(t, ": ping\ndata: {}\n\n", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("events = %+v", got)
	}
}

func TestScanReportsOverlongLine(t *testing.T) {
	_, err := events(t, "data: "+strings.Repeat("x", 300)+"\n\n", 64, 0)
	if !errors.Is(err, ErrLineTooLong) {
		t.Fatalf("err = %v: an oversized frame must be an error, not a skip", err)
	}
}

func TestScanReportsOverlongEvent(t *testing.T) {
	in := "data: " + strings.Repeat("y", 100) + "\ndata: " + strings.Repeat("z", 100) + "\n\n"
	_, err := events(t, in, 0, 120)
	if !errors.Is(err, ErrEventTooLong) {
		t.Fatalf("err = %v", err)
	}
}

func TestScanHandlesCarriageReturns(t *testing.T) {
	got, err := events(t, "event: done\r\ndata: {}\r\n\r\n", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Name != "done" || string(got[0].Data) != "{}" {
		t.Fatalf("events = %+v", got)
	}
}

func TestLinesSkipsBlanksAndKeepsLast(t *testing.T) {
	l := NewLines(strings.NewReader("{\"a\":1}\n\n{\"b\":2}"), 0)
	var got []string
	for l.Scan() {
		got = append(got, string(l.Bytes()))
	}
	if l.Err() != nil {
		t.Fatal(l.Err())
	}
	if len(got) != 2 || got[1] != `{"b":2}` {
		t.Fatalf("lines = %v", got)
	}
}

func TestLinesReportsOverlongLine(t *testing.T) {
	l := NewLines(strings.NewReader(strings.Repeat("q", 500)+"\n"), 64)
	if l.Scan() {
		t.Fatal("an oversized line must not be delivered")
	}
	if !errors.Is(l.Err(), ErrLineTooLong) {
		t.Fatalf("err = %v", l.Err())
	}
}

func TestObjectArgsShapesWhatWillNotFit(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{`{"a":1}`, `{"a":1}`},
		{``, `{}`},
		{`null`, `{}`},
		{`not json`, `{}`},
		{`"hello"`, `{"_raw":"hello"}`},
		{`[1,2]`, `{"_raw":[1,2]}`},
		{`7`, `{"_raw":7}`},
	} {
		if got := string(ObjectArgs([]byte(tc.in))); got != tc.want {
			t.Errorf("ObjectArgs(%q) = %s, want %s", tc.in, got, tc.want)
		}
	}
}

func TestTheEventNameDoesNotCarryIntoTheNextFrame(t *testing.T) {
	sc := NewScanner(strings.NewReader(
		"event: message_start\ndata: {\"a\":1}\n\n"+
			"data: {\"b\":2}\n\n"+
			"event: message_stop\ndata: {\"c\":3}\n\n"), 0, 0)

	var got []Event
	for sc.Scan() {
		got = append(got, sc.Event())
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("events = %d", len(got))
	}
	if got[0].Name != "message_start" {
		t.Errorf("first name = %q", got[0].Name)
	}
	if got[1].Name != "" {
		t.Errorf("second name = %q: it inherited the name of the frame before it", got[1].Name)
	}
	if got[2].Name != "message_stop" {
		t.Errorf("third name = %q", got[2].Name)
	}
}

func TestTheLastEventIDCarries(t *testing.T) {
	sc := NewScanner(strings.NewReader("id: 7\ndata: a\n\ndata: b\n\n"), 0, 0)
	var got []Event
	for sc.Scan() {
		got = append(got, sc.Event())
	}
	if len(got) != 2 {
		t.Fatalf("events = %d", len(got))
	}
	if got[0].ID != "7" || got[1].ID != "7" {
		t.Fatalf("ids = %q, %q", got[0].ID, got[1].ID)
	}
}
