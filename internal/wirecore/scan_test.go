// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"errors"
	"io"
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

// An event that outgrew the ceiling is not an event: half of one, handed over
// as though it were whole, lets whoever is upstream choose where the frame ends
// and so what it says.
func TestAnOverlongEventIsNotDelivered(t *testing.T) {
	body := "event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}` + "\n" +
		"data: " + strings.Repeat("A", 4096) + "\n\n"
	sc := NewScanner(strings.NewReader(body), 0, 200)
	for sc.Scan() {
		t.Fatalf("an event past the ceiling was delivered: name=%q data=%q", sc.Event().Name, sc.Event().Data)
	}
	if !errors.Is(sc.Err(), ErrEventTooLong) {
		t.Fatalf("err = %v, want %v", sc.Err(), ErrEventTooLong)
	}
}

// Reading on after the ceiling trips spends exactly the budget the ceiling
// exists to cap.
func TestAnOverlongEventStopsReading(t *testing.T) {
	r := &counter{Reader: strings.NewReader(strings.Repeat("data: AAAAAAAA\n", 1<<16) + "\n")}
	sc := NewScanner(r, 1<<20, 64)
	for sc.Scan() {
	}
	if !errors.Is(sc.Err(), ErrEventTooLong) {
		t.Fatalf("err = %v", sc.Err())
	}
	if r.n > 1<<16 {
		t.Fatalf("read %d bytes for a 64-byte ceiling", r.n)
	}
}

type counter struct {
	io.Reader
	n int
}

func (c *counter) Read(p []byte) (int, error) {
	n, err := c.Reader.Read(p)
	c.n += n
	return n, err
}

// Both readers mean the same thing by a line: the terminator is not part of it.
func TestBothReadersAgreeOnWhatALineIs(t *testing.T) {
	line := strings.Repeat("a", 64)
	sc := NewScanner(strings.NewReader("data: "+line[:58]+"\n\n"), 64, 1<<20)
	for sc.Scan() {
	}
	if sc.Err() != nil {
		t.Errorf("the event scanner refused a line of exactly the ceiling: %v", sc.Err())
	}
	l := NewLines(strings.NewReader(line+"\n"), 64)
	for l.Scan() {
	}
	if l.Err() != nil {
		t.Errorf("the line reader refused a line of exactly the ceiling: %v", l.Err())
	}
}

// A CR and an LF that fall either side of a read boundary are still one
// terminator, and the line they end is still the length it always was.
func TestATerminatorSplitAcrossReadsIsStillOneTerminator(t *testing.T) {
	for _, size := range []int{1<<16 - 1, 1 << 16, 1<<16 + 1} {
		body := strings.Repeat("a", size) + "\r\n"
		l := NewLines(strings.NewReader(body), size)
		for l.Scan() {
		}
		if l.Err() != nil {
			t.Errorf("a %d-byte line was refused by a %d-byte ceiling: %v", size, size, l.Err())
		}
		l = NewLines(strings.NewReader(body), size-1)
		for l.Scan() {
		}
		if l.Err() == nil {
			t.Errorf("a %d-byte line passed a %d-byte ceiling", size, size-1)
		}
	}
}
