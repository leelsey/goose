// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
)

// An event is almost always one data line, so a line ceiling below the event's
// refuses what the event ceiling was there to allow: a streamed image arrives
// as one line of several megabytes.
const (
	DefaultMaxLine  = DefaultMaxEvent
	DefaultMaxEvent = 8 << 20
)

var (
	ErrLineTooLong  = errors.New("wire: line exceeds limit")
	ErrEventTooLong = errors.New("wire: event exceeds limit")
)

type Event struct {
	Name string
	ID   string
	Data []byte
}

type Scanner struct {
	r        *bufio.Reader
	maxLine  int
	maxEvent int

	ev         Event
	pending    bool
	dispatched bool
	data       []byte
	err        error
	done       bool
}

func NewScanner(r io.Reader, maxLine, maxEvent int) *Scanner {
	if maxLine <= 0 {
		maxLine = DefaultMaxLine
	}
	if maxEvent <= 0 {
		maxEvent = DefaultMaxEvent
	}
	return &Scanner{r: bufio.NewReaderSize(r, 64<<10), maxLine: maxLine, maxEvent: maxEvent}
}

func (s *Scanner) Event() Event { return s.ev }

func (s *Scanner) Err() error { return s.err }

func (s *Scanner) Scan() bool {
	if s.done {
		return false
	}
	if s.dispatched {
		s.ev.Name, s.dispatched = "", false
	}
	for {
		line, err := s.readLine()
		if err != nil && !errors.Is(err, io.EOF) {
			s.err = err
			s.done = true
			return false
		}
		eof := errors.Is(err, io.EOF)
		line = trimEOL(line)

		if len(line) == 0 && !eof {
			if s.dispatch() {
				return true
			}
			continue
		}
		if len(line) > 0 {
			s.field(line)
			if s.done {
				// The ceiling tripped part way through an event. What has been
				// read is half of one, and handing that over would let whoever
				// is upstream choose where the frame ends and so what it says.
				s.pending, s.data = false, nil
				return false
			}
		}
		if eof {
			s.done = true
			if s.pending {
				return s.dispatch()
			}
			return false
		}
	}
}

func (s *Scanner) field(line []byte) {
	if line[0] == ':' {
		return
	}
	name, value := line, []byte(nil)
	if i := bytes.IndexByte(line, ':'); i >= 0 {
		name, value = line[:i], line[i+1:]
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}
	}
	switch string(name) {
	case "event":
		s.ev.Name = string(value)
		s.pending = true
	case "id":
		s.ev.ID = string(value)
		s.pending = true
	case "data":
		if len(s.data)+len(value)+1 > s.maxEvent {
			s.err = fmt.Errorf("%w: %d bytes", ErrEventTooLong, len(s.data)+len(value))
			s.done = true
			return
		}
		s.data = append(s.data, value...)
		s.data = append(s.data, '\n')
		s.pending = true
	}
}

func (s *Scanner) dispatch() bool {
	if !s.pending {
		return false
	}
	data := s.data
	if n := len(data); n > 0 && data[n-1] == '\n' {
		data = data[:n-1]
	}
	s.ev.Data = data
	s.data = nil
	s.pending = false
	s.dispatched = true
	return true
}

func (s *Scanner) readLine() ([]byte, error) {
	var buf []byte
	for {
		chunk, err := s.r.ReadSlice('\n')
		buf = append(buf, chunk...)
		// The terminator is not part of the line, so a line of exactly the
		// ceiling is a line that fits — including when its CR and LF fell
		// either side of a read boundary, which is why this is measured on the
		// whole of what has been read rather than on the piece just added.
		if n := len(trimEOL(buf)); n > s.maxLine {
			return nil, fmt.Errorf("%w: %d bytes", ErrLineTooLong, n)
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue
		}
		return buf, err
	}
}

func trimEOL(b []byte) []byte {
	b = bytes.TrimSuffix(b, []byte("\n"))
	return bytes.TrimSuffix(b, []byte("\r"))
}

type Lines struct {
	r       *bufio.Reader
	maxLine int
	line    []byte
	err     error
	done    bool
}

func NewLines(r io.Reader, maxLine int) *Lines {
	if maxLine <= 0 {
		maxLine = DefaultMaxLine
	}
	return &Lines{r: bufio.NewReaderSize(r, 64<<10), maxLine: maxLine}
}

func (l *Lines) Bytes() []byte { return l.line }

func (l *Lines) Err() error { return l.err }

func (l *Lines) Scan() bool {
	if l.done {
		return false
	}
	for {
		var buf []byte
		var err error
		for {
			var chunk []byte
			chunk, err = l.r.ReadSlice('\n')
			buf = append(buf, chunk...)
			// The terminator is not part of the line, the same as it is not for
			// the event scanner beside this.
			if n := len(trimEOL(buf)); n > l.maxLine {
				l.err = fmt.Errorf("%w: %d bytes", ErrLineTooLong, n)
				l.done = true
				return false
			}
			if errors.Is(err, bufio.ErrBufferFull) {
				continue
			}
			break
		}
		if err != nil && !errors.Is(err, io.EOF) {
			l.err = err
			l.done = true
			return false
		}
		eof := errors.Is(err, io.EOF)
		line := trimEOL(buf)
		if len(line) > 0 {
			l.line = line
			l.done = eof
			return true
		}
		if eof {
			l.done = true
			return false
		}
	}
}
