// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wire

import (
	"bytes"
	"net/http"
)

type SSE struct {
	w    Writer
	buf  bytes.Buffer
	open bool
	err  error
}

func NewSSE(w Writer) *SSE { return &SSE{w: w} }

func (s *SSE) Open(header http.Header) error {
	if s.open {
		return s.err
	}
	s.open = true
	h := header
	if h == nil {
		h = http.Header{}
	}
	if h.Get("Content-Type") == "" {
		h.Set("Content-Type", "text/event-stream")
	}
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	s.err = s.w.Open(http.StatusOK, h)
	return s.err
}

func (s *SSE) Send(name string, data []byte) error {
	if s.err != nil {
		return s.err
	}
	if !s.open {
		if err := s.Open(nil); err != nil {
			return err
		}
	}
	s.buf.Reset()
	if name != "" {
		s.buf.WriteString("event: ")
		s.buf.WriteString(name)
		s.buf.WriteByte('\n')
	}
	s.buf.WriteString("data: ")
	s.buf.Write(data)
	s.buf.WriteString("\n\n")
	if _, err := s.w.Write(s.buf.Bytes()); err != nil {
		s.err = err
		return err
	}
	return s.flush()
}

func (s *SSE) flush() error {
	if err := s.w.Flush(); err != nil {
		s.err = err
		return err
	}
	return nil
}

func (s *SSE) Err() error { return s.err }

type NDJSON struct {
	w    Writer
	open bool
	err  error
}

func NewNDJSON(w Writer) *NDJSON { return &NDJSON{w: w} }

func (n *NDJSON) Open(header http.Header) error {
	if n.open {
		return n.err
	}
	n.open = true
	h := header
	if h == nil {
		h = http.Header{}
	}
	if h.Get("Content-Type") == "" {
		h.Set("Content-Type", "application/x-ndjson")
	}
	n.err = n.w.Open(http.StatusOK, h)
	return n.err
}

func (n *NDJSON) Send(data []byte) error {
	if n.err != nil {
		return n.err
	}
	if !n.open {
		if err := n.Open(nil); err != nil {
			return err
		}
	}
	if _, err := n.w.Write(append(data, '\n')); err != nil {
		n.err = err
		return err
	}
	if err := n.w.Flush(); err != nil {
		n.err = err
		return err
	}
	return nil
}

func (n *NDJSON) Err() error { return n.err }

type BufferWriter struct {
	Status  int
	Head    http.Header
	Body    bytes.Buffer
	Flushes int
}

func (b *BufferWriter) Open(status int, header http.Header) error {
	b.Status = status
	b.Head = header
	return nil
}

func (b *BufferWriter) Write(p []byte) (int, error) { return b.Body.Write(p) }

func (b *BufferWriter) Flush() error { b.Flushes++; return nil }
