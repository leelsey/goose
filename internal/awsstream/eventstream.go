// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package awsstream decodes the binary event framing AWS streams with.
package awsstream

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

const (
	preludeLen = 12
	crcLen     = 4
	maxMessage = 32 << 20
)

var (
	ErrPreludeCRC = errors.New("awsstream: prelude checksum mismatch")
	ErrMessageCRC = errors.New("awsstream: message checksum mismatch")
	ErrTooLarge   = errors.New("awsstream: message exceeds limit")
)

type Message struct {
	Headers map[string]string
	Payload []byte
}

func (m Message) Type() string {
	if t := m.Headers[":event-type"]; t != "" {
		return t
	}
	return m.Headers[":exception-type"]
}

func (m Message) Kind() string { return m.Headers[":message-type"] }

type Reader struct {
	r    io.Reader
	max  int
	crc  *crc32.Table
	last Message
	err  error
	done bool
}

// Both checksums in the framing are CRC-32 over the IEEE polynomial, the one
// zlib uses. Castagnoli is the other CRC-32 in the standard library, and a
// reader built on it refuses every frame AWS sends.
func NewReader(r io.Reader, max int) *Reader {
	if max <= 0 {
		max = maxMessage
	}
	return &Reader{r: r, max: max, crc: crc32.IEEETable}
}

func (r *Reader) Message() Message { return r.last }

func (r *Reader) Err() error { return r.err }

func (r *Reader) Next() bool {
	if r.done {
		return false
	}
	prelude := make([]byte, preludeLen)
	if _, err := io.ReadFull(r.r, prelude); err != nil {
		r.done = true
		// ReadFull answers io.EOF only when it read nothing, which is where a
		// clean end falls: between messages. Anything else lost bytes, and a
		// stream that lost its last message is not a stream that finished.
		if errors.Is(err, io.EOF) {
			return false
		}
		if errors.Is(err, io.ErrUnexpectedEOF) {
			r.err = fmt.Errorf("awsstream: truncated message: %w", err)
			return false
		}
		r.err = err
		return false
	}
	total := binary.BigEndian.Uint32(prelude[0:4])
	headerLen := binary.BigEndian.Uint32(prelude[4:8])
	wantCRC := binary.BigEndian.Uint32(prelude[8:12])
	if crc32.Checksum(prelude[:8], r.crc) != wantCRC {
		r.fail(ErrPreludeCRC)
		return false
	}
	if total < preludeLen+crcLen || uint64(total) > uint64(r.max) {
		r.fail(fmt.Errorf("%w: %d bytes", ErrTooLarge, total))
		return false
	}

	rest := make([]byte, int(total)-preludeLen)
	if _, err := io.ReadFull(r.r, rest); err != nil {
		r.fail(fmt.Errorf("awsstream: truncated message: %w", err))
		return false
	}
	bodyEnd := len(rest) - crcLen
	if uint64(headerLen) > uint64(bodyEnd) {
		r.fail(errors.New("awsstream: header length exceeds message"))
		return false
	}
	msgCRC := binary.BigEndian.Uint32(rest[bodyEnd:])
	whole := append(append([]byte{}, prelude...), rest[:bodyEnd]...)
	if crc32.Checksum(whole, r.crc) != msgCRC {
		r.fail(ErrMessageCRC)
		return false
	}

	headers, err := parseHeaders(rest[:headerLen])
	if err != nil {
		r.fail(err)
		return false
	}
	r.last = Message{Headers: headers, Payload: rest[headerLen:bodyEnd]}
	return true
}

func (r *Reader) fail(err error) {
	r.done = true
	r.err = err
}

func parseHeaders(b []byte) (map[string]string, error) {
	out := map[string]string{}
	for len(b) > 0 {
		if len(b) < 2 {
			return nil, errors.New("awsstream: truncated header")
		}
		nameLen := int(b[0])
		b = b[1:]
		if len(b) < nameLen+1 {
			return nil, errors.New("awsstream: truncated header name")
		}
		name := string(b[:nameLen])
		valueType := b[nameLen]
		b = b[nameLen+1:]

		switch valueType {
		case 0, 1:
			out[name] = map[byte]string{0: "true", 1: "false"}[valueType]
		case 2:
			if len(b) < 1 {
				return nil, errors.New("awsstream: truncated byte header")
			}
			b = b[1:]
		case 3:
			if len(b) < 2 {
				return nil, errors.New("awsstream: truncated short header")
			}
			b = b[2:]
		case 4:
			if len(b) < 4 {
				return nil, errors.New("awsstream: truncated integer header")
			}
			b = b[4:]
		case 5, 8:
			if len(b) < 8 {
				return nil, errors.New("awsstream: truncated long header")
			}
			b = b[8:]
		case 6, 7:
			if len(b) < 2 {
				return nil, errors.New("awsstream: truncated string header")
			}
			n := int(binary.BigEndian.Uint16(b[:2]))
			b = b[2:]
			if len(b) < n {
				return nil, errors.New("awsstream: truncated string value")
			}
			if valueType == 7 {
				out[name] = string(b[:n])
			}
			b = b[n:]
		case 9:
			if len(b) < 16 {
				return nil, errors.New("awsstream: truncated uuid header")
			}
			b = b[16:]
		default:
			return nil, fmt.Errorf("awsstream: unknown header type %d", valueType)
		}
	}
	return out, nil
}

func Encode(headers map[string]string, payload []byte) []byte {
	var hb []byte
	for k, v := range headers {
		hb = append(hb, byte(len(k)))
		hb = append(hb, k...)
		hb = append(hb, 7)
		hb = binary.BigEndian.AppendUint16(hb, uint16(len(v)))
		hb = append(hb, v...)
	}
	total := uint32(preludeLen + len(hb) + len(payload) + 4)
	table := crc32.IEEETable

	msg := make([]byte, 0, total)
	msg = binary.BigEndian.AppendUint32(msg, total)
	msg = binary.BigEndian.AppendUint32(msg, uint32(len(hb)))
	msg = binary.BigEndian.AppendUint32(msg, crc32.Checksum(msg[:8], table))
	msg = append(msg, hb...)
	msg = append(msg, payload...)
	return binary.BigEndian.AppendUint32(msg, crc32.Checksum(msg, table))
}
