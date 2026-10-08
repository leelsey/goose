// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package awsstream

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"testing"
)

func frame(headers, payload []byte) []byte {
	total := uint32(preludeLen + len(headers) + len(payload) + 4)
	prelude := make([]byte, preludeLen)
	binary.BigEndian.PutUint32(prelude[0:4], total)
	binary.BigEndian.PutUint32(prelude[4:8], uint32(len(headers)))
	table := crc32.IEEETable
	binary.BigEndian.PutUint32(prelude[8:12], crc32.Checksum(prelude[:8], table))

	body := append(append([]byte{}, prelude...), headers...)
	body = append(body, payload...)
	sum := make([]byte, 4)
	binary.BigEndian.PutUint32(sum, crc32.Checksum(body, table))
	return append(body, sum...)
}

func FuzzReader(f *testing.F) {
	hdr := []byte{11, ':', 'e', 'v', 'e', 'n', 't', '-', 't', 'y', 'p', 'e', 7, 0, 5, 'c', 'h', 'u', 'n', 'k'}
	f.Add(frame(hdr, []byte(`{"delta":{"text":"hi"}}`)))
	f.Add(frame(nil, nil))
	f.Add(frame(hdr, nil))
	f.Add([]byte{})
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, body []byte) {
		r := NewReader(bytes.NewReader(body), 1<<20)
		for n := 0; r.Next() && n < 64; n++ {
			m := r.Message()
			_, _, _ = m.Type(), m.Kind(), len(m.Payload)
		}
		_ = r.Err()
	})
}
