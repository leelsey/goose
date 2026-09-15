// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"bytes"
	"strings"
	"testing"
)

func FuzzScanner(f *testing.F) {
	for _, seed := range []string{
		"data: {\"a\":1}\n\n",
		"event: x\ndata: {\"a\":1}\n\n",
		"data:{\"a\":1}\n\ndata: [DONE]\n\n",
		": comment\ndata: 1\n\n",
		"data: a\ndata: b\n\n",
		"id: 7\nevent: e\ndata: v\n\n",
		"data: no terminator",
		"\n\n\n",
		"",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, body string) {
		sc := NewScanner(strings.NewReader(body), 1<<16, 1<<16)
		for n := 0; sc.Scan(); n++ {
			_ = sc.Event()
			if n > len(body)+16 {
				t.Fatalf("the scanner yielded more events than the body has bytes: %q", body)
			}
		}
		_ = sc.Err()
	})
}

func FuzzLines(f *testing.F) {
	for _, seed := range []string{
		"{\"a\":1}\n{\"b\":2}\n", "{\"a\":1}", "\n\n", "", "a\rb\nc",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, body string) {
		l := NewLines(bytes.NewReader([]byte(body)), 1<<16)
		for n := 0; l.Scan(); n++ {
			_ = l.Bytes()
			if n > len(body)+16 {
				t.Fatalf("Lines yielded more lines than the body has bytes: %q", body)
			}
		}
		_ = l.Err()
	})
}
