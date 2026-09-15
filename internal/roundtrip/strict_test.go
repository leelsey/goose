// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/llm"
)

func TestAStrictToolReachesTheBackendThatHasTheField(t *testing.T) {
	front, seen := proxy(t)
	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(
		`{"model":"m","messages":[{"role":"user","content":"hi"}],
		 "tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(*seen), `"strict":true`) {
		t.Fatalf("the guarantee did not travel:\n%s", *seen)
	}
}

func TestABackendWithNoStrictnessSaysSo(t *testing.T) {
	front, _ := crossProxy(t)
	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json", strings.NewReader(
		`{"model":"m","messages":[{"role":"user","content":"hi"}],
		 "tools":[{"type":"function","function":{"name":"f","strict":true,"parameters":{"type":"object"}}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	_, _ = io.ReadAll(resp.Body)
	if !strings.Contains(resp.Header.Get(httpx.DiagnosticsHeader), llm.DropToolDefinition) {
		t.Fatalf("diagnostics = %q", resp.Header.Get(httpx.DiagnosticsHeader))
	}
}
