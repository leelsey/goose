// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/ollamaapi"
)

// A backend that takes no tool definitions is a backend the model cannot reach
// for one through, so there is nothing for the format to fail to report. The
// tools are dropped and said to be dropped, and the answer is served.
func TestAToolNoBackendWillTakeIsNotAReasonToRefuse(t *testing.T) {
	p := &stub{}
	c, err := goose.New(
		goose.WithProvider(p),
		goose.WithTools("mcp", []llm.Tool{{Name: "search", Schema: []byte(`{"type":"object"}`)}}),
		goose.WithRoutes(route.Rule{
			Name: "tooled", Match: route.Always(),
			Chain: []route.Target{{Provider: "stub"}}, Tools: []string{"mcp"},
		}),
	)
	if err != nil {
		t.Fatal(err)
	}
	s, err := New(c, Options{Codecs: []wire.Codec{ollamaapi.NewGenerate(ollamaapi.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	defer srv.Close()

	resp := post(t, srv, "/api/generate", `{"model":"m","prompt":"hi","stream":false}`, nil)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: a plain answer was available: %s", resp.StatusCode, body)
	}
	if !strings.Contains(resp.Header.Get(DiagnosticsHeader), llm.DropToolDefinition) {
		t.Fatalf("diagnostics = %q: the caller offered tools and got none", resp.Header.Get(DiagnosticsHeader))
	}
}
