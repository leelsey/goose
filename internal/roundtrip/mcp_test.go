// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
)

func TestAToolServerCrossesFromOneVendorToTheOther(t *testing.T) {
	var seen []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed",
			"output":[{"type":"message","role":"assistant","status":"completed",
			"content":[{"type":"output_text","text":"ok"}]}],
			"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	p, err := openai.New(openai.Options{
		ID: "up", BaseURL: upstream.URL + "/v1", APIKey: "k", Dialect: openai.OpenAIDialect(),
	})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{anthropicmsg.New(anthropicmsg.Options{})}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	defer front.Close()

	resp, err := front.Client().Post(front.URL+"/v1/messages", "application/json", strings.NewReader(
		`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"}],
		 "mcp_servers":[{"type":"url","url":"https://tools.test/mcp","name":"docs","authorization_token":"tok"}],
		 "tools":[{"type":"mcp_toolset","mcp_server_name":"docs",
		  "default_config":{"enabled":false},"configs":{"read":{"enabled":true}}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, body)
	}

	var sent struct {
		Tools []map[string]any `json:"tools"`
	}
	if err := json.Unmarshal(seen, &sent); err != nil {
		t.Fatalf("upstream body = %s: %v", seen, err)
	}
	if len(sent.Tools) != 1 {
		t.Fatalf("upstream body = %s: the connector did not cross", seen)
	}
	connector := sent.Tools[0]
	if connector["type"] != "mcp" || connector["server_url"] != "https://tools.test/mcp" {
		t.Fatalf("connector = %+v", connector)
	}
	if connector["authorization"] != "tok" || connector["server_label"] != "docs" {
		t.Fatalf("connector = %+v", connector)
	}
	if names, ok := connector["allowed_tools"].([]any); !ok || len(names) != 1 {
		t.Fatalf("allowed = %+v", connector["allowed_tools"])
	}
	if diags := resp.Header.Get("X-Goose-Diagnostics"); strings.Contains(diags, "drop.mcp_server") {
		t.Fatalf("diagnostics = %s: nothing was lost on this crossing", diags)
	}
}
