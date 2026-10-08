// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// An item this wire does not model reaches a client on the same wire as the
// backend wrote it, streamed or not. An approval request that never arrives
// leaves the agent waiting on it for good.
func TestAnUnmodelledItemReachesTheClientWhole(t *testing.T) {
	approval := `{"type":"mcp_approval_request","id":"mcpr_1","server_label":"docs","name":"read","arguments":"{}"}`
	part := &llm.Raw{Vendor: Name, Type: "mcp_approval_request", JSON: json.RawMessage(approval)}
	whole := func(t *testing.T, where string, v any) {
		t.Helper()
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"server_label":"docs"`) {
			t.Errorf("%s = %s: the item lost what this wire does not model", where, b)
		}
	}

	resp := &llm.Response{ID: "r", Model: "m", Choices: []llm.Choice{{
		Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{part}}, Stop: llm.StopEnd,
	}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(out.Body, &body); err != nil {
		t.Fatal(err)
	}
	if len(body.Output) != 1 {
		t.Fatalf("output = %s", out.Body)
	}
	whole(t, "unstreamed item", body.Output[0])

	var done, final any
	for _, e := range events(t,
		&llm.Frame{Delta: []llm.Part{part}},
		&llm.Frame{Stop: llm.StopEnd, TurnEnd: true},
	) {
		switch e.name {
		case "response.output_item.done":
			done = e.body["item"]
		case "response.completed":
			output, _ := e.body["response"].(map[string]any)["output"].([]any)
			if len(output) != 1 {
				t.Fatalf("closing output = %v", output)
			}
			final = output[0]
		}
	}
	if done == nil {
		t.Fatal("the stream never delivered the item")
	}
	whole(t, "streamed item", done)
	whole(t, "closing object", final)
}

// A freeform tool has no parameters, so read as a function it reaches the
// model as one that takes nothing and its grammar is gone.
func TestAFreeformToolIsNotReadAsAFunction(t *testing.T) {
	r, _ := decode(t, `{"model":"gpt-5","input":"hi","tools":[
		{"type":"function","name":"shell","parameters":{"type":"object"}},
		{"type":"custom","name":"apply_patch","description":"Apply a patch",
		 "format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}],
		"tool_choice":{"type":"custom","name":"apply_patch"}}`)
	if len(r.Tools) != 1 || r.Tools[0].Name != "shell" {
		t.Fatalf("tools = %+v: only the function is a function", r.Tools)
	}
	if r.ToolChoice != nil || string(r.Extra["tool_choice"]) != `{"type":"custom","name":"apply_patch"}` {
		t.Fatalf("choice = %+v, carried %s: it names the freeform tool, not a function", r.ToolChoice, r.Extra["tool_choice"])
	}
	if r.ExtraFrom != Name {
		t.Fatalf("extra from = %q", r.ExtraFrom)
	}
	var carried []struct {
		Type   string `json:"type"`
		Name   string `json:"name"`
		Format struct {
			Definition string `json:"definition"`
		} `json:"format"`
	}
	if err := json.Unmarshal(r.Extra["tools"], &carried); err != nil {
		t.Fatalf("carried = %s: %v", r.Extra["tools"], err)
	}
	if len(carried) != 1 || carried[0].Type != "custom" || carried[0].Name != "apply_patch" ||
		carried[0].Format.Definition != "start: /.+/" {
		t.Fatalf("carried = %s: the tool must travel as it was written", r.Extra["tools"])
	}
}

// A backend is sent the functions, then the tool servers, then the tools
// carried whole, so a client that mixed them has its order changed and is told.
func TestAToolOrderThatCannotBeKeptIsReported(t *testing.T) {
	const (
		fn     = `{"type":"function","name":"f"}`
		custom = `{"type":"custom","name":"apply_patch"}`
		server = `{"type":"mcp","server_label":"docs","server_url":"https://tools.test/mcp"}`
	)
	for tools, reordered := range map[string]bool{
		fn + "," + custom:                false,
		fn + "," + server + "," + custom: false,
		custom + "," + fn:                true,
		fn + "," + custom + "," + fn:     true,
		server + "," + fn:                true,
	} {
		_, _, diags, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Path: "/v1/responses",
			Body: []byte(`{"model":"gpt-5","input":"hi","tools":[` + tools + `]}`)})
		if err != nil {
			t.Fatal(err)
		}
		said := false
		for _, d := range diags {
			said = said || d.Code == llm.DropToolDefinition && d.Field == "tools"
		}
		if said != reordered {
			t.Errorf("tools [%s]: diagnostics = %+v: a changed order is reported, and only then", tools, diags)
		}
	}
}

// A choice over tools that all travel as written goes with them. Read as a
// mode it reaches the backend with no function to govern and is dropped, so
// "none" lets the model call the tool and "required" no longer forces it.
func TestAChoiceOverCarriedToolsTravelsWithThem(t *testing.T) {
	for _, choice := range []string{`"required"`, `"none"`, `"auto"`} {
		r, _ := decode(t, `{"model":"gpt-5","input":"hi",
			"tools":[{"type":"custom","name":"apply_patch"}],"tool_choice":`+choice+`}`)
		if r.ToolChoice != nil || string(r.Extra["tool_choice"]) != choice {
			t.Errorf("%s: choice = %+v, carried %s", choice, r.ToolChoice, r.Extra["tool_choice"])
		}
	}
	r, _ := decode(t, `{"model":"gpt-5","input":"hi","tools":[
		{"type":"function","name":"shell"},{"type":"custom","name":"apply_patch"}],"tool_choice":"required"}`)
	if r.ToolChoice == nil || r.ToolChoice.Mode != llm.ToolRequired || r.Extra["tool_choice"] != nil {
		t.Fatalf("choice = %+v, carried %s: a function is there for it to govern", r.ToolChoice, r.Extra["tool_choice"])
	}
}

// A file id names an image only the backend holding the file can resolve. As
// an image it carries neither bytes nor an address, and is sent as nothing.
func TestAnImageNamedByItsFileIsNotAnEmptyImage(t *testing.T) {
	body := `{"model":"m","input":[{"role":"user","content":[
		{"type":"input_image","file_id":"file-abc","detail":"high"},
		{"type":"input_image","image_url":"https://img.test/a.png"}]}]}`
	r, _, diags, err := New(Options{}).Decode(&wire.Inbound{Method: "POST", Path: "/v1/responses", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	parts := r.Messages[0].Content
	if len(parts) != 2 {
		t.Fatalf("parts = %+v", parts)
	}
	raw, ok := parts[0].(*llm.Raw)
	if !ok || raw.Vendor != Name || raw.Type != "input_image" || !strings.Contains(string(raw.JSON), `"file-abc"`) {
		t.Fatalf("file image = %#v: it must travel as the reference it is", parts[0])
	}
	if !diags.Has(llm.DropUnknownBlock) {
		t.Errorf("diagnostics = %v: carrying it verbatim is said", diags.Codes())
	}
	if img, ok := parts[1].(*llm.Image); !ok || img.URL != "https://img.test/a.png" {
		t.Fatalf("addressed image = %#v", parts[1])
	}
}
