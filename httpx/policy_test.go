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
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/usage"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openaichat"
	"github.com/leelsey/goose/wire/openairesp"
)

// held serves p through the OpenAI chat and Anthropic wires with the client
// held to pol, and keeps the ledger where the test can read it.
func held(t *testing.T, p llm.Provider, pol llm.LossPolicy, o Options) (*httptest.Server, *usage.Memory) {
	t.Helper()
	mem := usage.NewMemory()
	c, err := goose.New(goose.WithProvider(p), goose.WithLossPolicy(pol), goose.WithUsage(mem))
	if err != nil {
		t.Fatal(err)
	}
	if o.Codecs == nil {
		o.Codecs = []wire.Codec{openaichat.New(openaichat.Options{}), anthropicmsg.New(anthropicmsg.Options{})}
	}
	s, err := New(c, o)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(s.Handler())
	t.Cleanup(srv.Close)
	return srv, mem
}

func answer(t *testing.T, srv *httptest.Server, path, body string) (int, string) {
	t.Helper()
	resp := post(t, srv, path, body, nil)
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(b)
}

// searching is an Anthropic backend that answers with a server tool's call
// and result, which goose carries verbatim, and then text.
func searching(t *testing.T) llm.Provider {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[`+
				`{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"geese"}},`+
				`{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]},`+
				`{"type":"text","text":"Geese honk."}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":\"geese\"}"}}`,
			`{"type":"content_block_stop","index":0}`,
			`{"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[]}}`,
			`{"type":"content_block_stop","index":1}`,
			`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Geese honk."}}`,
			`{"type":"content_block_stop","index":2}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
			`{"type":"message_stop"}`,
		} {
			_, _ = io.WriteString(w, "data: "+ev+"\n\n")
		}
	}))
	t.Cleanup(upstream.Close)
	p, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: upstream.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// A block goose carried verbatim reaches a client on its own wire whole, and
// has lost nothing there. A client on another wire gets it dropped by the
// encoder, and strict refuses that before anything is written, streamed or not.
func TestStrictRefusesACarriedBlockTheCallersWireCannotWriteOut(t *testing.T) {
	srv, _ := held(t, searching(t), llm.LossPolicy{Strict: true}, Options{})
	for _, stream := range []string{"false", "true"} {
		status, body := answer(t, srv, "/v1/chat/completions",
			`{"model":"claude-x","stream":`+stream+`,"messages":[{"role":"user","content":"do geese honk?"}]}`)
		if status != http.StatusBadRequest || !strings.Contains(body, llm.DropUnknownBlock) || strings.Contains(body, "Geese honk.") {
			t.Fatalf("stream %s: status = %d, body = %s: the answer went out without the blocks it rests on", stream, status, body)
		}
		status, body = answer(t, srv, "/v1/messages",
			`{"model":"claude-x","max_tokens":64,"stream":`+stream+`,"messages":[{"role":"user","content":"do geese honk?"}]}`)
		if status != http.StatusOK || !strings.Contains(body, "web_search_tool_result") {
			t.Fatalf("stream %s: status = %d, body = %s: the client's own wire takes the block whole", stream, status, body)
		}
	}
}

// What the proxy's own decoding lost is a loss in the request itself, which no
// link could avoid, so the policy refuses it before anything is sent.
func TestALossInDecodingIsRefusedBeforeAnythingIsSent(t *testing.T) {
	for _, stream := range []string{"false", "true"} {
		p := &stub{}
		srv, _ := held(t, p, llm.LossPolicy{Strict: true}, Options{})
		status, body := answer(t, srv, "/v1/messages",
			`{"model":"m","max_tokens":4096,"stream":`+stream+`,"thinking":{"type":"turbo","budget_tokens":2048},`+
				`"messages":[{"role":"user","content":"hi"}]}`)
		if status != http.StatusBadRequest || !strings.Contains(body, llm.DropExtraField) {
			t.Fatalf("stream %s: status = %d, body = %s", stream, status, body)
		}
		if p.seen != nil {
			t.Fatalf("stream %s: the backend was sent a request the policy had already refused", stream)
		}
	}
}

// What the encoding of an unstreamed answer lost is held to the same policy
// as the answer, and nothing has been written yet, so the refusal replaces the
// answer. The call was served and billed, and the ledger still says so.
func TestALossInEncodingIsRefusedBeforeTheAnswerIsWritten(t *testing.T) {
	for _, pol := range []llm.LossPolicy{{Strict: true}, {Deny: []string{llm.DropImage}}} {
		p := &stub{resp: &llm.Response{
			ID: "x", Model: "served", Provider: "stub",
			Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
				&llm.Text{Text: "here it is"}, &llm.Image{MIME: "image/png", Data: []byte("png")},
			}}, Stop: llm.StopEnd}},
			Usage: llm.Usage{Prompt: 11, Completion: 2, Reported: true},
		}}
		srv, mem := held(t, p, pol, Options{})
		status, body := answer(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"draw"}]}`)
		if status != http.StatusBadRequest || !strings.Contains(body, llm.DropImage) || strings.Contains(body, "here it is") {
			t.Fatalf("policy %+v: status = %d, body = %s: the answer went out without its image", pol, status, body)
		}
		rows := mem.Records()
		if len(rows) != 1 || !rows[0].Sent || rows[0].Outcome != llm.OutcomeOK || rows[0].Usage.Prompt != 11 {
			t.Fatalf("policy %+v: rows = %+v: the call was served and billed, refused or not", pol, rows)
		}
	}
}

// A stream's encoder meets what its wire has no room for only as it writes the
// stream out. The policy refuses it there, in the wire's own failure, and
// nothing after it is written, as the unstreamed answer is refused whole.
func TestALossInEncodingAStreamIsRefusedWhereItIsMet(t *testing.T) {
	image := &llm.Image{MIME: "image/png", Data: []byte("png")}
	for _, tc := range []struct {
		path, body string
		lost       llm.Part
		code       string
	}{
		{"/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`, image, llm.DropImage},
		{"/v1/messages", `{"model":"m","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`, image, llm.DropImage},
		{"/v1/responses", `{"model":"m","stream":true,"input":"hi"}`, &llm.ToolResult{ID: "t1", Content: []llm.Part{&llm.Text{Text: "42"}}}, llm.DropToolResult},
		{"/v1beta/models/m:streamGenerateContent?alt=sse", `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`, &llm.Refusal{Text: "no"}, llm.DropRefusal},
		{"/api/chat", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, image, llm.DropImage},
		{"/api/generate", `{"model":"m","prompt":"hi"}`, image, llm.DropImage},
	} {
		t.Run(tc.path, func(t *testing.T) {
			for _, pol := range []llm.LossPolicy{{Strict: true}, {Deny: []string{tc.code}}} {
				p := &stub{frames: []*llm.Frame{
					{Role: llm.RoleAssistant, Delta: []llm.Part{tc.lost}},
					{Delta: []llm.Part{&llm.Text{Text: "what came after"}}},
					{Stop: llm.StopEnd},
					{Usage: &llm.Usage{Prompt: 11, Completion: 2, Reported: true}, TurnEnd: true},
				}}
				srv, mem := held(t, p, pol, Options{Codecs: []wire.Codec{
					openaichat.New(openaichat.Options{}), anthropicmsg.New(anthropicmsg.Options{}),
					openairesp.New(openairesp.Options{}), geminiapi.New(geminiapi.Options{}),
					ollamaapi.NewChat(ollamaapi.Options{}), ollamaapi.NewGenerate(ollamaapi.Options{}),
				}})
				_, body := answer(t, srv, tc.path, tc.body)
				if !strings.Contains(body, "translation would lose content: "+tc.code) || strings.Contains(body, "what came after") {
					t.Fatalf("policy %+v: body = %s: the stream went on past what its wire lost", pol, body)
				}
				if rows := mem.Records(); len(rows) != 1 || !rows[0].Sent {
					t.Fatalf("policy %+v: rows = %+v: the call was sent, refused or not", pol, rows)
				}
			}
		})
	}
}

// A stop reason a wire cannot spell is reported as another. A stream meets it
// last, and strict refuses it there as it refuses the unstreamed answer.
func TestAStopReasonTheCallersWireCannotSpellIsRefusedOnAStream(t *testing.T) {
	p := &stub{frames: []*llm.Frame{
		{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Text{Text: "the answer"}}},
		{Stop: llm.StopUnknown},
		{Usage: &llm.Usage{Prompt: 11, Completion: 2, Reported: true}, TurnEnd: true},
	}}
	srv, _ := held(t, p, llm.LossPolicy{Strict: true}, Options{})
	for _, c := range []struct{ path, body string }{
		{"/v1/chat/completions", `{"model":"m","stream":true,"messages":[{"role":"user","content":"hi"}]}`},
		{"/v1/messages", `{"model":"m","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}`},
	} {
		if _, body := answer(t, srv, c.path, c.body); !strings.Contains(body, "translation would lose content: "+llm.DropUnknownBlock) {
			t.Fatalf("%s: body = %s: the stop reason was misreported and the answer served as whole", c.path, body)
		}
	}
}

// Reasoning a client's wire has no field for is dropped by its encoder. On a
// stream that is known as the frame carrying it is handed over, so strict
// refuses it there rather than leaving it to vanish.
func TestStrictRefusesReasoningTheCallersWireHasNoFieldFor(t *testing.T) {
	p := &stub{
		frames: []*llm.Frame{
			{Role: llm.RoleAssistant, Delta: []llm.Part{&llm.Thinking{Text: "because"}}},
			{Delta: []llm.Part{&llm.Text{Text: "the answer"}}},
			{Stop: llm.StopEnd},
			{Usage: &llm.Usage{Prompt: 11, Completion: 2, Reported: true}, TurnEnd: true},
		},
		resp: &llm.Response{
			ID: "x", Model: "served", Provider: "stub",
			Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
				&llm.Thinking{Text: "because"}, &llm.Text{Text: "the answer"},
			}}, Stop: llm.StopEnd}},
			Usage: llm.Usage{Prompt: 11, Completion: 2, Reported: true},
		},
	}
	srv, _ := held(t, p, llm.LossPolicy{Strict: true}, Options{})
	for _, stream := range []string{"false", "true"} {
		status, body := answer(t, srv, "/v1/chat/completions",
			`{"model":"m","stream":`+stream+`,"messages":[{"role":"user","content":"why?"}]}`)
		if status != http.StatusBadRequest || !strings.Contains(body, llm.DropThinking) || strings.Contains(body, "the answer") {
			t.Fatalf("stream %s: status = %d, body = %s: the reasoning was dropped and the rest served as whole", stream, status, body)
		}
	}
}

// A stream that brings no content ends while it is primed and is booked there.
// What the proxy's decoding noted of the request is on its row all the same.
func TestAStreamWithNoContentKeepsTheRequestsNotesOnItsRow(t *testing.T) {
	p := &stub{tools: true, frames: []*llm.Frame{
		{Role: llm.RoleAssistant},
		{Stop: llm.StopEnd},
		{Usage: &llm.Usage{Prompt: 11, Reported: true}, TurnEnd: true},
	}}
	srv, mem := held(t, p, llm.LossPolicy{}, Options{})
	status, body := answer(t, srv, "/v1/chat/completions", `{"model":"m","stream":true,`+
		`"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],`+
		`"functions":[{"name":"g","parameters":{"type":"object"}}],`+
		`"messages":[{"role":"user","content":"hi"}]}`)
	if status != http.StatusOK {
		t.Fatalf("status = %d, body = %s", status, body)
	}
	rows := mem.Records()
	if len(rows) != 1 || !llm.Diagnostics(rows[0].Diags).Has(llm.DropToolDefinition) {
		t.Fatalf("rows = %+v: the request's own notes are missing from the row", rows)
	}
}

// Redaction keeps a backend's words from the caller. A refusal goose made
// itself, before sending anything, quotes nobody, and the caller needs its
// code and reason to put the request right.
func TestGoosesOwnRefusalKeepsItsWordsWhenUpstreamErrorsAreRedacted(t *testing.T) {
	p := &stub{}
	srv, _ := held(t, p, llm.LossPolicy{Strict: true}, Options{RedactUpstream: true})
	status, body := answer(t, srv, "/v1/messages", `{"model":"m","max_tokens":64,"messages":[`+
		`{"role":"user","content":"hi"},`+
		`{"role":"assistant","content":[{"type":"thinking","thinking":"because","signature":"sig"},{"type":"text","text":"so"}]},`+
		`{"role":"user","content":"go on"}]}`)
	if status != http.StatusBadRequest || !strings.Contains(body, "translation would lose content") || !strings.Contains(body, llm.DropSignature) {
		t.Fatalf("status = %d, body = %s: goose's own refusal was dressed up as the upstream failing", status, body)
	}
	if p.seen != nil {
		t.Fatal("the backend was sent a request the policy had refused")
	}
}

// A loss refused in the answer is goose's refusal as much as one refused before
// sending. The request went and was billed, and the caller is told so, but in
// goose's own words, which quote no backend and survive redaction.
func TestALossRefusedInTheAnswerKeepsItsWordsWhenUpstreamErrorsAreRedacted(t *testing.T) {
	srv, _ := held(t, searching(t), llm.LossPolicy{Strict: true}, Options{RedactUpstream: true})
	status, body := answer(t, srv, "/v1/chat/completions",
		`{"model":"claude-x","messages":[{"role":"user","content":"do geese honk?"}]}`)
	if status != http.StatusBadRequest || !strings.Contains(body, "translation would lose content: "+llm.DropUnknownBlock) {
		t.Fatalf("status = %d, body = %s: the answer's refusal was dressed up as the upstream failing", status, body)
	}

	p := &stub{resp: &llm.Response{
		ID: "x", Model: "served", Provider: "stub",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Text{Text: "here it is"}, &llm.Image{MIME: "image/png", Data: []byte("png")},
		}}, Stop: llm.StopEnd}},
		Usage: llm.Usage{Prompt: 11, Completion: 2, Reported: true},
	}}
	srv, _ = held(t, p, llm.LossPolicy{Strict: true}, Options{RedactUpstream: true})
	status, body = answer(t, srv, "/v1/chat/completions", `{"model":"m","messages":[{"role":"user","content":"draw"}]}`)
	if status != http.StatusBadRequest || !strings.Contains(body, "translation would lose content: "+llm.DropImage) {
		t.Fatalf("status = %d, body = %s: the encoding's refusal was dressed up as the upstream failing", status, body)
	}
}

// A refusal is the answer that most needs saying why, so the codes that caused
// it go out on the diagnostics header whether the refusal came before the
// answer was written or before a stream was handed over.
func TestARefusalCarriesTheCodesThatCausedIt(t *testing.T) {
	image := &llm.Image{MIME: "image/png", Data: []byte("png")}
	whole := &stub{resp: &llm.Response{
		ID: "x", Model: "served", Provider: "stub",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Text{Text: "here it is"}, image,
		}}, Stop: llm.StopEnd}},
		Usage: llm.Usage{Prompt: 11, Completion: 2, Reported: true},
	}}
	streamed := &stub{frames: []*llm.Frame{
		{Delta: []llm.Part{image}},
		{Delta: []llm.Part{&llm.Text{Text: "here it is"}}, Stop: llm.StopEnd},
	}}
	for _, tc := range []struct {
		name string
		p    *stub
		body string
	}{
		{"unstreamed", whole, `{"model":"m","messages":[{"role":"user","content":"draw"}]}`},
		{"stream refused at hand-over", streamed, `{"model":"m","stream":true,"messages":[{"role":"user","content":"draw"}]}`},
	} {
		srv, _ := held(t, tc.p, llm.LossPolicy{Strict: true}, Options{})
		resp := post(t, srv, "/v1/chat/completions", tc.body, nil)
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusBadRequest || !strings.Contains(resp.Header.Get(DiagnosticsHeader), llm.DropImage) {
			t.Fatalf("%s: status = %d, %s = %q: the refusal does not say what caused it",
				tc.name, resp.StatusCode, DiagnosticsHeader, resp.Header.Get(DiagnosticsHeader))
		}
	}
}
