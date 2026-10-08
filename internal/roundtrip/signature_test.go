// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package roundtrip

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/httpx"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/anthropic"
	"github.com/leelsey/goose/provider/bedrock"
	"github.com/leelsey/goose/provider/google"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/route"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/openaichat"
	"github.com/leelsey/goose/wire/openairesp"
)

// scripted is a proxy with every client wire in front of one backend that
// answers each call with the next reply and keeps what it was sent.
func scripted(t *testing.T, build func(url string) (llm.Provider, error), replies ...string) (*httptest.Server, func() []string) {
	t.Helper()
	var mu sync.Mutex
	var bodies []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		n := len(bodies)
		bodies = append(bodies, string(b))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(replies[min(n, len(replies)-1)]))
	}))
	t.Cleanup(upstream.Close)
	p, err := build(upstream.URL)
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(p))
	if err != nil {
		t.Fatal(err)
	}
	s, err := httpx.New(c, httpx.Options{Codecs: []wire.Codec{
		openaichat.New(openaichat.Options{}), anthropicmsg.New(anthropicmsg.Options{}),
		openairesp.New(openairesp.Options{}), geminiapi.New(geminiapi.Options{}),
	}})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(s.Handler())
	t.Cleanup(front.Close)
	return front, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), bodies...)
	}
}

func posted(t *testing.T, front *httptest.Server, path, body string) (string, string) {
	t.Helper()
	resp, err := front.Client().Post(front.URL+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("%s: status %d: %s", path, resp.StatusCode, raw)
	}
	return string(raw), resp.Header.Get(httpx.DiagnosticsHeader)
}

func geminiBackend(url string) (llm.Provider, error) {
	return google.New(google.Options{ID: "up", BaseURL: url, APIKey: "k"})
}

const (
	geminiCall = `{"candidates":[{"content":{"role":"model","parts":[
		{"functionCall":{"id":"call_1","name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"SIG-abc"}]},
		"finishReason":"STOP"}]}`
	geminiDone = `{"candidates":[{"content":{"role":"model","parts":[{"text":"20C"}]},"finishReason":"STOP"}]}`
)

// A tool loop against Gemini runs past its first call on every client wire.
// The result is named after the call it answers, which Gemini requires on
// every model, and the call goes back with its signature where the wire kept
// one, or with Google's placeholder where it could not, which Gemini 3 needs.
func TestAGeminiToolLoopRunsPastItsFirstCallOnEveryWire(t *testing.T) {
	for _, tc := range []struct {
		name, path, first string
		second            func(answer string) string
		kept              bool
	}{
		{"chat", "/v1/chat/completions",
			`{"model":"gemini-3-pro","messages":[{"role":"user","content":"weather?"}],
			  "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object"}}}]}`,
			func(answer string) string {
				var out struct {
					Choices []struct {
						Message struct {
							ToolCalls json.RawMessage `json:"tool_calls"`
						} `json:"message"`
					} `json:"choices"`
				}
				if err := json.Unmarshal([]byte(answer), &out); err != nil || len(out.Choices) == 0 {
					t.Fatalf("answer = %s", answer)
				}
				return `{"model":"gemini-3-pro","messages":[{"role":"user","content":"weather?"},
				  {"role":"assistant","content":null,"tool_calls":` + string(out.Choices[0].Message.ToolCalls) + `},
				  {"role":"tool","tool_call_id":"call_1","content":"20C"}]}`
			}, true},
		{"anthropic", "/v1/messages",
			`{"model":"gemini-3-pro","max_tokens":64,"messages":[{"role":"user","content":"weather?"}],
			  "tools":[{"name":"get_weather","input_schema":{"type":"object"}}]}`,
			func(string) string {
				return `{"model":"gemini-3-pro","max_tokens":64,"messages":[{"role":"user","content":"weather?"},
				  {"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"Paris"}}]},
				  {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":"20C"}]}]}`
			}, false},
		{"responses", "/v1/responses",
			`{"model":"gemini-3-pro","input":"weather?","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]}`,
			func(string) string {
				return `{"model":"gemini-3-pro","input":[{"role":"user","content":"weather?"},
				  {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
				  {"type":"function_call_output","call_id":"call_1","output":"20C"}]}`
			}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			front, sent := scripted(t, geminiBackend, geminiCall, geminiDone)
			answer, diags := posted(t, front, tc.path, tc.first)
			if carried := strings.Contains(answer, "SIG-abc"); carried != tc.kept {
				t.Fatalf("answer = %s: the signature is carried where this wire has a field for it", answer)
			}
			if lost := strings.Contains(diags, llm.DropSignature); lost == tc.kept {
				t.Fatalf("diagnostics = %q: a signature that did not travel is said to have been lost", diags)
			}

			posted(t, front, tc.path, tc.second(answer))
			backend := sent()[1]
			if !strings.Contains(backend, `"functionResponse":{"name":"get_weather"`) {
				t.Fatalf("backend got %s: a function response names its function", backend)
			}
			want := `"thoughtSignature":"skip_thought_signature_validator"`
			if tc.kept {
				want = `"thoughtSignature":"SIG-abc"`
			}
			if !strings.Contains(backend, want) {
				t.Fatalf("backend got %s, want %s", backend, want)
			}
		})
	}
}

// A signature is the backend's that minted it. Replayed from one vendor's
// wire to another vendor's backend, it is stripped before the call, which
// that backend would otherwise refuse with a 400 that ends the chain.
func TestAnotherVendorsSignatureIsNotReplayedToABackend(t *testing.T) {
	t.Run("an OpenAI reasoning item to Claude", func(t *testing.T) {
		front, sent := scripted(t, func(url string) (llm.Provider, error) {
			return anthropic.New(anthropic.Options{ID: "up", BaseURL: url, APIKey: "k"})
		}, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
		_, diags := posted(t, front, "/v1/responses", `{"model":"claude","input":[{"role":"user","content":"hi"},
			{"type":"reasoning","id":"rs_up","summary":[{"type":"summary_text","text":"thought"}],"encrypted_content":"gAAAA-openai"},
			{"role":"assistant","content":"hello"},{"role":"user","content":"go on"}]}`)
		if strings.Contains(sent()[0], "gAAAA-openai") || !strings.Contains(diags, llm.DropSignature) {
			t.Fatalf("backend got %s, diagnostics %q", sent()[0], diags)
		}
	})
	t.Run("a Gemini thought to Claude on Bedrock", func(t *testing.T) {
		front, sent := scripted(t, func(url string) (llm.Provider, error) {
			return bedrock.New(bedrock.Options{ID: "up", Region: "us-east-1", BaseURL: url,
				Credentials: func(context.Context) (bedrock.Credentials, error) {
					return bedrock.Credentials{AccessKeyID: "AK", SecretAccessKey: "SK"}, nil
				}})
		}, `{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn",
			"usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}`)
		_, diags := posted(t, front, "/v1beta/models/claude:generateContent", `{"contents":[
			{"role":"user","parts":[{"text":"hi"}]},
			{"role":"model","parts":[{"text":"plan","thought":true,"thoughtSignature":"R0VNSU5JLVNJRw=="},{"text":"hello"}]},
			{"role":"user","parts":[{"text":"go on"}]}]}`)
		if strings.Contains(sent()[0], "R0VNSU5JLVNJRw==") || !strings.Contains(diags, llm.DropSignature) {
			t.Fatalf("backend got %s, diagnostics %q", sent()[0], diags)
		}
	})
}

// A chain whose first link is OpenAI Responses and whose history holds Claude's
// thinking, under the default policy that accepts the loss. The Anthropic
// signature is stripped for OpenAI, and the thinking left with nothing OpenAI
// can find it by is not sent as a reasoning item it would refuse with a 400
// that ends the chain.
func TestAForeignThinkingDoesNotEndAChainAtOpenAI(t *testing.T) {
	var mu sync.Mutex
	var gptBodies []string
	claudeHits := 0
	gpt := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gptBodies = append(gptBodies, string(b))
		mu.Unlock()
		if strings.Contains(string(b), `"type":"reasoning"`) {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"Missing required parameter: 'input[1].id'.","type":"invalid_request_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`))
	}))
	t.Cleanup(gpt.Close)
	claude := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		claudeHits++
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(claude.Close)
	op, err := openai.New(openai.Options{ID: "gpt", BaseURL: gpt.URL, APIKey: "k", Dialect: openai.OpenAIDialect()})
	if err != nil {
		t.Fatal(err)
	}
	ap, err := anthropic.New(anthropic.Options{ID: "claude", BaseURL: claude.URL, APIKey: "k"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := goose.New(goose.WithProvider(op), goose.WithProvider(ap),
		goose.WithRoutes(route.Rule{Name: "chain", Match: route.Always(), Chain: []route.Target{
			{Provider: "gpt"}, {Provider: "claude"},
		}}))
	if err != nil {
		t.Fatal(err)
	}
	for name, thought := range map[string]*llm.Thinking{
		"claude": {Text: "because", Signature: "ErUB-anthropic", Signer: llm.FamilyAnthropic},
		"gemini": {Text: "because", Signature: "CiQB-google", Signer: llm.FamilyGoogle},
	} {
		mu.Lock()
		gptBodies = nil
		mu.Unlock()
		_, err := c.Chat(context.Background(), &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
			{Role: llm.RoleAssistant, Content: []llm.Part{thought, &llm.Text{Text: "hello"}}},
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "go on"}}},
		}}, goose.WithFormat("openai-responses"))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		mu.Lock()
		sent := gptBodies
		mu.Unlock()
		if len(sent) != 1 || strings.Contains(sent[0], `"type":"reasoning"`) {
			t.Fatalf("%s: gpt got %q", name, sent)
		}
	}
	if claudeHits != 0 {
		t.Fatal("the chain moved past a link that could serve the turn")
	}
}

// Claude's thinking reaches a Gemini client marked as Anthropic's, and the
// field it travels in is bytes, which Gemini's typed SDKs decode from base64.
// Handed back as such a client re-encodes it, it reaches Claude as issued.
func TestClaudesThinkingSurvivesATypedGeminiClient(t *testing.T) {
	const sig = "EqQBCkgIARABGAIiQPla+/8="
	front, sent := scripted(t, func(url string) (llm.Provider, error) {
		return anthropic.New(anthropic.Options{ID: "up", BaseURL: url, APIKey: "k"})
	}, `{"id":"m1","type":"message","role":"assistant","model":"m","content":[
		{"type":"thinking","thinking":"plan","signature":"`+sig+`"},{"type":"text","text":"ok"}],
		"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	answer, _ := posted(t, front, "/v1beta/models/claude:generateContent",
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`)
	type part struct {
		Text             string `json:"text,omitempty"`
		Thought          bool   `json:"thought,omitempty"`
		ThoughtSignature []byte `json:"thoughtSignature,omitempty"`
	}
	var typed struct {
		Candidates []struct {
			Content struct {
				Role  string `json:"role"`
				Parts []part `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	if err := json.Unmarshal([]byte(answer), &typed); err != nil || len(typed.Candidates) == 0 {
		t.Fatalf("answer = %s: a typed client cannot read it: %v", answer, err)
	}
	model, _ := json.Marshal(typed.Candidates[0].Content)
	posted(t, front, "/v1beta/models/claude:generateContent", `{"contents":[{"role":"user","parts":[{"text":"hi"}]},`+
		string(model)+`,{"role":"user","parts":[{"text":"go on"}]}]}`)
	if backend := sent()[1]; !strings.Contains(backend, `"signature":"`+sig+`"`) {
		t.Fatalf("backend got %s: Claude is handed back the signature it issued", backend)
	}
}
