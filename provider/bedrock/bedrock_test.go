// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/awsstream"
	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
)

func serve(t *testing.T, h http.HandlerFunc) (*Provider, *[]string, *[]*http.Request) {
	t.Helper()
	var bodies []string
	var reqs []*http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		reqs = append(reqs, r)
		h(w, r)
	}))
	t.Cleanup(srv.Close)
	p, err := New(Options{
		Region: "us-east-1", BaseURL: srv.URL,
		Credentials: func(context.Context) (Credentials, error) {
			return Credentials{AccessKeyID: "AK", SecretAccessKey: "SK", SessionToken: "TOKEN"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return p, &bodies, &reqs
}

func attempt(req *llm.Request) *llm.Attempt {
	return &llm.Attempt{Request: req, Model: "anthropic.claude-sonnet-4:0", Recorder: llm.NopRecorder{}}
}

func hello() *llm.Request {
	return &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}}
}

func TestRequestIsSignedAndModelSitsInThePath(t *testing.T) {
	p, _, reqs := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	if _, err := p.Complete(context.Background(), attempt(hello())); err != nil {
		t.Fatal(err)
	}
	r := (*reqs)[0]
	if !strings.Contains(r.URL.Path, "anthropic.claude-sonnet-4:0") {
		t.Fatalf("path = %q: the model id lives in the URL here", r.URL.Path)
	}
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "AWS4-HMAC-SHA256 ") || !strings.Contains(auth, "us-east-1/bedrock/aws4_request") {
		t.Fatalf("authorization = %q", auth)
	}
	if r.Header.Get("X-Amz-Security-Token") != "TOKEN" {
		t.Fatal("the session token must travel")
	}
}

func TestUsageCacheFiguresSitBesideTheInputCount(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},
			"stopReason":"end_turn","usage":{"inputTokens":12,"outputTokens":40,
			"cacheReadInputTokens":900,"cacheWriteInputTokens":100},"metrics":{"latencyMs":250}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Usage.Prompt != 1012 {
		t.Fatalf("prompt = %d: the cache counts are beside inputTokens on this service", resp.Usage.Prompt)
	}
	if resp.Usage.DurationMS != 250 {
		t.Fatalf("duration = %d", resp.Usage.DurationMS)
	}
}

func TestRolesAlternateStrictly(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[]}},"stopReason":"end_turn","usage":{}}`))
	})
	_, err := p.Complete(context.Background(), attempt(&llm.Request{Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "one"}}},
		{Role: llm.RoleTool, Content: []llm.Part{&llm.ToolResult{ID: "t1", Content: []llm.Part{&llm.Text{Text: "done"}}}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}},
	}}))
	if err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Messages []struct {
			Role    string           `json:"role"`
			Content []map[string]any `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if len(sent.Messages) != 2 || len(sent.Messages[0].Content) != 2 {
		t.Fatalf("messages = %+v: the tool result must join the user turn", sent.Messages)
	}
}

func toolLoop(results ...llm.Part) *llm.Request {
	return &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "make the directory"}}},
		{Role: llm.RoleAssistant, Content: []llm.Part{&llm.ToolUse{ID: "t1", Name: "shell", Args: json.RawMessage(`{"cmd":"mkdir x"}`)}}},
		{Role: llm.RoleTool, Content: results},
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "summarise"}}},
	}}
}

type sentBody struct {
	Messages []struct {
		Role    string `json:"role"`
		Content []struct {
			ToolResult *struct {
				Content json.RawMessage `json:"content"`
			} `json:"toolResult"`
		} `json:"content"`
	} `json:"messages"`
	ToolConfig *struct {
		Tools []struct {
			ToolSpec struct {
				Name        string          `json:"name"`
				InputSchema json.RawMessage `json:"inputSchema"`
			} `json:"toolSpec"`
		} `json:"tools"`
	} `json:"toolConfig"`
}

func sent(t *testing.T, body string) sentBody {
	t.Helper()
	var out sentBody
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func TestAToolResultWithNothingInItStillCarriesABlock(t *testing.T) {
	for name, content := range map[string][]llm.Part{
		"none":         nil,
		"empty text":   {&llm.Text{}},
		"only dropped": {&llm.Image{URL: "https://example.com/a.png"}},
	} {
		t.Run(name, func(t *testing.T) {
			p, bodies, _ := serve(t, cached())
			if _, err := p.Complete(context.Background(), attempt(toolLoop(&llm.ToolResult{ID: "t1", Content: content}))); err != nil {
				t.Fatal(err)
			}
			var found bool
			for _, m := range sent(t, (*bodies)[0]).Messages {
				for _, block := range m.Content {
					if block.ToolResult == nil {
						continue
					}
					found = true
					var inner []map[string]any
					if err := json.Unmarshal(block.ToolResult.Content, &inner); err != nil || len(inner) == 0 {
						t.Fatalf("content = %s: Converse refuses a tool result with no block in it", block.ToolResult.Content)
					}
					if text, _ := inner[0]["text"].(string); strings.TrimSpace(text) == "" {
						t.Fatalf("content = %s: a blank block says nothing either", block.ToolResult.Content)
					}
				}
			}
			if !found {
				t.Fatalf("body = %s: the tool result went missing", (*bodies)[0])
			}
		})
	}
}

func TestToolHistoryWithoutToolsDeclaresTheToolsItNames(t *testing.T) {
	p, bodies, _ := serve(t, cached())
	resp, err := p.Complete(context.Background(), attempt(toolLoop(&llm.ToolResult{ID: "t1", Name: "shell", Content: []llm.Part{&llm.Text{Text: "done"}}})))
	if err != nil {
		t.Fatal(err)
	}
	cfg := sent(t, (*bodies)[0]).ToolConfig
	if cfg == nil || len(cfg.Tools) != 1 || cfg.Tools[0].ToolSpec.Name != "shell" || len(cfg.Tools[0].ToolSpec.InputSchema) == 0 {
		t.Fatalf("body = %s: Converse refuses toolUse and toolResult blocks without a toolConfig naming them", (*bodies)[0])
	}
	if !resp.Diagnostics.Has(llm.DropToolDefinition) {
		t.Fatalf("diagnostics = %v: tools the request did not offer were declared without a word", resp.Diagnostics.Codes())
	}

	p, bodies, _ = serve(t, cached())
	req := toolLoop(&llm.ToolResult{ID: "t1", Content: []llm.Part{&llm.Text{Text: "done"}}})
	req.Tools = []llm.Tool{{Name: "search", Schema: []byte(`{"type":"object"}`)}}
	if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
		t.Fatal(err)
	}
	if cfg := sent(t, (*bodies)[0]).ToolConfig; cfg == nil || len(cfg.Tools) != 1 || cfg.Tools[0].ToolSpec.Name != "search" {
		t.Fatalf("body = %s: the tools a request offers are the ones sent", (*bodies)[0])
	}

	p, bodies, _ = serve(t, cached())
	if _, err := p.Complete(context.Background(), attempt(hello())); err != nil {
		t.Fatal(err)
	}
	if sent(t, (*bodies)[0]).ToolConfig != nil {
		t.Fatalf("body = %s: a conversation with no tools in it needs no toolConfig", (*bodies)[0])
	}
}

func TestThinkingKeepsItsSignature(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[
			{"reasoningContent":{"reasoningText":{"text":"thought","signature":"sig"}}},
			{"text":"answer"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	req := hello()
	req.Messages = append(req.Messages, llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
		&llm.Thinking{Text: "earlier", Signature: "sig-0"},
	}})
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains((*bodies)[0], `"signature":"sig-0"`) {
		t.Fatalf("body = %s: replayed reasoning must carry its signature", (*bodies)[0])
	}
	th, ok := resp.Choices[0].Message.Content[0].(*llm.Thinking)
	if !ok || th.Signature != "sig" {
		t.Fatalf("thinking = %+v", resp.Choices[0].Message.Content[0])
	}
}

// A thinking block with no signature, whether it never had one or a foreign
// one was stripped from it upstream, is not this backend's own reasoning and
// must not be replayed as reasoningText with an empty signature: sent that
// way it is indistinguishable from a block Claude never verified, and Claude
// refuses it. Regression test for r3-cloud-1.
func TestUnsignedThinkingIsNotReplayedWithAnEmptySignature(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"answer"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	req := hello()
	req.Messages = append(req.Messages, llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
		&llm.Thinking{Text: "plan", Signature: ""},
		&llm.Text{Text: "hi again"},
	}})
	resp, err := p.Complete(context.Background(), attempt(req))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains((*bodies)[0], `"signature":""`) {
		t.Fatalf("body = %s: an unsigned block must not be sent as reasoningText with an empty signature", (*bodies)[0])
	}
	if !resp.Diagnostics.Has(llm.DropSignature) {
		t.Fatalf("diagnostics = %+v: the drop must be reported", resp.Diagnostics)
	}
}

func TestStreamDecodesTheEventStream(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		write := func(name string, payload string) {
			_, _ = w.Write(awsstream.Encode(map[string]string{":message-type": "event", ":event-type": name}, []byte(payload)))
		}
		write("messageStart", `{"role":"assistant"}`)
		write("contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"t1","name":"search"}}}`)
		write("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"q\":"}}}`)
		write("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"1}"}}}`)
		write("contentBlockStop", `{"contentBlockIndex":0}`)
		write("messageStop", `{"stopReason":"tool_use"}`)
		write("metadata", `{"usage":{"inputTokens":10,"outputTokens":4,"cacheReadInputTokens":90},"metrics":{"latencyMs":33}}`)
	})
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := wirecore.Fold(s, wirecore.FoldOptions{})
	if err != nil {
		t.Fatal(err)
	}
	tu := llm.ToolUses(resp.Choices[0].Message.Content)
	if len(tu) != 1 || string(tu[0].Args) != `{"q":1}` {
		t.Fatalf("tool call = %+v", tu)
	}
	if resp.Choices[0].Stop != llm.StopToolUse {
		t.Fatalf("stop = %q", resp.Choices[0].Stop)
	}
	if resp.Usage.Prompt != 100 || resp.Usage.Completion != 4 {
		t.Fatalf("usage = %+v: the figures arrive on their own event after the answer", resp.Usage)
	}
}

func TestStreamExceptionIsTypedAndRetryable(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(awsstream.Encode(map[string]string{":message-type": "event", ":event-type": "messageStart"}, []byte(`{"role":"assistant"}`)))
		_, _ = w.Write(awsstream.Encode(map[string]string{
			":message-type": "exception", ":exception-type": "throttlingException",
		}, []byte(`{"message":"slow down"}`)))
	})
	s, err := p.Stream(context.Background(), attempt(hello()))
	if err != nil {
		t.Fatal(err)
	}
	_, ferr := wirecore.Fold(s, wirecore.FoldOptions{})
	if ferr == nil {
		t.Fatal("an exception inside the stream must not read as a short success")
	}
	e := llm.AsError(ferr)
	if e.Kind != llm.KindRateLimit || !e.Retryable {
		t.Fatalf("error = %+v: the router decides on this", e)
	}
}

func TestMissingCredentialsNeverReachTheWire(t *testing.T) {
	p, err := New(Options{Region: "eu-west-2", Credentials: func(context.Context) (Credentials, error) {
		return Credentials{}, nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	_, err = p.Complete(context.Background(), attempt(hello()))
	if err == nil {
		t.Fatal("an unsigned request must not be sent")
	}
	if llm.AsError(err).Sent {
		t.Fatal("a request that never left must not book as billed")
	}
}

func TestProbeAnswersWhetherTheCredentialResolves(t *testing.T) {
	good, err := New(Options{
		ID: "up", Region: "us-east-1",
		Credentials: func(context.Context) (Credentials, error) {
			return Credentials{AccessKeyID: "AKID", SecretAccessKey: "secret"}, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := good.Probe(context.Background()); err != nil {
		t.Fatalf("a resolvable credential is ready: %v", err)
	}

	for name, creds := range map[string]func(context.Context) (Credentials, error){
		"unresolvable": func(context.Context) (Credentials, error) {
			return Credentials{}, errors.New("no profile")
		},
		"half a pair": func(context.Context) (Credentials, error) {
			return Credentials{AccessKeyID: "AKID"}, nil
		},
		"empty": func(context.Context) (Credentials, error) { return Credentials{}, nil },
	} {
		t.Run(name, func(t *testing.T) {
			p, err := New(Options{ID: "up", Region: "us-east-1", Credentials: creds})
			if err != nil {
				t.Fatal(err)
			}
			if err := p.Probe(context.Background()); err == nil {
				t.Fatal("a process that cannot authenticate is not ready")
			}
		})
	}
}

func TestAnUnmodelledPieceIsCarriedOnBothPaths(t *testing.T) {
	parts, diags := decodeBlocks([]json.RawMessage{
		json.RawMessage(`{"video":{"format":"mp4","source":{"bytes":"AAA"}}}`),
	})
	if !hasRawPart(parts) {
		t.Errorf("the unary path dropped it: %+v", parts)
	}
	if !diags.Has(llm.DropUnknownBlock) {
		t.Errorf("diagnostics = %v", diags.Codes())
	}

	b := &parser{provider: "bedrock", blocks: map[int]string{}}
	frame := b.unmodelled([]byte(`{"contentBlockIndex":0,"delta":{"citation":{"title":"x"}}}`), "delta")
	if frame == nil || !hasRawPart(frame.Delta) {
		t.Errorf("the stream dropped it: %+v", frame)
	}
	if b.unmodelled([]byte(`{"delta":{}}`), "delta") != nil {
		t.Error("an empty piece was reported as a drop")
	}
}

func hasRawPart(parts []llm.Part) bool {
	for _, p := range parts {
		if _, ok := p.(*llm.Raw); ok {
			return true
		}
	}
	return false
}

func TestAnErrorFrameIsAFailureWhicheverWayItIsSpelt(t *testing.T) {
	b := &parser{provider: "bedrock", blocks: map[int]string{}}
	e := b.exception(awsstream.Message{
		Headers: map[string]string{
			":message-type":  "error",
			":error-code":    "throttlingException",
			":error-message": "slow down",
		},
	})
	if e == nil {
		t.Fatal("no error")
	}
	if e.Kind != llm.KindRateLimit || !e.Retryable {
		t.Fatalf("error = %+v", e)
	}
	if e.Message != "slow down" || e.Code != "throttlingException" {
		t.Fatalf("error = %+v", e)
	}

	bare := b.exception(awsstream.Message{Headers: map[string]string{":message-type": "error"}})
	if bare.Message == "" {
		t.Error("a failure with no words is still a failure")
	}
}

func cached() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},
			"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`))
	}
}

func TestCachePointsMarkWhatWasAskedToBeKept(t *testing.T) {
	p, bodies, _ := serve(t, cached())
	req := &llm.Request{
		Model:       "m",
		System:      []llm.Part{&llm.Text{Text: "long standing instructions"}},
		SystemCache: &llm.CacheHint{},
		Tools:       []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`), Cache: &llm.CacheHint{}}},
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}, Cache: &llm.CacheHint{}},
		},
	}
	if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
		t.Fatal(err)
	}
	body := (*bodies)[0]
	if n := strings.Count(body, "cachePoint"); n != 3 {
		t.Fatalf("cache points = %d, want 3: the system, the turn and the tool were each asked for\n%s", n, body)
	}

	var out struct {
		System []map[string]any `json:"system"`
	}
	if err := json.Unmarshal([]byte(body), &out); err != nil {
		t.Fatal(err)
	}
	if len(out.System) != 2 || out.System[1]["cachePoint"] == nil {
		t.Fatalf("system = %v: the point marks everything before it, so it comes last", out.System)
	}
}

func TestCachePointsStopAtTheFourth(t *testing.T) {
	p, bodies, _ := serve(t, cached())
	req := &llm.Request{Model: "m", System: []llm.Part{&llm.Text{Text: "instructions"}}, SystemCache: &llm.CacheHint{}}
	for _, text := range []string{"one", "two", "three", "four", "five"} {
		req.Messages = append(req.Messages,
			llm.Message{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: text}}, Cache: &llm.CacheHint{}},
			llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "ok"}}})
	}
	if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
		t.Fatal(err)
	}
	if n := strings.Count((*bodies)[0], "cachePoint"); n != maxBreakpoints {
		t.Fatalf("cache points = %d, want %d", n, maxBreakpoints)
	}
}

func TestAModelThatCannotCacheSaysSoRatherThanFailing(t *testing.T) {
	p, bodies, _ := serve(t, cached())
	a := attempt(&llm.Request{
		Model: "m", System: []llm.Part{&llm.Text{Text: "instructions"}}, SystemCache: &llm.CacheHint{},
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	})
	a.Model = "anthropic.claude-3-sonnet-20240229-v1:0"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains((*bodies)[0], "cachePoint") {
		t.Fatal("a cache point on a model that does not serve caching is a refused request")
	}
	if !resp.Diagnostics.Has(llm.DropCacheHint) {
		t.Fatalf("diagnostics = %+v: the hint went missing without a word", resp.Diagnostics)
	}
}

func TestAToolChoiceOfNoneIsReported(t *testing.T) {
	p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},
			"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1}}`))
	})
	resp, err := p.Complete(context.Background(), attempt(&llm.Request{
		Model:      "m",
		Tools:      []llm.Tool{{Name: "f", Schema: []byte(`{"type":"object"}`)}},
		ToolChoice: &llm.ToolChoice{Mode: llm.ToolNone},
		Messages:   []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
	}))
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Diagnostics.Has(llm.DropToolDefinition) {
		t.Fatalf("diagnostics = %+v: the choice went missing without a word", resp.Diagnostics)
	}
}

func TestAdaptiveThinkingReachesTheModelAsNamed(t *testing.T) {
	p, bodies, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"output":{"message":{"role":"assistant","content":[{"text":"ok"}]}},"stopReason":"end_turn","usage":{}}`))
	})
	req := hello()
	req.Thinking = &llm.ThinkingConfig{Enabled: true, Budget: -1, Adaptive: true}
	if _, err := p.Complete(context.Background(), attempt(req)); err != nil {
		t.Fatal(err)
	}
	var sent struct {
		Fields struct {
			Thinking json.RawMessage `json:"thinking"`
		} `json:"additionalModelRequestFields"`
	}
	if err := json.Unmarshal([]byte((*bodies)[0]), &sent); err != nil {
		t.Fatal(err)
	}
	if string(sent.Fields.Thinking) != `{"type":"adaptive"}` {
		t.Fatalf("thinking = %s: the models that take this mode refuse a budget", sent.Fields.Thinking)
	}
}

func TestOnlyModelsThatServeCachingAreSentCachePoints(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"anthropic.claude-3-7-sonnet-20250219-v1:0", true},
		{"us.anthropic.claude-sonnet-4-20250514-v1:0", true},
		{"amazon.nova-lite-v1:0", true},
		{"anthropic.claude-3-5-sonnet-20241022-v2:0", false},
		{"anthropic.claude-3-sonnet-20240229-v1:0", false},
		{"meta.llama3-70b-instruct-v1:0", false},
	}
	for _, c := range cases {
		if got := cacheable(c.model); got != c.want {
			t.Errorf("cacheable(%q) = %v, want %v: a cache point on a model that refuses one fails the whole request",
				c.model, got, c.want)
		}
	}
}

// The region is spliced into the host a call is signed for and sent to, so
// one that could end the host early or name another is refused at once.
func TestARegionThatCouldMoveTheHostIsRefused(t *testing.T) {
	creds := func(context.Context) (Credentials, error) {
		return Credentials{AccessKeyID: "AK", SecretAccessKey: "SK"}, nil
	}
	for _, region := range []string{"x@127.0.0.1:18409/#", "us-east-1.evil.example", "US-EAST-1", "a/b"} {
		if _, err := New(Options{Region: region, Credentials: creds}); err == nil {
			t.Fatalf("region %q was spliced into the host", region)
		}
	}
	if _, err := New(Options{Region: "eu-west-2", Credentials: creds}); err != nil {
		t.Fatalf("a region's name was refused: %v", err)
	}
}
