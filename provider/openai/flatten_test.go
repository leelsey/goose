// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
)

func TestAFlattenedToolResultIsReported(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	})
	req := &llm.Request{
		Model: "m",
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "look"}}},
			{Role: llm.RoleAssistant, Content: []llm.Part{&llm.ToolUse{ID: "1", Name: "shot"}}},
			{Role: llm.RoleTool, Content: []llm.Part{&llm.ToolResult{ID: "1", Content: []llm.Part{
				&llm.Text{Text: "here"},
				&llm.Image{MIME: "image/png", Data: []byte{1, 2, 3}},
			}}}},
		},
	}
	resp, err := p.Complete(context.Background(), attempt("m", req))
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Diagnostics.Has(llm.DropImage) {
		t.Fatalf("diagnostics = %v: the image inside the tool result went unreported", resp.Diagnostics.Codes())
	}
}

func TestTheStreamedResponseIDIsTheBackendsOwn(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"resp_real","model":"m","status":"in_progress"}}`,
			`{"type":"response.output_text.delta","delta":"hi"}`,
			`{"type":"response.completed","response":{"id":"resp_real","model":"m","status":"completed","usage":{"input_tokens":1,"output_tokens":1}}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	a := attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}})
	a.Format = "openai-responses"
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	if got := s.Meta().ID; got != "resp_real" {
		t.Fatalf("id = %q, want the backend's own before the first frame is written", got)
	}
	if got := s.Meta().Session; got != "resp_real" {
		t.Fatalf("session = %q", got)
	}
	var text strings.Builder
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if tx, ok := part.(*llm.Text); ok {
				text.WriteString(tx.Text)
			}
		}
	}
	if text.String() != "hi" {
		t.Fatalf("text = %q", text.String())
	}
}

func TestAGeneratedImageIsReadOffTheAnswer(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","model":"gpt-image-1","status":"completed","output":[
			{"type":"image_generation_call","id":"ig_1","status":"completed","result":"AQID","output_format":"png","revised_prompt":"a goose"}]}`))
	})
	a := attempt("gpt-image-1", &llm.Request{Model: "gpt-image-1", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "draw"}}},
	}})
	a.Format = "openai-responses"
	resp, err := p.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	img, ok := resp.Choices[0].Message.Content[0].(*llm.Image)
	if !ok {
		t.Fatalf("content = %T", resp.Choices[0].Message.Content[0])
	}
	if string(img.Data) != "\x01\x02\x03" || img.MIME != "image/png" || img.Prompt != "a goose" {
		t.Fatalf("image = %+v", img)
	}
}

func TestStreamedImagePreviewsAreWholeRenderings(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"resp_1","model":"gpt-image-1","status":"in_progress"}}`,
			`{"type":"response.image_generation_call.partial_image","item_id":"ig_1","partial_image_index":0,"partial_image_b64":"AQ==","output_format":"png"}`,
			`{"type":"response.output_item.done","item":{"type":"image_generation_call","id":"ig_1","result":"AQID","output_format":"png","revised_prompt":"a goose"}}`,
			`{"type":"response.completed","response":{"id":"resp_1","model":"gpt-image-1","status":"completed"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	a := attempt("gpt-image-1", &llm.Request{Model: "gpt-image-1", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "draw"}}},
	}})
	a.Format = "openai-responses"
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var seen []*llm.Image
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if img, ok := part.(*llm.Image); ok {
				seen = append(seen, img)
			}
		}
	}
	if len(seen) != 2 {
		t.Fatalf("images = %d, want a preview and the finished one", len(seen))
	}
	if !seen[0].Partial || string(seen[0].Data) != "\x01" {
		t.Errorf("preview = %+v", seen[0])
	}
	if seen[1].Partial || string(seen[1].Data) != "\x01\x02\x03" || seen[1].Prompt != "a goose" {
		t.Errorf("finished = %+v", seen[1])
	}
}

func TestAnImageOnlyModelIsRefusedByAFormatThatCannotCarryOne(t *testing.T) {
	p, bodies, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		t.Error("the backend was asked for an answer the client could not read")
	})
	a := attempt("gpt-image-1", &llm.Request{Model: "gpt-image-1", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "draw"}}},
	}})
	a.Format = "openai"
	a.Renders = llm.Renders{ToolCalls: true, Candidates: 128}
	_, err := p.Complete(context.Background(), a)
	if err == nil {
		t.Fatal("the call was made")
	}
	e := llm.AsError(err)
	if e.Kind != llm.KindUnsupported || e.Sent {
		t.Fatalf("error = %+v: nothing left the process, so nothing was billed", e)
	}
	if !strings.Contains(e.Message, "/v1/responses") {
		t.Errorf("the refusal must name the endpoint that serves it: %s", e.Message)
	}
	if len(*bodies) != 0 {
		t.Errorf("requests = %d", len(*bodies))
	}

	a.Format = "openai-responses"
	a.Renders = llm.Renders{ToolCalls: true, Images: true, Candidates: 1}
	if err := p.guardImageOnly(a); err != nil {
		t.Fatalf("the wire that has an item for it was refused: %v", err)
	}
}

func TestAnUnmodelledOutputItemIsCarriedOnBothPaths(t *testing.T) {
	const item = `{"type":"web_search_call","id":"ws_1","status":"completed"}`

	unary, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"r","model":"m","status":"completed","output":[` + item + `]}`))
	})
	a := attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}})
	a.Format = "openai-responses"
	resp, err := unary.Complete(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	if !hasRaw(resp.Choices[0].Message.Content, "web_search_call") {
		t.Fatalf("unary dropped it: %+v", resp.Choices[0].Message.Content)
	}

	streamed, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range []string{
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.output_item.done","item":` + item + `}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		} {
			_, _ = w.Write([]byte("data: " + e + "\n\n"))
		}
	})
	b := attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}})
	b.Format = "openai-responses"
	s, err := streamed.Stream(context.Background(), b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var got []llm.Part
	for f := range s.Frames() {
		got = append(got, f.Delta...)
	}
	if !hasRaw(got, "web_search_call") {
		t.Fatalf("the stream dropped it: %+v", got)
	}
}

func hasRaw(parts []llm.Part, kind string) bool {
	for _, p := range parts {
		if r, ok := p.(*llm.Raw); ok && r.Type == kind {
			return true
		}
	}
	return false
}

func TestAStreamedReasoningSignatureIsRead(t *testing.T) {
	p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range []string{
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.reasoning_summary_text.delta","delta":"because"}`,
			`{"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","encrypted_content":"sig"}}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		} {
			_, _ = w.Write([]byte("data: " + ev + "\n\n"))
		}
	})
	a := attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
	}})
	a.Format = "openai-responses"
	s, err := p.Stream(context.Background(), a)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var signature, text string
	for f := range s.Frames() {
		for _, part := range f.Delta {
			if th, ok := part.(*llm.Thinking); ok {
				text += th.Text
				if th.Signature != "" {
					signature = th.Signature
				}
			}
		}
	}
	if text != "because" || signature != "sig" {
		t.Fatalf("thinking = %q signature = %q", text, signature)
	}
}

func TestAMessageDeliveredWholeIsNotAnEmptyAnswer(t *testing.T) {
	for name, events := range map[string][]string{
		"nothing streamed": {
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1","role":"assistant"}}`,
			`{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello there"}]}}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		},
		"half streamed": {
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1","role":"assistant"}}`,
			`{"type":"response.output_text.delta","item_id":"msg_1","delta":"hello"}`,
			`{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello there"}]}}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		},
		"all streamed": {
			`{"type":"response.created","response":{"id":"r","model":"m","status":"in_progress"}}`,
			`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1","role":"assistant"}}`,
			`{"type":"response.output_text.delta","item_id":"msg_1","delta":"hello there"}`,
			`{"type":"response.output_item.done","item":{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello there"}]}}`,
			`{"type":"response.completed","response":{"id":"r","model":"m","status":"completed"}}`,
		},
	} {
		p, _, _ := serve(t, Options{Dialect: OpenAIDialect()}, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, ev := range events {
				_, _ = w.Write([]byte("data: " + ev + "\n\n"))
			}
		})
		a := attempt("m", &llm.Request{Model: "m", Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}},
		}})
		a.Format = "openai-responses"
		s, err := p.Stream(context.Background(), a)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Prime(); err != nil {
			t.Fatal(err)
		}
		var text strings.Builder
		for f := range s.Frames() {
			for _, part := range f.Delta {
				if tx, ok := part.(*llm.Text); ok {
					text.WriteString(tx.Text)
				}
			}
		}
		s.Close()
		if got := text.String(); got != "hello there" {
			t.Errorf("%s: text = %q, want the whole answer exactly once", name, got)
		}
	}
}
