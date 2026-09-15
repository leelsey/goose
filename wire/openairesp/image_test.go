// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

func TestAGeneratedImageIsRendered(t *testing.T) {
	out, diags, err := New(Options{}).EncodeResponse(&llm.Response{
		Model: "gpt-image-1",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
			&llm.Image{ID: "ig_1", MIME: "image/png", Data: []byte{1, 2, 3}, Prompt: "a goose, rewritten"},
		}}, Stop: llm.StopEnd}},
	}, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if diags.Has(llm.DropImage) {
		t.Fatalf("the image was dropped: %v", diags.Codes())
	}
	body := string(out.Body)
	for _, want := range []string{
		`"type":"image_generation_call"`,
		`"result":"` + base64.StdEncoding.EncodeToString([]byte{1, 2, 3}) + `"`,
		`"output_format":"png"`,
		`"revised_prompt":"a goose, rewritten"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("body is missing %s:\n%s", want, body)
		}
	}
}

func TestAStreamedImageIsAnnouncedAndCompleted(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "gpt-image-1"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Image{ID: "ig_1", MIME: "image/png", Data: []byte{9}, Partial: true}}},
		{Delta: []llm.Part{&llm.Image{ID: "ig_1", MIME: "image/png", Data: []byte{9, 9}, Prompt: "rewritten"}}},
		{Stop: llm.StopEnd},
	})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var w wire.BufferWriter
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{}, &w); err != nil {
		t.Fatal(err)
	}
	body := w.Body.String()
	for _, want := range []string{
		"response.image_generation_call.in_progress",
		"response.image_generation_call.partial_image",
		"response.image_generation_call.completed",
		`"partial_image_index":0`,
		`"result":"` + base64.StdEncoding.EncodeToString([]byte{9, 9}) + `"`,
		`"revised_prompt":"rewritten"`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("stream is missing %s:\n%s", want, body)
		}
	}
	last := lastEvent(t, body, "response.completed")
	if !strings.Contains(last, `"type":"image_generation_call"`) {
		t.Errorf("terminal response omits the image:\n%s", last)
	}
}

func lastEvent(t *testing.T, body, name string) string {
	t.Helper()
	var found string
	for _, line := range strings.Split(body, "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var probe struct {
			Type string `json:"type"`
		}
		if json.Unmarshal([]byte(data), &probe) == nil && probe.Type == name {
			found = data
		}
	}
	if found == "" {
		t.Fatalf("no %s event in:\n%s", name, body)
	}
	return found
}

func TestTwoGeneratedImagesAreTwoItems(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "gpt-image-1"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Image{ID: "ig_1", MIME: "image/png", Data: []byte{1}}}},
		{Delta: []llm.Part{&llm.Image{ID: "ig_2", MIME: "image/png", Data: []byte{2}}}},
		{Stop: llm.StopEnd},
	})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var w wire.BufferWriter
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{}, &w); err != nil {
		t.Fatal(err)
	}
	last := lastEvent(t, w.Body.String(), "response.completed")
	var env struct {
		Response struct {
			Output []struct {
				Type   string `json:"type"`
				ID     string `json:"id"`
				Result string `json:"result"`
			} `json:"output"`
		} `json:"response"`
	}
	if err := json.Unmarshal([]byte(last), &env); err != nil {
		t.Fatal(err)
	}
	var images []string
	for _, it := range env.Response.Output {
		if it.Type == "image_generation_call" {
			images = append(images, it.ID+":"+it.Result)
		}
	}
	if len(images) != 2 {
		t.Fatalf("images = %v", images)
	}
	if images[0] != "ig_1:"+base64.StdEncoding.EncodeToString([]byte{1}) ||
		images[1] != "ig_2:"+base64.StdEncoding.EncodeToString([]byte{2}) {
		t.Fatalf("images = %v", images)
	}
}

func TestSequenceNumbersStartAtZeroAndDoNotSkip(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "hi"}}},
		{Stop: llm.StopEnd},
	})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var w wire.BufferWriter
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{}, &w); err != nil {
		t.Fatal(err)
	}
	want := 0
	for _, line := range strings.Split(w.Body.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok {
			continue
		}
		var ev struct {
			Sequence *int `json:"sequence_number"`
		}
		if json.Unmarshal([]byte(data), &ev) != nil || ev.Sequence == nil {
			continue
		}
		if *ev.Sequence != want {
			t.Fatalf("sequence_number = %d, want %d\n%s", *ev.Sequence, want, w.Body.String())
		}
		want++
	}
	if want == 0 {
		t.Fatal("no numbered events")
	}
}

func TestAnAnswerThatOutgrowsTheBudgetIsReportedIncomplete(t *testing.T) {
	long := strings.Repeat("x", 64)
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: long}}},
		{Delta: []llm.Part{&llm.Text{Text: long}}},
		{Stop: llm.StopEnd},
	})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var w wire.BufferWriter
	diags, err := New(Options{MaxBuffer: 80}).EncodeStream(context.Background(), s, wire.Options{}, &w)
	if err != nil {
		t.Fatal(err)
	}
	body := w.Body.String()
	if strings.Count(body, long) < 2 {
		t.Errorf("a delta was withheld:\n%s", body)
	}
	if !strings.Contains(body, "response.incomplete") {
		t.Errorf("the terminal event did not say the object is short:\n%s", body)
	}
	if !strings.Contains(body, `"reason":"max_output_tokens"`) {
		t.Errorf("no reason on the terminal event:\n%s", body)
	}
	if !diags.Has(llm.WireBufferFull) {
		t.Errorf("diagnostics = %v", diags.Codes())
	}
}

func TestAnImageIsNotChargedToTheTextBudget(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Image{ID: "ig_1", MIME: "image/png", Data: make([]byte, 4096)}}},
		{Delta: []llm.Part{&llm.Text{Text: "small"}}},
		{Stop: llm.StopEnd},
	})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var w wire.BufferWriter
	diags, err := New(Options{MaxBuffer: 64}).EncodeStream(context.Background(), s, wire.Options{}, &w)
	if err != nil {
		t.Fatal(err)
	}
	if diags.Has(llm.WireBufferFull) {
		t.Errorf("the picture spent the allowance the text is rebuilt from")
	}
	if !strings.Contains(w.Body.String(), "response.completed") {
		t.Errorf("the answer was reported short:\n%s", w.Body.String())
	}
}

func TestAFailedGenerationIsNotCompleted(t *testing.T) {
	s := llm.StaticStream(llm.StreamMeta{ID: "resp_1", Model: "m"}, []*llm.Frame{
		{Delta: []llm.Part{&llm.Text{Text: "half"}}},
		{Stop: llm.StopError, TurnEnd: true},
	})
	if err := s.Prime(); err != nil {
		t.Fatal(err)
	}
	var w wire.BufferWriter
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{}, &w); err != nil {
		t.Fatal(err)
	}
	body := w.Body.String()
	if strings.Contains(body, "response.completed") {
		t.Errorf("a failed generation was delivered as an answer:\n%s", body)
	}
	if !strings.Contains(body, "response.failed") {
		t.Errorf("the failure was not named:\n%s", body)
	}
}
