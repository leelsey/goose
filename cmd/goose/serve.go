// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/mcp"
	"github.com/leelsey/goose/usage"
)

func gooseTools(client *goose.Client) []mcp.Tool {
	return []mcp.Tool{
		{
			Definition: llm.Tool{
				Name: "ask",
				Description: "Ask a model a question through goose. The proxy's routes choose the provider, " +
					"and a chain that falls back does so here too. Naming a model asks for that one.",
				Schema:       json.RawMessage(askSchema),
				OutputSchema: json.RawMessage(askResult),
			},
			Call: func(ctx context.Context, args json.RawMessage) (*mcp.Result, error) {
				return ask(ctx, client, args)
			},
		},
		{
			Definition: llm.Tool{
				Name:         "models",
				Description:  "List the models this proxy can route to, with the provider that serves each.",
				Schema:       json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
				OutputSchema: json.RawMessage(modelsResult),
			},
			Call: func(ctx context.Context, args json.RawMessage) (*mcp.Result, error) {
				return models(ctx, client)
			},
		},
	}
}

const askSchema = `{
  "type": "object",
  "properties": {
    "prompt": {"type": "string", "description": "what to ask the model"},
    "model": {"type": "string", "description": "model to route to; left out, the proxy's routes decide"},
    "system": {"type": "string", "description": "instructions that precede the prompt"},
    "max_tokens": {"type": "integer", "minimum": 1},
    "temperature": {"type": "number", "minimum": 0, "maximum": 2}
  },
  "required": ["prompt"],
  "additionalProperties": false
}`

const askResult = `{
  "type": "object",
  "properties": {
    "text": {"type": "string"},
    "model": {"type": "string"},
    "provider": {"type": "string"},
    "stop": {"type": "string"},
    "usage": {"type": "object"},
    "notes": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["text", "model"]
}`

const modelsResult = `{
  "type": "object",
  "properties": {
    "models": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "owner": {"type": "string"}, "display": {"type": "string"}},
        "required": ["id"]
      }
    },
    "notes": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["models"]
}`

func ask(ctx context.Context, client *goose.Client, args json.RawMessage) (*mcp.Result, error) {
	var in struct {
		Prompt      string   `json:"prompt"`
		Model       string   `json:"model"`
		System      string   `json:"system"`
		MaxTokens   *int     `json:"max_tokens"`
		Temperature *float64 `json:"temperature"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("arguments: %w", err)
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return nil, fmt.Errorf("prompt is empty")
	}
	req := &llm.Request{
		Model:           in.Model,
		Messages:        []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: in.Prompt}}}},
		MaxOutputTokens: in.MaxTokens,
		Temperature:     in.Temperature,
	}
	if in.System != "" {
		req.System = []llm.Part{&llm.Text{Text: in.System}}
	}
	if mcp.Watched(ctx) {
		return answerStreamed(ctx, client, req)
	}
	resp, err := client.Chat(ctx, req)
	if err != nil {
		return &mcp.Result{IsError: true, Content: []llm.Part{&llm.Text{Text: err.Error()}}}, nil
	}
	var text string
	stop := resp.Stop
	if len(resp.Choices) > 0 {
		text = llm.TextOf(resp.Choices[0].Message.Content)
		if resp.Choices[0].Stop != llm.StopNone {
			stop = resp.Choices[0].Stop
		}
	}
	fields := map[string]any{
		"text": text, "model": resp.Model, "provider": resp.Provider,
		"stop": string(stop), "usage": resp.Usage,
	}
	if notes := details(resp.Diagnostics); len(notes) > 0 {
		fields["notes"] = notes
	}
	structured, _ := json.Marshal(fields)
	if text == "" {
		return &mcp.Result{
			IsError:    true,
			Content:    []llm.Part{&llm.Text{Text: "the model returned no text (stopped: " + stopOr(stop) + ")"}},
			Structured: structured,
		}, nil
	}
	return &mcp.Result{
		Content:    []llm.Part{&llm.Text{Text: text}},
		Structured: structured,
	}, nil
}

func answerStreamed(ctx context.Context, client *goose.Client, req *llm.Request) (*mcp.Result, error) {
	stream, err := client.Stream(ctx, req)
	if err != nil {
		return &mcp.Result{IsError: true, Content: []llm.Part{&llm.Text{Text: err.Error()}}}, nil
	}
	defer stream.Close()

	var text strings.Builder
	stop := llm.StopNone
	for frame := range stream.Frames() {
		if part := llm.TextOf(frame.Delta); part != "" {
			text.WriteString(part)
			mcp.Report(ctx, mcp.Progress{Done: float64(text.Len())})
		}
		if frame.Stop != llm.StopNone {
			stop = frame.Stop
		}
	}
	if err := stream.Err(); err != nil {
		out := err.Error()
		if text.Len() > 0 {
			out = text.String() + "\n\n" + out
		}
		return &mcp.Result{IsError: true, Content: []llm.Part{&llm.Text{Text: out}}}, nil
	}

	meta := stream.Meta()
	fields := map[string]any{
		"text": text.String(), "model": meta.Model, "provider": meta.Provider,
		"stop": string(stop), "usage": stream.Usage(),
	}
	if notes := details(stream.Diagnostics()); len(notes) > 0 {
		fields["notes"] = notes
	}
	structured, _ := json.Marshal(fields)
	if text.Len() == 0 {
		return &mcp.Result{
			IsError:    true,
			Content:    []llm.Part{&llm.Text{Text: "the model returned no text (stopped: " + stopOr(stop) + ")"}},
			Structured: structured,
		}, nil
	}
	return &mcp.Result{
		Content:    []llm.Part{&llm.Text{Text: text.String()}},
		Structured: structured,
	}, nil
}

func stopOr(s llm.StopReason) string {
	if s == llm.StopNone {
		return "no reason given"
	}
	return string(s)
}

func details(diags llm.Diagnostics) []string {
	var out []string
	for _, d := range diags {
		if d.Severity == llm.SeverityInfo {
			continue
		}
		out = append(out, d.Detail)
	}
	return out
}

func models(ctx context.Context, client *goose.Client) (*mcp.Result, error) {
	list, diags := client.Models(ctx)
	out := make([]map[string]string, 0, len(list))
	var lines []string
	for _, m := range list {
		entry := map[string]string{"id": m.ID}
		if m.Owner != "" {
			entry["owner"] = m.Owner
		}
		if m.Display != "" {
			entry["display"] = m.Display
		}
		out = append(out, entry)
		lines = append(lines, strings.TrimSpace(m.ID+"  "+m.Owner))
	}
	notes := details(diags)
	fields := map[string]any{"models": out}
	if len(notes) > 0 {
		fields["notes"] = notes
	}
	structured, _ := json.Marshal(fields)
	text := strings.Join(lines, "\n")
	if text == "" {
		text = "no models"
	}
	if len(notes) > 0 {
		text += "\n\n" + strings.Join(notes, "\n")
	}
	return &mcp.Result{Content: []llm.Part{&llm.Text{Text: text}}, Structured: structured}, nil
}

func serveStdio(tools *mcp.Manager, stdout, stderr *os.File, cfg *Config, origin string, mem *usage.Memory) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Fprintf(stderr, "goose %s  config %s\n", version, origin)
	for _, s := range tools.Servers() {
		fmt.Fprintf(stderr, "  tools    %-12s %-24s %-11s %d tools",
			s.Name, dashIfEmpty(s.Info.Name), dashIfEmpty(s.Info.ProtocolVersion), s.Tools)
		if s.Truncated {
			fmt.Fprint(stderr, " (and more than goose will hold)")
		}
		if n := len(s.Refused); n > 0 {
			fmt.Fprintf(stderr, " (%d refused: %s)", n, strings.Join(s.Refused, ", "))
		}
		fmt.Fprintln(stderr)
	}
	err := tools.ServeStdio(ctx, os.Stdin, stdout)
	if mem != nil && cfg.Usage.Summary {
		summary(stderr, mem)
	}
	return err
}
