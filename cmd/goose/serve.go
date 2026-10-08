// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/mcp"
	"github.com/leelsey/goose/route"
)

func gooseTools(client *goose.Client, cfg *Config, logger *slog.Logger) []mcp.Tool {
	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	how := asking{redact: cfg.Server.RedactUpstream, unnamed: unnamed(cfg), log: logger}
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
				return ask(ctx, client, how, args)
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
				return models(ctx, client, how.redact)
			},
		},
	}
}

const askSchema = `{
  "type": "object",
  "properties": {
    "prompt": {"type": "string", "description": "what to ask the model"},
    "model": {"type": "string", "description": "model to route to; left out, the one the proxy's route or provider configures"},
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

// asking is what ask takes from the configuration.
type asking struct {
	redact  bool // redact_upstream_errors
	unnamed string
	log     *slog.Logger
}

// said is what a failed call may be repeated as. A backend's account of its
// failure routinely quotes the request back, and redact_upstream_errors holds
// it back on the listener's other endpoints, so it does here too. goose's own
// refusal, such as a loss the policy would not take or a capability a backend
// lacks, is not an upstream's words, so it keeps its message. What is held back
// goes to the debug log, as the listener's does, so -v can still see it.
func (a asking) said(err error) string {
	e := llm.AsError(err)
	if !a.redact || e.Provider == "" || refusal(e) {
		return err.Error()
	}
	a.log.Debug("upstream failure", "provider", e.Provider, "status", e.Status,
		"code", e.Code, "error", e.Message, "body", e.Body)
	held := *e
	held.Message = "details withheld"
	return held.Error()
}

// failed is a call the tool turns down, answered as the tool's own failure so
// the model that made it reads why, rather than as a fault in the protocol.
func failed(text string) *mcp.Result {
	return &mcp.Result{IsError: true, Content: []llm.Part{&llm.Text{Text: text}}}
}

// refusal says a failure is goose's own rather than an upstream's: a known
// code goose itself raises, no vendor body, and either refused before being
// sent or a loss the policy would not take. It is the rule httpx's listener
// uses to tell the two apart.
func refusal(e *llm.Error) bool {
	if e.Body != "" || !llm.KnownCode(e.Code) {
		return false
	}
	return !e.Sent || e.Kind == llm.KindLoss
}

// unnamed is the backend a call naming no model reaches with no model to ask
// it for, or "" where the route, the provider or the backend supplies one.
// Every wire carries a model, so only ask sends such a call, and a backend
// that addresses models by name refuses it in words that do not say ask could
// have named one.
func unnamed(cfg *Config) string {
	rules, err := rulesOf(cfg)
	if err != nil {
		return ""
	}
	dec, err := (&route.Table{Rules: rules}).Route(&route.Call{})
	if err != nil || len(dec.Targets) == 0 || dec.Targets[0].Model != "" {
		return ""
	}
	id := dec.Targets[0].Provider
	for _, p := range cfg.Providers {
		if (p.ID == id || p.ID == "" && p.Kind == id) && !choosesModel(p) {
			return id
		}
	}
	return ""
}

// choosesModel says whether a backend asked for no model settles on one: an
// agent CLI runs the one its login defaults to, an Azure deployment is named in
// its options, and OpenRouter and a server known only by its shape may serve
// one whatever they are asked for.
func choosesModel(p Provider) bool {
	switch p.Kind {
	case "claude", "codex", "antigravity", "copilot", "openrouter", "compatible":
		return true
	case "azure":
		return p.Options["deployment"] != ""
	}
	return false
}

func ask(ctx context.Context, client *goose.Client, how asking, args json.RawMessage) (*mcp.Result, error) {
	var in struct {
		Prompt      string   `json:"prompt"`
		Model       string   `json:"model"`
		System      string   `json:"system"`
		MaxTokens   *int     `json:"max_tokens"`
		Temperature *float64 `json:"temperature"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return failed("arguments: " + err.Error()), nil
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return failed("prompt is empty"), nil
	}
	if in.Model == "" && how.unnamed != "" {
		return failed(fmt.Sprintf("pass model: without one this call reaches %q, which has no model configured to ask for", how.unnamed)), nil
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
		return answerStreamed(ctx, client, how, req)
	}
	resp, err := client.Chat(ctx, req)
	if err != nil {
		return &mcp.Result{IsError: true, Content: []llm.Part{&llm.Text{Text: how.said(err)}}}, nil
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
	if notes := details(resp.Diagnostics, false); len(notes) > 0 {
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

func answerStreamed(ctx context.Context, client *goose.Client, how asking, req *llm.Request) (*mcp.Result, error) {
	stream, err := client.Stream(ctx, req)
	if err != nil {
		return &mcp.Result{IsError: true, Content: []llm.Part{&llm.Text{Text: how.said(err)}}}, nil
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
		out := how.said(err)
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
	if notes := details(stream.Diagnostics(), false); len(notes) > 0 {
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

// details is what diags tell a caller. named puts the backend before each,
// where several backends were asked and one that failed says only how.
func details(diags llm.Diagnostics, named bool) []string {
	var out []string
	for _, d := range diags {
		if d.Severity == llm.SeverityInfo {
			continue
		}
		if named && d.Provider != "" {
			out = append(out, d.Provider+": "+d.Detail)
			continue
		}
		out = append(out, d.Detail)
	}
	return out
}

func models(ctx context.Context, client *goose.Client, redact bool) (*mcp.Result, error) {
	list, diags := client.Models(ctx, redact)
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
	notes := details(diags, true)
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

// logged writes a line for every call to the tools in list, after it returns.
func logged(list []mcp.Tool, logger *slog.Logger) []mcp.Tool {
	out := make([]mcp.Tool, len(list))
	for i, t := range list {
		call, name := t.Call, t.Definition.Name
		t.Call = func(ctx context.Context, args json.RawMessage) (*mcp.Result, error) {
			started := time.Now()
			result, err := call(ctx, args)
			attrs := []any{"tool", name, "ms", time.Since(started).Milliseconds()}
			switch {
			case err != nil:
				attrs = append(attrs, "error", err.Error())
			case result != nil:
				var served struct {
					Provider string          `json:"provider"`
					Usage    json.RawMessage `json:"usage"`
				}
				_ = json.Unmarshal(result.Structured, &served)
				attrs = append(attrs, "failed", result.IsError)
				if served.Provider != "" {
					attrs = append(attrs, "provider", served.Provider)
				}
				if len(served.Usage) > 0 && string(served.Usage) != "null" {
					attrs = append(attrs, "usage", string(served.Usage))
				}
			}
			logger.Info("tool call", attrs...)
			return result, err
		}
		out[i] = t
	}
	return out
}

func serveStdio(tools *mcp.Manager, stdout, stderr *os.File, origin string) error {
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
	return tools.ServeStdio(ctx, os.Stdin, stdout)
}
