// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wire_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openaichat"
	"github.com/leelsey/goose/wire/openairesp"
)

func TestEveryWireAgreesOnAStatus(t *testing.T) {
	want := map[llm.ErrorKind]int{
		llm.KindInvalidRequest: 400,
		llm.KindUnsupported:    422,
		llm.KindLoss:           400,
		llm.KindAuth:           401,
		llm.KindPermission:     403,
		llm.KindNotFound:       404,
		llm.KindNoRoute:        404,
		llm.KindRateLimit:      429,
		llm.KindOverloaded:     503,
		llm.KindSaturated:      503,
		llm.KindTimeout:        504,
		llm.KindCancelled:      wire.StatusClientClosed,
	}
	codecs := []wire.Codec{
		openaichat.New(openaichat.Options{}),
		openairesp.New(openairesp.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
		geminiapi.New(geminiapi.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
		ollamaapi.NewGenerate(ollamaapi.Options{}),
	}
	for kind, status := range want {
		for _, c := range codecs {
			out := c.EncodeError(&llm.Error{Kind: kind, Message: "x"}, wire.Options{})
			if out == nil {
				t.Fatalf("%s: %s produced no body", c.Name(), kind)
			}
			if out.Status != status {
				t.Errorf("%s: %s -> %d, every wire says %d", c.Name(), kind, out.Status, status)
			}
			if len(out.Body) == 0 {
				t.Errorf("%s: %s produced an empty body", c.Name(), kind)
			}
		}
	}
}

func TestAnUpstreamStatusSurvives(t *testing.T) {
	for _, c := range []wire.Codec{
		openaichat.New(openaichat.Options{}),
		openairesp.New(openairesp.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
		geminiapi.New(geminiapi.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
	} {
		out := c.EncodeError(&llm.Error{Kind: llm.KindUpstream, Status: 418, Message: "x"}, wire.Options{})
		if out.Status != 418 {
			t.Errorf("%s: status = %d, want the one the upstream gave", c.Name(), out.Status)
		}
	}
}

func TestAFailureWithNoCodeCarriesNone(t *testing.T) {
	for _, c := range []wire.Codec{
		openaichat.New(openaichat.Options{}),
		openairesp.New(openairesp.Options{}),
	} {
		out := c.EncodeError(&llm.Error{Kind: llm.KindAuth, Message: "no"}, wire.Options{})
		var body struct {
			Error struct {
				Code *string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(out.Body, &body); err != nil {
			t.Fatalf("%s: %v", c.Name(), err)
		}
		if body.Error.Code != nil {
			t.Errorf("%s: code = %q, want null", c.Name(), *body.Error.Code)
		}

		out = c.EncodeError(&llm.Error{Kind: llm.KindAuth, Code: "invalid_api_key", Message: "no"}, wire.Options{})
		if err := json.Unmarshal(out.Body, &body); err != nil {
			t.Fatalf("%s: %v", c.Name(), err)
		}
		if body.Error.Code == nil || *body.Error.Code != "invalid_api_key" {
			t.Errorf("%s: a code the backend gave must survive", c.Name())
		}
	}
}

func TestEveryWireCarriesTheUpstreamsWait(t *testing.T) {
	for _, c := range []wire.Codec{
		openaichat.New(openaichat.Options{}),
		openairesp.New(openairesp.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
		geminiapi.New(geminiapi.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
		ollamaapi.NewGenerate(ollamaapi.Options{}),
	} {
		out := c.EncodeError(&llm.Error{
			Kind: llm.KindRateLimit, Message: "slow down", RetryAfter: 30 * time.Second,
		}, wire.Options{})
		if got := out.Header.Get("Retry-After"); got != "30" {
			t.Errorf("%s: Retry-After = %q, want 30", c.Name(), got)
		}
		out = c.EncodeError(&llm.Error{Kind: llm.KindRateLimit, Message: "slow down"}, wire.Options{})
		if got := out.Header.Get("Retry-After"); got != "" {
			t.Errorf("%s: Retry-After = %q with no wait to report", c.Name(), got)
		}
	}
}

func TestTheTierTravelsWhereTheWireHasAFieldForIt(t *testing.T) {
	resp := &llm.Response{
		Model:   "m",
		Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{&llm.Text{Text: "hi"}}}, Stop: llm.StopEnd}},
		Usage:   llm.Usage{Prompt: 10, Completion: 1, Tier: "flex", Reported: true},
	}
	for _, c := range []wire.Codec{
		openaichat.New(openaichat.Options{}),
		openairesp.New(openairesp.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
	} {
		out, diags, err := c.EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatalf("%s: %v", c.Name(), err)
		}
		if !strings.Contains(string(out.Body), `"service_tier":"flex"`) {
			t.Errorf("%s: body = %s", c.Name(), out.Body)
		}
		for _, d := range diags {
			if d.Code == llm.AcctFigureUncarried && d.Field == llm.FigureTier {
				t.Errorf("%s: reported a tier it wrote", c.Name())
			}
		}
	}

	for _, c := range []wire.Codec{
		geminiapi.New(geminiapi.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
	} {
		_, diags, err := c.EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatalf("%s: %v", c.Name(), err)
		}
		found := false
		for _, d := range diags {
			if d.Code == llm.AcctFigureUncarried && d.Field == llm.FigureTier {
				found = true
			}
		}
		if !found {
			t.Errorf("%s: diagnostics = %+v: this wire has no field for a tier", c.Name(), diags)
		}
	}
}
