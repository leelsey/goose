// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wire_test

import (
	"net/http"
	"net/url"
	"testing"

	"github.com/leelsey/goose/wire"
	"github.com/leelsey/goose/wire/anthropicmsg"
	"github.com/leelsey/goose/wire/geminiapi"
	"github.com/leelsey/goose/wire/ollamaapi"
	"github.com/leelsey/goose/wire/openaichat"
	"github.com/leelsey/goose/wire/openairesp"
)

func codecs() []wire.Codec {
	return []wire.Codec{
		openaichat.New(openaichat.Options{}),
		openairesp.New(openairesp.Options{}),
		anthropicmsg.New(anthropicmsg.Options{}),
		geminiapi.New(geminiapi.Options{}),
		ollamaapi.NewChat(ollamaapi.Options{}),
		ollamaapi.NewGenerate(ollamaapi.Options{}),
	}
}

func FuzzDecode(f *testing.F) {
	for _, seed := range []string{
		`{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
		`{"model":"m","messages":[{"role":"assistant","content":[{"type":"tool_use","id":"1","name":"x","input":{}}]}]}`,
		`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`,
		`{"input":[{"role":"user","content":"hi"}],"model":"m"}`,
		`{"model":"m","prompt":"hi"}`,
		`{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}`,
		`{"messages":[{"role":"user","content":null}],"n":99999999}`,
		`{"model":"m","messages":[],"tools":[{"type":"function","function":{"name":"a","parameters":{}}}]}`,
		`[]`, `null`, `0`, `""`, `{}`,
	} {
		f.Add(seed)
	}
	all := codecs()
	f.Fuzz(func(t *testing.T, body string) {
		for _, codec := range all {
			for _, r := range codec.Routes() {
				in := &wire.Inbound{
					Method: r.Method,
					Path:   pathFor(r.Pattern),
					Query:  url.Values{},
					Header: http.Header{},
					Body:   []byte(body),
				}
				req, _, _, err := codec.Decode(in)
				if err != nil {
					continue
				}
				if req == nil {
					t.Fatalf("%s: nil request with nil error", codec.Name())
				}
			}
		}
	})
}

func pathFor(pattern string) string {
	if pattern == "POST /v1beta/models/" || pattern == "/v1beta/models/" {
		return "/v1beta/models/m:generateContent"
	}
	return pattern
}
