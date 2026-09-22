// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"context"
	"encoding/json"
	"net/http"
	neturl "net/url"
	"strings"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

func (p *Provider) Models(ctx context.Context) ([]llm.Model, error) {
	var out []llm.Model
	token := ""
	for page := 0; page < 20; page++ {
		url := httpcore.JoinURL(p.opts.BaseURL, p.opts.Prefix) + "/models?pageSize=200"
		if token != "" {
			url += "&pageToken=" + neturl.QueryEscape(token)
		}
		resp, err := p.call.Do(ctx, httpcore.Request{
			Method: http.MethodGet, URL: url, Header: p.header(nil), Sign: p.opts.Sign,
		})
		if err != nil {
			return nil, err
		}
		var body struct {
			Models []struct {
				Name        string `json:"name"`
				DisplayName string `json:"displayName"`
			} `json:"models"`
			NextPageToken string `json:"nextPageToken"`
		}
		err = json.NewDecoder(resp.Body).Decode(&body)
		resp.Body.Close()
		if err != nil {
			return nil, httpcore.MalformedList(p.ID(), err)
		}
		for _, m := range body.Models {
			out = append(out, llm.Model{
				ID: strings.TrimPrefix(m.Name, "models/"), Display: m.DisplayName, Owner: p.ID(),
			})
		}
		if body.NextPageToken == "" {
			break
		}
		token = body.NextPageToken
	}
	return out, nil
}

func (p *Provider) Probe(ctx context.Context) error {
	return p.call.Reach(ctx, httpcore.Request{
		Method: http.MethodGet,
		URL:    httpcore.JoinURL(p.opts.BaseURL, p.opts.Prefix) + "/models?pageSize=1",
		Header: p.header(nil), Sign: p.opts.Sign,
	})
}
