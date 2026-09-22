// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	neturl "net/url"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

func (p *Provider) Models(ctx context.Context) ([]llm.Model, error) {
	var out []llm.Model
	after := ""
	for page := 0; page < 20; page++ {
		url := httpcore.JoinURL(p.opts.BaseURL, "/v1/models?limit=100")
		if after != "" {
			url += "&after_id=" + neturl.QueryEscape(after)
		}
		resp, err := p.call.Do(ctx, httpcore.Request{
			Method: http.MethodGet, URL: url, Header: p.header(nil), Sign: p.opts.Sign,
		})
		if err != nil {
			return nil, err
		}
		var body struct {
			Data []struct {
				ID          string    `json:"id"`
				DisplayName string    `json:"display_name"`
				CreatedAt   time.Time `json:"created_at"`
			} `json:"data"`
			HasMore bool   `json:"has_more"`
			LastID  string `json:"last_id"`
		}
		err = json.NewDecoder(resp.Body).Decode(&body)
		resp.Body.Close()
		if err != nil {
			return nil, httpcore.MalformedList(p.ID(), err)
		}
		for _, m := range body.Data {
			out = append(out, llm.Model{ID: m.ID, Display: m.DisplayName, Owner: p.ID(), Created: m.CreatedAt})
		}
		if !body.HasMore || body.LastID == "" {
			break
		}
		after = body.LastID
	}
	return out, nil
}

func (p *Provider) Probe(ctx context.Context) error {
	return p.call.Reach(ctx, httpcore.Request{
		Method: http.MethodGet,
		URL:    httpcore.JoinURL(p.opts.BaseURL, "/v1/models?limit=1"),
		Header: p.header(nil), Sign: p.opts.Sign,
	})
}
