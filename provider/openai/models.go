// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

func (p *Provider) Models(ctx context.Context) ([]llm.Model, error) {
	resp, err := p.call.Do(ctx, httpcore.Request{
		Method: http.MethodGet, URL: p.endpoint("/models", ""), Header: p.header(nil), Sign: p.opts.Sign,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var body struct {
		Data []struct {
			ID      string `json:"id"`
			Created int64  `json:"created"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, httpcore.MalformedList(p.ID(), err)
	}
	out := make([]llm.Model, 0, len(body.Data))
	for _, m := range body.Data {
		model := llm.Model{ID: m.ID, Owner: m.OwnedBy}
		if model.Owner == "" {
			model.Owner = p.ID()
		}
		if m.Created > 0 {
			model.Created = time.Unix(m.Created, 0)
		}
		out = append(out, model)
	}
	return out, nil
}

func (p *Provider) Probe(ctx context.Context) error {
	return p.call.Reach(ctx, httpcore.Request{
		Method: http.MethodGet, URL: p.endpoint("/models", ""), Header: p.header(nil), Sign: p.opts.Sign,
	})
}
