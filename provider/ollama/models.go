// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollama

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
		Method: http.MethodGet, URL: httpcore.JoinURL(p.opts.BaseURL, "/api/tags"), Header: p.header(),
	})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var body struct {
		Models []struct {
			Name       string    `json:"name"`
			Model      string    `json:"model"`
			ModifiedAt time.Time `json:"modified_at"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, httpcore.MalformedList(p.ID(), err)
	}
	out := make([]llm.Model, 0, len(body.Models))
	for _, m := range body.Models {
		id := m.Name
		if id == "" {
			id = m.Model
		}
		out = append(out, llm.Model{ID: id, Owner: p.ID(), Created: m.ModifiedAt})
	}
	return out, nil
}

func (p *Provider) Probe(ctx context.Context) error {
	return p.call.Reach(ctx, httpcore.Request{
		Method: http.MethodGet, URL: httpcore.JoinURL(p.opts.BaseURL, "/api/tags"), Header: p.header(),
	})
}
