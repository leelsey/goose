// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollamaapi

import (
	"encoding/json"
	"net/http"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type tag struct {
	Name     string `json:"name"`
	Model    string `json:"model"`
	Modified string `json:"modified_at,omitempty"`
}

func (c *Chat) ModelRoutes() []wire.Route {
	return []wire.Route{{Method: "GET", Pattern: "/api/tags"}}
}

func (c *Chat) EncodeModels(models []llm.Model) (*wire.Outbound, error) {
	return encodeTags(models)
}

func (g *Generate) ModelRoutes() []wire.Route {
	return []wire.Route{{Method: "GET", Pattern: "/api/tags"}}
}

func (g *Generate) EncodeModels(models []llm.Model) (*wire.Outbound, error) {
	return encodeTags(models)
}

func encodeTags(models []llm.Model) (*wire.Outbound, error) {
	tags := make([]tag, 0, len(models))
	for _, m := range models {
		t := tag{Name: m.ID, Model: m.ID}
		if !m.Created.IsZero() {
			t.Modified = stamp(m.Created)
		}
		tags = append(tags, t)
	}
	body, err := json.Marshal(map[string]any{"models": tags})
	if err != nil {
		return nil, err
	}
	return &wire.Outbound{
		Status: http.StatusOK,
		Header: http.Header{"Content-Type": []string{"application/json"}},
		Body:   body,
	}, nil
}
