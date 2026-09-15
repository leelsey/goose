// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type listed struct {
	Name        string   `json:"name"`
	DisplayName string   `json:"displayName,omitempty"`
	Methods     []string `json:"supportedGenerationMethods"`
}

func (c *Codec) ModelRoutes() []wire.Route {
	return []wire.Route{{Method: "GET", Pattern: "/v1beta/models"}}
}

func (c *Codec) EncodeModels(models []llm.Model) (*wire.Outbound, error) {
	served := []string{"generateContent", "streamGenerateContent"}
	out := make([]listed, 0, len(models))
	for _, m := range models {
		name := m.ID
		if !strings.HasPrefix(name, "models/") {
			name = "models/" + name
		}
		out = append(out, listed{Name: name, DisplayName: m.Display, Methods: served})
	}
	body, err := json.Marshal(map[string]any{"models": out})
	if err != nil {
		return nil, err
	}
	return &wire.Outbound{
		Status: http.StatusOK,
		Header: http.Header{"Content-Type": []string{"application/json"}},
		Body:   body,
	}, nil
}
