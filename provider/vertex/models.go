// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package vertex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"

	"github.com/leelsey/goose/internal/httpcore"
	"github.com/leelsey/goose/llm"
)

// garden wraps a publisher's surface with the catalogue, which Model Garden
// keeps beside the project rather than inside it: the path that answers a call
// serves no listing, so asking it leaves every Vertex backend reading as
// unready.
type garden struct {
	llm.Provider
	list   string
	header http.Header
	sign   func(ctx context.Context, r *http.Request, body []byte) error
	call   *httpcore.Caller
}

func (g *garden) Bind(env llm.Env) error {
	if env.HTTP != nil {
		g.call.HTTP = env.HTTP
	}
	if b, ok := g.Provider.(llm.Binder); ok {
		return b.Bind(env)
	}
	return nil
}

func (g *garden) Complete(ctx context.Context, a *llm.Attempt) (*llm.Response, error) {
	resp, err := g.Provider.Complete(ctx, a)
	if err != nil {
		refine(llm.AsError(err))
	}
	return resp, err
}

func (g *garden) Stream(ctx context.Context, a *llm.Attempt) (*llm.Stream, error) {
	s, err := g.Provider.Stream(ctx, a)
	if err != nil {
		refine(llm.AsError(err))
	}
	return s, err
}

// refine reads what the status does not say. Vertex refuses a model it will
// not serve this project or region with the 400 a malformed request gets as
// often as with a 404, and only a malformed request fails the same on every
// backend.
func refine(e *llm.Error) {
	if e.Kind != llm.KindInvalidRequest {
		return
	}
	said := strings.ToLower(e.Message)
	if !strings.Contains(said, "publisher model") {
		return
	}
	for _, unserved := range []string{"not found", "not allowed to use", "not servable", "does not have access"} {
		if strings.Contains(said, unserved) {
			e.Kind = llm.KindNotFound
			return
		}
	}
}

func (g *garden) Pass(ctx context.Context, r *llm.PassRequest) (*llm.PassResponse, error) {
	passer, ok := g.Provider.(llm.Passer)
	if !ok {
		return nil, llm.Errorf(llm.KindUnsupported, "vertex: this surface cannot forward opaque requests")
	}
	return passer.Pass(ctx, r)
}

func (g *garden) request(query string) httpcore.Request {
	return httpcore.Request{
		Method: http.MethodGet, URL: g.list + "?" + query, Header: g.header, Sign: g.sign,
	}
}

func (g *garden) Models(ctx context.Context) ([]llm.Model, error) {
	var out []llm.Model
	token := ""
	for page := 0; page < 20; page++ {
		query := "pageSize=200"
		if token != "" {
			query += "&pageToken=" + url.QueryEscape(token)
		}
		resp, err := g.call.Do(ctx, g.request(query))
		if err != nil {
			return nil, err
		}
		var body struct {
			PublisherModels []struct {
				Name string `json:"name"`
			} `json:"publisherModels"`
			NextPageToken string `json:"nextPageToken"`
		}
		err = json.NewDecoder(resp.Body).Decode(&body)
		resp.Body.Close()
		if err != nil {
			return nil, httpcore.MalformedList(g.ID(), err)
		}
		for _, m := range body.PublisherModels {
			// A name arrives qualified — "publishers/google/models/gemini-..." —
			// and the publisher is already the provider's own.
			if id := lastSegment(m.Name); id != "" {
				out = append(out, llm.Model{ID: id, Owner: g.ID()})
			}
		}
		if body.NextPageToken == "" {
			break
		}
		token = body.NextPageToken
	}
	return out, nil
}

func (g *garden) Probe(ctx context.Context) error {
	return g.call.Reach(ctx, g.request("pageSize=1"))
}
