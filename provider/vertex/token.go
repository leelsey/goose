// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package vertex

import (
	"context"
	"net/http"

	"github.com/leelsey/goose/internal/cloudauth"
)

type TokenSource interface {
	Token(ctx context.Context) (string, error)
}

type TokenFunc func(ctx context.Context) (string, error)

func (f TokenFunc) Token(ctx context.Context) (string, error) { return f(ctx) }

func StaticToken(token string) TokenSource { return cloudauth.StaticToken(token) }

func ServiceAccountToken(keyJSON []byte, client *http.Client, scopes ...string) (TokenSource, string, error) {
	key, err := cloudauth.ParseServiceAccountKey(keyJSON)
	if err != nil {
		return nil, "", err
	}
	source, err := cloudauth.ServiceAccountSource(key, client, scopes, nil)
	if err != nil {
		return nil, "", err
	}
	return source, key.ProjectID, nil
}

func MetadataToken(client *http.Client, host string) TokenSource {
	return cloudauth.MetadataSource(client, host, nil)
}

func FirstWorkingToken(sources ...TokenSource) TokenSource {
	inner := make([]cloudauth.TokenSource, 0, len(sources))
	for _, s := range sources {
		if s == nil {
			continue
		}
		inner = append(inner, cloudauth.TokenFunc(s.Token))
	}
	return cloudauth.FirstWorkingSource(inner...)
}
