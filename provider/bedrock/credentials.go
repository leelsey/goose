// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"net/http"

	"github.com/leelsey/goose/internal/cloudauth"
)

type CredentialSource func(ctx context.Context) (Credentials, error)

func StaticCredentials(c Credentials) CredentialSource {
	return func(context.Context) (Credentials, error) { return c, nil }
}

func ContainerCredentials(client *http.Client, relative, full, token string) CredentialSource {
	return adapt(cloudauth.ContainerCredentials(client, relative, full, token, nil))
}

func InstanceCredentials(client *http.Client, host string) CredentialSource {
	return adapt(cloudauth.InstanceCredentials(client, host, nil))
}

func FirstWorkingCredentials(sources ...CredentialSource) CredentialSource {
	inner := make([]cloudauth.CredentialSource, 0, len(sources))
	for _, s := range sources {
		if s == nil {
			continue
		}
		inner = append(inner, func(ctx context.Context) (cloudauth.Creds, error) {
			c, err := s(ctx)
			return cloudauth.Creds(c), err
		})
	}
	return adapt(cloudauth.FirstWorkingCredentials(inner...))
}

func adapt(s cloudauth.CredentialSource) CredentialSource {
	return func(ctx context.Context) (Credentials, error) {
		c, err := s(ctx)
		return Credentials(c), err
	}
}
