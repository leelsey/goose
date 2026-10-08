// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"errors"
	"net/http"

	"github.com/leelsey/goose/internal/cloudauth"
)

type CredentialSource func(ctx context.Context) (Credentials, error)

func StaticCredentials(c Credentials) CredentialSource {
	return func(context.Context) (Credentials, error) { return c, nil }
}

func ContainerCredentials(client *http.Client, relative, full, token string) CredentialSource {
	return container(cloudauth.ContainerCredentials(client, relative, full, token, "", nil))
}

// ContainerCredentialsFromFile reads the endpoint's token from tokenFile at every
// refresh, as AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE names it: an EKS pod
// identity is given only the file, and the token in it is rotated while the
// process runs.
func ContainerCredentialsFromFile(client *http.Client, relative, full, tokenFile string) CredentialSource {
	return container(cloudauth.ContainerCredentials(client, relative, full, "", tokenFile, nil))
}

// container answers where neither URI names an endpoint too, as outside a
// container neither is set: a nil source would be called by whoever holds it,
// where an error lets FirstWorkingCredentials go on to the next.
func container(s cloudauth.CredentialSource) CredentialSource {
	if s == nil {
		return func(context.Context) (Credentials, error) {
			return Credentials{}, errors.New("aws container credentials: no endpoint is named, relative or full")
		}
	}
	return adapt(s)
}

// WebIdentityCredentials assumes roleARN with the token in tokenFile, as
// AWS_ROLE_ARN and AWS_WEB_IDENTITY_TOKEN_FILE name them for an EKS service
// account's role (IRSA). The file is read at every refresh, since the token in
// it is rotated while the process runs. endpoint is STS's own, such as
// https://sts.us-east-1.amazonaws.com, and the global one where empty; an empty
// sessionName is chosen at each exchange.
func WebIdentityCredentials(client *http.Client, endpoint, roleARN, tokenFile, sessionName string) CredentialSource {
	return adapt(cloudauth.WebIdentityCredentials(client, endpoint, roleARN, tokenFile, sessionName, nil))
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
