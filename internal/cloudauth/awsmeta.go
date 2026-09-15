// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type CredentialSource func(ctx context.Context) (Creds, error)

const (
	containerHost   = "http://169.254.170.2"
	instanceHost    = "http://169.254.169.254"
	metadataTimeout = 5 * time.Second
	tokenTTL        = "21600"
)

type expiring struct {
	fetch func(ctx context.Context) (Creds, time.Time, error)
	now   func() time.Time

	mu      sync.Mutex
	creds   Creds
	expires time.Time
}

func (e *expiring) Get(ctx context.Context) (Creds, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.creds.valid() && e.now().Before(e.expires.Add(-time.Minute)) {
		return e.creds, nil
	}
	ctx, cancel := context.WithTimeout(ctx, metadataTimeout)
	defer cancel()
	creds, expires, err := e.fetch(ctx)
	if err != nil {
		return Creds{}, err
	}
	if expires.IsZero() {
		// No stated life: re-read next call rather than held forever.
		expires = e.now()
	}
	e.creds, e.expires = creds, expires
	return creds, nil
}

type containerCreds struct {
	AccessKeyID     string `json:"AccessKeyId"`
	SecretAccessKey string `json:"SecretAccessKey"`
	Token           string `json:"Token"`
	Expiration      string `json:"Expiration"`
}

func (c containerCreds) creds() (Creds, time.Time) {
	var expires time.Time
	if t, err := time.Parse(time.RFC3339, c.Expiration); err == nil {
		expires = t
	}
	return Creds{
		AccessKeyID:     c.AccessKeyID,
		SecretAccessKey: c.SecretAccessKey,
		SessionToken:    c.Token,
	}, expires
}

func ContainerCredentials(client *http.Client, relative, full, token string, now func() time.Time) CredentialSource {
	url := full
	if relative != "" {
		if !strings.HasPrefix(relative, "/") {
			relative = "/" + relative
		}
		url = containerHost + relative
	}
	if url == "" {
		return nil
	}
	if now == nil {
		now = time.Now
	}
	e := &expiring{now: now, fetch: func(ctx context.Context) (Creds, time.Time, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return Creds{}, time.Time{}, err
		}
		if token != "" {
			req.Header.Set("Authorization", token)
		}
		body, err := read(ctx, client, req)
		if err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws container credentials: %w", err)
		}
		var out containerCreds
		if err := json.Unmarshal(body, &out); err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws container credentials: %w", err)
		}
		creds, expires := out.creds()
		if !creds.valid() {
			return Creds{}, time.Time{}, errors.New("aws container credentials: the endpoint returned no keys")
		}
		return creds, expires, nil
	}}
	return e.Get
}

func InstanceCredentials(client *http.Client, host string, now func() time.Time) CredentialSource {
	if host == "" {
		host = instanceHost
	}
	host = strings.TrimRight(host, "/")
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	if now == nil {
		now = time.Now
	}
	e := &expiring{now: now, fetch: func(ctx context.Context) (Creds, time.Time, error) {
		session, err := imdsToken(ctx, client, host)
		if err != nil {
			return Creds{}, time.Time{}, err
		}
		role, err := imdsGet(ctx, client, host+"/latest/meta-data/iam/security-credentials/", session)
		if err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws instance credentials: %w", err)
		}
		name := strings.TrimSpace(strings.SplitN(string(role), "\n", 2)[0])
		if name == "" {
			return Creds{}, time.Time{}, errors.New("aws instance credentials: no role is attached to this instance")
		}
		body, err := imdsGet(ctx, client, host+"/latest/meta-data/iam/security-credentials/"+name, session)
		if err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws instance credentials: %w", err)
		}
		var out containerCreds
		if err := json.Unmarshal(body, &out); err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws instance credentials: %w", err)
		}
		creds, expires := out.creds()
		if !creds.valid() {
			return Creds{}, time.Time{}, errors.New("aws instance credentials: the role returned no keys")
		}
		return creds, expires, nil
	}}
	return e.Get
}

func imdsToken(ctx context.Context, client *http.Client, host string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, host+"/latest/api/token", nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("X-aws-ec2-metadata-token-ttl-seconds", tokenTTL)
	body, err := read(ctx, client, req)
	if err != nil {
		return "", nil
	}
	return string(body), nil
}

func imdsGet(ctx context.Context, client *http.Client, url, session string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if session != "" {
		req.Header.Set("X-aws-ec2-metadata-token", session)
	}
	return read(ctx, client, req)
}

func read(_ context.Context, client *http.Client, req *http.Request) ([]byte, error) {
	c := client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%d", resp.StatusCode)
	}
	return body, nil
}

func FirstWorkingCredentials(sources ...CredentialSource) CredentialSource {
	return func(ctx context.Context) (Creds, error) {
		var errs []string
		for _, s := range sources {
			if s == nil {
				continue
			}
			creds, err := s(ctx)
			if err == nil && creds.valid() {
				return creds, nil
			}
			if err != nil {
				errs = append(errs, err.Error())
			}
		}
		if len(errs) == 0 {
			return Creds{}, errors.New("aws: no credentials configured")
		}
		return Creds{}, errors.New("aws: no usable credentials: " + strings.Join(errs, "; "))
	}
}
