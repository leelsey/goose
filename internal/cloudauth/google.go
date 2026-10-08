// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

type TokenSource interface {
	Token(ctx context.Context) (string, error)
}

type TokenFunc func(ctx context.Context) (string, error)

func (f TokenFunc) Token(ctx context.Context) (string, error) { return f(ctx) }

func StaticToken(token string) TokenSource {
	return TokenFunc(func(context.Context) (string, error) {
		if token == "" {
			return "", errors.New("google: empty token")
		}
		return token, nil
	})
}

type ServiceAccountKey struct {
	Type        string `json:"type"`
	ProjectID   string `json:"project_id"`
	PrivateKey  string `json:"private_key"`
	ClientEmail string `json:"client_email"`
	TokenURI    string `json:"token_uri"`
}

func ParseServiceAccountKey(data []byte) (*ServiceAccountKey, error) {
	var key ServiceAccountKey
	if err := json.Unmarshal(data, &key); err != nil {
		return nil, fmt.Errorf("google: service account key: %w", err)
	}
	if key.ClientEmail == "" || key.PrivateKey == "" {
		return nil, errors.New("google: service account key is missing client_email or private_key")
	}
	if key.TokenURI == "" {
		key.TokenURI = "https://oauth2.googleapis.com/token"
	}
	return &key, nil
}

const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

const exchangeTimeout = 30 * time.Second

func ServiceAccountSource(key *ServiceAccountKey, client *http.Client, scopes []string, now func() time.Time) (TokenSource, error) {
	priv, err := parsePrivateKey(key.PrivateKey)
	if err != nil {
		return nil, err
	}
	if len(scopes) == 0 {
		scopes = []string{cloudPlatformScope}
	}
	if now == nil {
		now = time.Now
	}
	s := &saSource{key: key, priv: priv, client: client, scope: strings.Join(scopes, " "), now: now}
	return s, nil
}

type saSource struct {
	key    *ServiceAccountKey
	priv   *rsa.PrivateKey
	client *http.Client
	scope  string
	now    func() time.Time

	mu       sync.Mutex
	token    string
	expires  time.Time
	inFlight chan struct{}
}

// A token exchange already under way is waited on rather than repeated, and
// every caller, the one that started it included, waits on its own context.
// Holding the lock across the exchange would queue every request behind it with
// no way to give up, and would let whichever one started it cancel the token
// they are all waiting for.
func (s *saSource) Token(ctx context.Context) (string, error) {
	for {
		s.mu.Lock()
		if s.token != "" && s.now().Before(s.expires.Add(-time.Minute)) {
			token := s.token
			s.mu.Unlock()
			return token, nil
		}
		if wait := s.inFlight; wait != nil {
			s.mu.Unlock()
			select {
			case <-wait:
				continue
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
		done := make(chan struct{})
		s.inFlight = done
		s.mu.Unlock()

		return detach(ctx, func() (string, error) {
			token, _, err := s.refresh(ctx, done)
			return token, err
		})
	}
}

// refresh always clears the in-flight marker and wakes the waiters, including
// when the exchange panics: leaving either behind parks every future caller on
// a refresh that will never finish.
func (s *saSource) refresh(ctx context.Context, done chan struct{}) (token string, expires time.Time, err error) {
	defer func() {
		s.mu.Lock()
		s.inFlight = nil
		if err == nil {
			s.token, s.expires = token, expires
		}
		s.mu.Unlock()
		close(done)
	}()
	return s.exchange(ctx)
}

func (s *saSource) exchange(parent context.Context) (string, time.Time, error) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), exchangeTimeout)
	defer cancel()

	assertion, err := s.assertion()
	if err != nil {
		return "", time.Time{}, err
	}
	form := url.Values{
		"grant_type": {"urn:ietf:params:oauth:grant-type:jwt-bearer"},
		"assertion":  {assertion},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.key.TokenURI, strings.NewReader(form.Encode()))
	if err != nil {
		return "", time.Time{}, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	client := s.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("google: token exchange: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if resp.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("google: token exchange: %d %s", resp.StatusCode, oauthError(body))
	}
	var out struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return "", time.Time{}, fmt.Errorf("google: token exchange: %w", err)
	}
	if out.AccessToken == "" {
		return "", time.Time{}, errors.New("google: token exchange returned no token")
	}
	return out.AccessToken, s.now().Add(time.Duration(out.ExpiresIn) * time.Second), nil
}

func (s *saSource) assertion() (string, error) {
	issued := s.now().Unix()
	header := map[string]any{"alg": "RS256", "typ": "JWT"}
	claims := map[string]any{
		"iss":   s.key.ClientEmail,
		"scope": s.scope,
		"aud":   s.key.TokenURI,
		"iat":   issued,
		"exp":   issued + 3600,
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	cb, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	signing := b64(hb) + "." + b64(cb)
	sum := sha256.Sum256([]byte(signing))
	sig, err := rsa.SignPKCS1v15(rand.Reader, s.priv, crypto.SHA256, sum[:])
	if err != nil {
		return "", err
	}
	return signing + "." + b64(sig), nil
}

func b64(b []byte) string { return base64.RawURLEncoding.EncodeToString(b) }

func parsePrivateKey(pemKey string) (*rsa.PrivateKey, error) {
	block, _ := pem.Decode([]byte(pemKey))
	if block == nil {
		return nil, errors.New("google: private key is not PEM")
	}
	if key, err := x509.ParsePKCS8PrivateKey(block.Bytes); err == nil {
		rsaKey, ok := key.(*rsa.PrivateKey)
		if !ok {
			return nil, errors.New("google: private key is not RSA")
		}
		return rsaKey, nil
	}
	return x509.ParsePKCS1PrivateKey(block.Bytes)
}

// host is spelt the way GCE_METADATA_HOST is, as Google's own libraries read
// it: a host and an optional port, with the scheme left to the reader. A full
// URL is still taken as one.
func MetadataSource(client *http.Client, host string, now func() time.Time) TokenSource {
	if host == "" {
		host = "metadata.google.internal"
	}
	host = strings.TrimRight(host, "/")
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	if now == nil {
		now = time.Now
	}
	return &metaSource{client: client, host: host, now: now}
}

type metaSource struct {
	client *http.Client
	host   string
	now    func() time.Time

	mu       sync.Mutex
	token    string
	expires  time.Time
	inFlight chan struct{}
}

func (m *metaSource) Token(ctx context.Context) (string, error) {
	for {
		m.mu.Lock()
		if m.token != "" && m.now().Before(m.expires.Add(-time.Minute)) {
			token := m.token
			m.mu.Unlock()
			return token, nil
		}
		if wait := m.inFlight; wait != nil {
			m.mu.Unlock()
			select {
			case <-wait:
				continue
			case <-ctx.Done():
				return "", ctx.Err()
			}
		}
		done := make(chan struct{})
		m.inFlight = done
		m.mu.Unlock()

		return detach(ctx, func() (string, error) {
			token, _, err := m.refresh(ctx, done)
			return token, err
		})
	}
}

func (m *metaSource) refresh(parent context.Context, done chan struct{}) (token string, expires time.Time, err error) {
	defer func() {
		m.mu.Lock()
		m.inFlight = nil
		if err == nil {
			m.token, m.expires = token, expires
		}
		m.mu.Unlock()
		close(done)
	}()

	ctx, cancel := context.WithTimeout(context.WithoutCancel(parent), exchangeTimeout)
	defer cancel()

	url := m.host + "/computeMetadata/v1/instance/service-accounts/default/token"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", time.Time{}, err
	}
	req.Header.Set("Metadata-Flavor", "Google")
	c := m.client
	if c == nil {
		c = http.DefaultClient
	}
	resp, err := c.Do(req)
	if err != nil {
		return "", time.Time{}, fmt.Errorf("google: metadata token: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
	if resp.StatusCode != http.StatusOK {
		return "", time.Time{}, fmt.Errorf("google: metadata token: %d %s", resp.StatusCode, oauthError(body))
	}
	var out struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return "", time.Time{}, err
	}
	if out.AccessToken == "" {
		return "", time.Time{}, errors.New("google: metadata returned no token")
	}
	return out.AccessToken, m.now().Add(time.Duration(out.ExpiresIn) * time.Second), nil
}

func FirstWorkingSource(sources ...TokenSource) TokenSource {
	return TokenFunc(func(ctx context.Context) (string, error) {
		var errs []string
		for _, s := range sources {
			if s == nil {
				continue
			}
			token, err := s.Token(ctx)
			if err == nil && token != "" {
				return token, nil
			}
			if err != nil {
				errs = append(errs, err.Error())
			}
		}
		if len(errs) == 0 {
			return "", errors.New("google: no credentials configured")
		}
		return "", errors.New("google: no usable credentials: " + strings.Join(errs, "; "))
	})
}

func oauthError(body []byte) string {
	var out struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(body, &out) == nil && out.Error != "" {
		return out.Error
	}
	return "the token endpoint refused the request"
}
