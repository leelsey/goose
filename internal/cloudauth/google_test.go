// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func keyJSON(t *testing.T, tokenURI string) []byte {
	t.Helper()
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	pemKey := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})
	body, err := json.Marshal(map[string]string{
		"type": "service_account", "project_id": "p",
		"client_email": "robot@example.iam.gserviceaccount.com",
		"private_key":  string(pemKey), "token_uri": tokenURI,
	})
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func TestServiceAccountExchangeSendsASignedAssertion(t *testing.T) {
	var calls atomic.Int64
	var assertion string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		if got := r.Form.Get("grant_type"); got != "urn:ietf:params:oauth:grant-type:jwt-bearer" {
			t.Errorf("grant_type = %q", got)
		}
		assertion = r.Form.Get("assertion")
		fmt.Fprint(w, `{"access_token":"tok-1","expires_in":3600}`)
	}))
	defer srv.Close()

	key, err := ParseServiceAccountKey(keyJSON(t, srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	src, err := ServiceAccountSource(key, srv.Client(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := src.Token(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if tok != "tok-1" {
		t.Fatalf("token = %q", tok)
	}

	parts := strings.Split(assertion, ".")
	if len(parts) != 3 {
		t.Fatalf("assertion is not a JWT: %q", assertion)
	}
	claims, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		ISS   string `json:"iss"`
		Scope string `json:"scope"`
		AUD   string `json:"aud"`
		EXP   int64  `json:"exp"`
	}
	if err := json.Unmarshal(claims, &body); err != nil {
		t.Fatal(err)
	}
	if body.ISS != "robot@example.iam.gserviceaccount.com" || body.AUD != srv.URL {
		t.Fatalf("claims = %+v", body)
	}
	if body.Scope != cloudPlatformScope {
		t.Fatalf("scope = %q", body.Scope)
	}
	if body.EXP == 0 {
		t.Fatal("the assertion never expires")
	}

	if _, err := src.Token(context.Background()); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("exchanged %d times: a cached token is not re-bought", calls.Load())
	}
}

func TestATokenIsReplacedBeforeItExpires(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		fmt.Fprintf(w, `{"access_token":"tok-%d","expires_in":120}`, n)
	}))
	defer srv.Close()

	key, err := ParseServiceAccountKey(keyJSON(t, srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	clock := time.Now()
	src, err := ServiceAccountSource(key, srv.Client(), nil, func() time.Time { return clock })
	if err != nil {
		t.Fatal(err)
	}
	if _, err := src.Token(context.Background()); err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(30 * time.Second)
	if tok, _ := src.Token(context.Background()); tok != "tok-1" {
		t.Fatalf("token = %q: a token with life left is not re-bought", tok)
	}
	clock = clock.Add(80 * time.Second)
	if tok, _ := src.Token(context.Background()); tok != "tok-2" {
		t.Fatalf("token = %q: a token about to expire was reused", tok)
	}
}

func TestAnExchangeFailureCarriesOnlyItsCode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":"invalid_grant","error_description":"Invalid JWT: robot@example.iam.gserviceaccount.com is not authorised for project secret-project-42"}`)
	}))
	defer srv.Close()

	key, err := ParseServiceAccountKey(keyJSON(t, srv.URL))
	if err != nil {
		t.Fatal(err)
	}
	src, err := ServiceAccountSource(key, srv.Client(), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = src.Token(context.Background())
	if err == nil {
		t.Fatal("a refused exchange is not a token")
	}
	if !strings.Contains(err.Error(), "invalid_grant") {
		t.Fatalf("error = %v: the code is what the operator needs", err)
	}
	if strings.Contains(err.Error(), "secret-project-42") || strings.Contains(err.Error(), "not authorised") {
		t.Fatalf("error = %v: the description names the account and the project", err)
	}
}

func TestAKeyWithoutAnEmailOrAPrivateKeyIsRefused(t *testing.T) {
	for _, body := range []string{`{}`, `{"client_email":"a@b"}`, `{"private_key":"x"}`, `not json`} {
		if _, err := ParseServiceAccountKey([]byte(body)); err == nil {
			t.Fatalf("accepted %q", body)
		}
	}
}

func TestAStaticTokenRefusesToBeEmpty(t *testing.T) {
	if _, err := StaticToken("").Token(context.Background()); err == nil {
		t.Fatal("an empty token is not a credential")
	}
	tok, err := StaticToken("t").Token(context.Background())
	if err != nil || tok != "t" {
		t.Fatalf("token = %q, err = %v", tok, err)
	}
}

func TestAMetadataTokenIsCachedAcrossCallers(t *testing.T) {
	var reads int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			t.Errorf("the metadata service refuses a request without its header")
		}
		reads++
		_, _ = w.Write([]byte(`{"access_token":"tok","expires_in":3600}`))
	}))
	defer srv.Close()

	clock := time.Now()
	src := MetadataSource(srv.Client(), srv.URL, func() time.Time { return clock })
	for range 3 {
		token, err := src.Token(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if token != "tok" {
			t.Fatalf("token = %q", token)
		}
	}
	if reads != 1 {
		t.Fatalf("reads = %d: a live token was fetched again", reads)
	}
	clock = clock.Add(2 * time.Hour)
	if _, err := src.Token(context.Background()); err != nil {
		t.Fatal(err)
	}
	if reads != 2 {
		t.Fatalf("reads = %d: an expired token was served", reads)
	}
}
