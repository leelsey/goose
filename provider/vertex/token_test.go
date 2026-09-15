// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package vertex

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestTheTokenSourcesAreUsableFromOutside(t *testing.T) {
	if tok, err := StaticToken("abc").Token(context.Background()); err != nil || tok != "abc" {
		t.Fatalf("static token = %q, %v", tok, err)
	}

	meta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			t.Errorf("metadata read without the flavour header")
		}
		json.NewEncoder(w).Encode(map[string]any{"access_token": "from-metadata", "expires_in": 3600})
	}))
	defer meta.Close()

	source := FirstWorkingToken(
		TokenFunc(func(context.Context) (string, error) { return "", context.Canceled }),
		MetadataToken(meta.Client(), meta.URL),
	)
	tok, err := source.Token(context.Background())
	if err != nil || tok != "from-metadata" {
		t.Fatalf("first working token = %q, %v", tok, err)
	}
}

func TestAServiceAccountKeyNamesItsProject(t *testing.T) {
	key := `{"type":"service_account","project_id":"the-project","client_email":"a@b.c",
	         "private_key":"-----BEGIN PRIVATE KEY-----\nnot a key\n-----END PRIVATE KEY-----\n"}`
	if _, _, err := ServiceAccountToken([]byte(key), nil); err == nil {
		t.Fatal("a malformed private key was accepted")
	}
	if _, _, err := ServiceAccountToken([]byte(`{`), nil); err == nil {
		t.Fatal("malformed JSON was accepted")
	}
	if _, _, err := ServiceAccountToken([]byte(`{"project_id":"p"}`), nil); err == nil ||
		!strings.Contains(err.Error(), "client_email") {
		t.Fatalf("a key with no identity was accepted: %v", err)
	}
}
