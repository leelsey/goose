// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const stsAnswer = `<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
  <AssumeRoleWithWebIdentityResult>
    <SubjectFromWebIdentityToken>system:serviceaccount:default:app</SubjectFromWebIdentityToken>
    <AssumedRoleUser>
      <Arn>arn:aws:sts::111122223333:assumed-role/app/goose</Arn>
      <AssumedRoleId>AROAEXAMPLE:goose</AssumedRoleId>
    </AssumedRoleUser>
    <Credentials>
      <SessionToken>ST</SessionToken>
      <SecretAccessKey>SK</SecretAccessKey>
      <Expiration>%s</Expiration>
      <AccessKeyId>ASIAPOD</AccessKeyId>
    </Credentials>
  </AssumeRoleWithWebIdentityResult>
  <ResponseMetadata><RequestId>r</RequestId></ResponseMetadata>
</AssumeRoleWithWebIdentityResponse>`

func TestAWebIdentityIsExchangedForItsRolesKeys(t *testing.T) {
	file := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(file, []byte("first\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		switch {
		case r.Method != http.MethodPost || r.URL.Path != "/":
			t.Errorf("%s %s: STS is asked by a POST to its root", r.Method, r.URL.Path)
		case r.PostForm.Get("Action") != "AssumeRoleWithWebIdentity" || r.PostForm.Get("Version") != "2011-06-15":
			t.Errorf("form = %v", r.PostForm)
		case r.PostForm.Get("RoleArn") != "arn:aws:iam::111122223333:role/app" || r.PostForm.Get("RoleSessionName") == "":
			t.Errorf("form = %v: the role and a session name travel", r.PostForm)
		case r.Header.Get("Authorization") != "":
			t.Error("the token is the credential; nothing else is sent")
		}
		seen = append(seen, r.PostForm.Get("WebIdentityToken"))
		fmt.Fprintf(w, stsAnswer, time.Now().Add(time.Hour).UTC().Format(time.RFC3339))
	}))
	defer srv.Close()

	clock := time.Now()
	src := WebIdentityCredentials(srv.Client(), srv.URL, "arn:aws:iam::111122223333:role/app", file, "", func() time.Time { return clock })
	for range 2 {
		creds, err := src(context.Background())
		if err != nil || creds.AccessKeyID != "ASIAPOD" || creds.SecretAccessKey != "SK" || creds.SessionToken != "ST" {
			t.Fatalf("credentials = %+v, %v", creds, err)
		}
	}
	if err := os.WriteFile(file, []byte("second"), 0o600); err != nil {
		t.Fatal(err)
	}
	clock = clock.Add(2 * time.Hour)
	if _, err := src(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || seen[0] != "first" || seen[1] != "second" {
		t.Fatalf("tokens sent = %q: a live set is kept, and a rotated token is sent at the next refresh", seen)
	}
}

func TestARefusalFromSTSIsToldByItsCode(t *testing.T) {
	file := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(file, []byte("expired"), 0o600); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`<ErrorResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/"><Error><Type>Sender</Type>` +
			`<Code>InvalidIdentityToken</Code><Message>No OpenIDConnect provider found in your account for https://oidc.example</Message>` +
			`</Error><RequestId>r</RequestId></ErrorResponse>`))
	}))
	defer srv.Close()

	_, err := WebIdentityCredentials(srv.Client(), srv.URL, "arn:aws:iam::111122223333:role/app", file, "s", nil)(context.Background())
	if err == nil || !strings.Contains(err.Error(), "InvalidIdentityToken") || strings.Contains(err.Error(), "oidc.example") {
		t.Fatalf("err = %v: STS's code is told, and the message beside it is not", err)
	}
}

func TestAWebIdentityWithoutItsRoleOrItsTokenAsksNothing(t *testing.T) {
	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { asked.Add(1) }))
	defer srv.Close()
	file := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(file, []byte("tok"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := WebIdentityCredentials(srv.Client(), srv.URL, "", file, "", nil)(context.Background()); err == nil {
		t.Fatal("a web identity with no role to assume yielded credentials")
	}
	missing := filepath.Join(t.TempDir(), "none")
	if _, err := WebIdentityCredentials(srv.Client(), srv.URL, "arn:aws:iam::111122223333:role/app", missing, "", nil)(context.Background()); err == nil {
		t.Fatal("a token file that cannot be read yielded credentials")
	}
	if asked.Load() != 0 {
		t.Fatal("STS was asked without a role or a token")
	}
}
