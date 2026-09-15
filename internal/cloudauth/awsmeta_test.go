// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestContainerCredentialsAreReadAndCached(t *testing.T) {
	var reads int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "tok" {
			t.Errorf("the endpoint checks the token it was given: %q", r.Header.Get("Authorization"))
		}
		reads++
		_, _ = w.Write([]byte(`{"AccessKeyId":"AK","SecretAccessKey":"SK","Token":"ST",
			"Expiration":"` + time.Now().Add(time.Hour).Format(time.RFC3339) + `"}`))
	}))
	defer srv.Close()

	clock := time.Now()
	src := ContainerCredentials(srv.Client(), "", srv.URL, "tok", func() time.Time { return clock })
	for range 3 {
		creds, err := src(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if creds.AccessKeyID != "AK" || creds.SecretAccessKey != "SK" || creds.SessionToken != "ST" {
			t.Fatalf("creds = %+v", creds)
		}
	}
	if reads != 1 {
		t.Fatalf("reads = %d: a live set was fetched again", reads)
	}
}

func TestContainerCredentialsAreAbsentWithoutAnEndpoint(t *testing.T) {
	if ContainerCredentials(nil, "", "", "", nil) != nil {
		t.Fatal("a process that is not in a task offered container credentials")
	}
}

func TestInstanceCredentialsUseASessionToken(t *testing.T) {
	var sawToken bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPut && r.URL.Path == "/latest/api/token":
			if r.Header.Get("X-aws-ec2-metadata-token-ttl-seconds") == "" {
				t.Error("a session token is asked for with a life")
			}
			_, _ = w.Write([]byte("session"))
		case r.Header.Get("X-aws-ec2-metadata-token") != "session":
			w.WriteHeader(http.StatusUnauthorized)
		case r.URL.Path == "/latest/meta-data/iam/security-credentials/":
			sawToken = true
			_, _ = w.Write([]byte("role-one\n"))
		case strings.HasSuffix(r.URL.Path, "/role-one"):
			_, _ = w.Write([]byte(`{"AccessKeyId":"AK","SecretAccessKey":"SK","Token":"ST"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	src := InstanceCredentials(srv.Client(), srv.URL, nil)
	creds, err := src(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if !sawToken {
		t.Fatal("the role was read without the session token")
	}
	if creds.AccessKeyID != "AK" || creds.SessionToken != "ST" {
		t.Fatalf("creds = %+v", creds)
	}
}

func TestFirstWorkingCredentialsReportsEveryFailure(t *testing.T) {
	fail := func(msg string) CredentialSource {
		return func(context.Context) (Creds, error) { return Creds{}, errTest(msg) }
	}
	_, err := FirstWorkingCredentials(fail("one"), fail("two"))(context.Background())
	if err == nil || !strings.Contains(err.Error(), "one") || !strings.Contains(err.Error(), "two") {
		t.Fatalf("err = %v", err)
	}
	good := func(context.Context) (Creds, error) {
		return Creds{AccessKeyID: "AK", SecretAccessKey: "SK"}, nil
	}
	creds, err := FirstWorkingCredentials(fail("one"), good)(context.Background())
	if err != nil || creds.AccessKeyID != "AK" {
		t.Fatalf("creds = %+v err = %v", creds, err)
	}
}

type errTest string

func (e errTest) Error() string { return string(e) }

func TestARelativeURICannotMoveTheHostItIsJoinedTo(t *testing.T) {
	var asked string
	client := &http.Client{Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		asked = r.URL.Host
		return &http.Response{
			StatusCode: http.StatusOK,
			Body: io.NopCloser(strings.NewReader(
				`{"AccessKeyId":"A","SecretAccessKey":"S","Token":"T"}`)),
			Header: http.Header{},
		}, nil
	})}
	src := ContainerCredentials(client, "@elsewhere.example/creds", "", "token", time.Now)
	if _, err := src(context.Background()); err != nil {
		t.Fatal(err)
	}
	if asked != "169.254.170.2" {
		t.Fatalf("the credentials were asked of %q", asked)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
