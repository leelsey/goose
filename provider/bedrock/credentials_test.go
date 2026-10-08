// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestTheCredentialSourcesAreUsableFromOutside(t *testing.T) {
	want := Credentials{AccessKeyID: "AKIA", SecretAccessKey: "secret"}
	got, err := StaticCredentials(want)(context.Background())
	if err != nil || got != want {
		t.Fatalf("static credentials = %+v, %v", got, err)
	}

	task := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "token" {
			t.Errorf("the container token did not travel")
		}
		json.NewEncoder(w).Encode(map[string]string{
			"AccessKeyId": "AKIATASK", "SecretAccessKey": "task-secret", "Token": "session",
			"Expiration": time.Now().Add(time.Hour).UTC().Format(time.RFC3339),
		})
	}))
	defer task.Close()

	source := FirstWorkingCredentials(
		func(context.Context) (Credentials, error) { return Credentials{}, context.Canceled },
		ContainerCredentials(task.Client(), "", task.URL, "token"),
	)
	creds, err := source(context.Background())
	if err != nil || creds.AccessKeyID != "AKIATASK" || creds.SessionToken != "session" {
		t.Fatalf("container credentials = %+v, %v", creds, err)
	}
}

func TestAPodIdentityTokenIsReadFromItsFile(t *testing.T) {
	file := filepath.Join(t.TempDir(), "eks-pod-identity-token")
	if err := os.WriteFile(file, []byte("projected"), 0o600); err != nil {
		t.Fatal(err)
	}
	agent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "projected" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		json.NewEncoder(w).Encode(map[string]string{
			"AccessKeyId": "AKIAPOD", "SecretAccessKey": "pod-secret", "Token": "session",
			"Expiration": time.Now().Add(time.Hour).UTC().Format(time.RFC3339),
		})
	}))
	defer agent.Close()

	creds, err := ContainerCredentialsFromFile(agent.Client(), "", agent.URL, file)(context.Background())
	if err != nil || creds.AccessKeyID != "AKIAPOD" {
		t.Fatalf("pod credentials = %+v, %v", creds, err)
	}
}

// Outside a container neither URI is set, and code that passes the AWS SDK's
// variables straight through must reach the next source rather than panic.
func TestAContainerSourceWithNoEndpointFailsAndIsPassed(t *testing.T) {
	file := filepath.Join(t.TempDir(), "eks-pod-identity-token")
	if err := os.WriteFile(file, []byte("projected"), 0o600); err != nil {
		t.Fatal(err)
	}
	fallback := StaticCredentials(Credentials{AccessKeyID: "AKIANEXT", SecretAccessKey: "next-secret"})
	for name, src := range map[string]CredentialSource{
		"ContainerCredentials":         ContainerCredentials(nil, "", "", ""),
		"ContainerCredentialsFromFile": ContainerCredentialsFromFile(nil, "", "", file),
	} {
		if creds, err := src(context.Background()); err == nil {
			t.Fatalf("%s = %+v: no endpoint is named, so there is nothing to answer", name, creds)
		}
		creds, err := FirstWorkingCredentials(src, fallback)(context.Background())
		if err != nil || creds.AccessKeyID != "AKIANEXT" {
			t.Fatalf("%s: credentials = %+v, %v: the next source answers", name, creds, err)
		}
	}
}

func TestAWebIdentityIsUsableFromOutside(t *testing.T) {
	file := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(file, []byte("projected"), 0o600); err != nil {
		t.Fatal(err)
	}
	sts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.PostFormValue("WebIdentityToken") != "projected" || r.PostFormValue("RoleSessionName") != "app" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = w.Write([]byte(`<AssumeRoleWithWebIdentityResponse><AssumeRoleWithWebIdentityResult><Credentials>` +
			`<AccessKeyId>ASIAPOD</AccessKeyId><SecretAccessKey>pod-secret</SecretAccessKey><SessionToken>session</SessionToken>` +
			`<Expiration>` + time.Now().Add(time.Hour).UTC().Format(time.RFC3339) + `</Expiration>` +
			`</Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>`))
	}))
	defer sts.Close()

	creds, err := WebIdentityCredentials(sts.Client(), sts.URL, "arn:aws:iam::111122223333:role/app", file, "app")(context.Background())
	if err != nil || creds.AccessKeyID != "ASIAPOD" || creds.SessionToken != "session" {
		t.Fatalf("web identity credentials = %+v, %v", creds, err)
	}
}

func TestNoUsableCredentialsIsAnError(t *testing.T) {
	if _, err := FirstWorkingCredentials()(context.Background()); err == nil {
		t.Fatal("an empty set of sources yielded credentials")
	}
	if _, err := FirstWorkingCredentials(nil)(context.Background()); err == nil {
		t.Fatal("a nil source yielded credentials")
	}
}
