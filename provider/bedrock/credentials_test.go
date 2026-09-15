// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

func TestNoUsableCredentialsIsAnError(t *testing.T) {
	if _, err := FirstWorkingCredentials()(context.Background()); err == nil {
		t.Fatal("an empty set of sources yielded credentials")
	}
	if _, err := FirstWorkingCredentials(nil)(context.Background()); err == nil {
		t.Fatal("a nil source yielded credentials")
	}
}
