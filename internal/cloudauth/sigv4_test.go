// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestSignatureMatchesTheReferenceVector(t *testing.T) {
	body := []byte("")
	req, err := http.NewRequest("GET", "https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08", nil)
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
	when := time.Date(2015, 8, 30, 12, 36, 0, 0, time.UTC)
	creds := Creds{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"}

	if err := SignV4(req, body, creds, "us-east-1", "iam", when); err != nil {
		t.Fatal(err)
	}
	auth := req.Header.Get("Authorization")
	if !strings.Contains(auth, "Credential=AKIDEXAMPLE/20150830/us-east-1/iam/aws4_request") {
		t.Fatalf("authorization = %q", auth)
	}
	const want = "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
	if !strings.Contains(auth, want) {
		t.Fatalf("authorization = %q\nwant it to contain %q", auth, want)
	}
	if req.Header.Get("X-Amz-Date") != "20150830T123600Z" {
		t.Fatalf("date = %q", req.Header.Get("X-Amz-Date"))
	}
}

func TestPathIsEncodedTwice(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v1%3A0/converse", nil)
	got := canonicalURI(req)
	if got != "/model/anthropic.claude-v1%253A0/converse" {
		t.Fatalf("canonical uri = %q", got)
	}
}

func TestSessionTokenIsSigned(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://example.amazonaws.com/x", nil)
	creds := Creds{AccessKeyID: "a", SecretAccessKey: "b", SessionToken: "token"}
	if err := SignV4(req, nil, creds, "eu-west-2", "bedrock", time.Unix(0, 0)); err != nil {
		t.Fatal(err)
	}
	if req.Header.Get("X-Amz-Security-Token") != "token" {
		t.Fatal("the session token must travel")
	}
	if !strings.Contains(req.Header.Get("Authorization"), "x-amz-security-token") {
		t.Fatal("a temporary credential that is not signed with its token is rejected")
	}
}

func TestEmptyCredentialsFailClosed(t *testing.T) {
	req, _ := http.NewRequest("POST", "https://example.amazonaws.com/x", nil)
	if err := SignV4(req, nil, Creds{}, "eu-west-2", "bedrock", time.Unix(0, 0)); err == nil {
		t.Fatal("an unauthenticated request must not leave the process")
	}
	if req.Header.Get("Authorization") != "" {
		t.Fatal("a failed signing must not leave a header behind")
	}
}
