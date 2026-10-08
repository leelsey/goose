// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package cloudauth

import (
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const stsHost = "https://sts.amazonaws.com"

// WebIdentityCredentials assumes roleARN through STS's AssumeRoleWithWebIdentity
// with the token in tokenFile, which is how an EKS service account's role
// (IRSA) reaches a pod. The token is the credential, so the call is not signed,
// and the file is read at every refresh because the token in it is rotated
// under the running process.
func WebIdentityCredentials(client *http.Client, endpoint, roleARN, tokenFile, sessionName string, now func() time.Time) CredentialSource {
	if endpoint == "" {
		endpoint = stsHost
	}
	endpoint = strings.TrimRight(endpoint, "/") + "/"
	if now == nil {
		now = time.Now
	}
	e := &expiring{now: now, timeout: exchangeTimeout, fetch: func(ctx context.Context) (Creds, time.Time, error) {
		if roleARN == "" {
			return Creds{}, time.Time{}, errors.New("aws web identity credentials: no role is named to assume")
		}
		token, err := os.ReadFile(tokenFile)
		if err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws web identity credentials: %w", err)
		}
		session := sessionName
		if session == "" {
			session = "goose-" + strconv.FormatInt(now().UnixNano(), 10)
		}
		form := url.Values{
			"Action":           {"AssumeRoleWithWebIdentity"},
			"Version":          {"2011-06-15"},
			"RoleArn":          {roleARN},
			"RoleSessionName":  {session},
			"WebIdentityToken": {strings.TrimSpace(string(token))},
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
		if err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws web identity credentials: %w", err)
		}
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		c := client
		if c == nil {
			c = http.DefaultClient
		}
		resp, err := c.Do(req)
		if err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws web identity credentials: %w", err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<16))
		if resp.StatusCode != http.StatusOK {
			return Creds{}, time.Time{}, fmt.Errorf("aws web identity credentials: %d %s", resp.StatusCode, stsError(body))
		}
		var out struct {
			Credentials struct {
				AccessKeyID     string `xml:"AccessKeyId"`
				SecretAccessKey string `xml:"SecretAccessKey"`
				SessionToken    string `xml:"SessionToken"`
				Expiration      string `xml:"Expiration"`
			} `xml:"AssumeRoleWithWebIdentityResult>Credentials"`
		}
		if err := xml.Unmarshal(body, &out); err != nil {
			return Creds{}, time.Time{}, fmt.Errorf("aws web identity credentials: %w", err)
		}
		creds, expires := containerCreds{
			AccessKeyID:     out.Credentials.AccessKeyID,
			SecretAccessKey: out.Credentials.SecretAccessKey,
			Token:           out.Credentials.SessionToken,
			Expiration:      out.Credentials.Expiration,
		}.creds()
		if !creds.valid() {
			return Creds{}, time.Time{}, errors.New("aws web identity credentials: STS returned no keys")
		}
		return creds, expires, nil
	}}
	return e.Get
}

// stsError is STS's code for a refusal, as oauthError is Google's, leaving out
// the message beside it, which can name the account's identity provider.
func stsError(body []byte) string {
	var out struct {
		Code string `xml:"Error>Code"`
	}
	if xml.Unmarshal(body, &out) == nil && out.Code != "" {
		return out.Code
	}
	return "STS refused the request"
}
