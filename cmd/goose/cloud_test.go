// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/BurntSushi/toml"

	"github.com/leelsey/goose/provider/bedrock"
)

// nodeRole stands in for EC2 instance metadata handing out the node's own role,
// and counts every time anything asks it.
func nodeRole(t *testing.T) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked.Add(1)
		switch {
		case r.Method == http.MethodPut:
			_, _ = w.Write([]byte("session"))
		case strings.HasSuffix(r.URL.Path, "/security-credentials/"):
			_, _ = w.Write([]byte("node-role"))
		default:
			_, _ = w.Write([]byte(`{"AccessKeyId":"NODE","SecretAccessKey":"node-secret","Token":"ST"}`))
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &asked
}

func TestAPodIdentitySendsItsTokenFileAndNeverFallsBackToTheNode(t *testing.T) {
	imds, asked := nodeRole(t)
	file := filepath.Join(t.TempDir(), "eks-pod-identity-token")
	if err := os.WriteFile(file, []byte("projected"), 0o600); err != nil {
		t.Fatal(err)
	}
	var refuse atomic.Bool
	agent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if refuse.Load() || r.Header.Get("Authorization") != "projected" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = w.Write([]byte(`{"AccessKeyId":"POD","SecretAccessKey":"pod-secret","Token":"ST"}`))
	}))
	defer agent.Close()

	env := map[string]string{
		"AWS_CONTAINER_CREDENTIALS_FULL_URI":     agent.URL + "/v1/credentials",
		"AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE": file,
		"AWS_EC2_METADATA_SERVICE_ENDPOINT":      imds.URL,
		"AWS_SHARED_CREDENTIALS_FILE":            filepath.Join(t.TempDir(), "none"),
	}
	src := awsCredentials(Provider{}, func(k string) string { return env[k] })
	creds, err := src(context.Background())
	if err != nil || creds.AccessKeyID != "POD" {
		t.Fatalf("credentials = %+v, %v: the pod's agent is answered with the token in its file", creds, err)
	}

	refuse.Store(true)
	if creds, err := src(context.Background()); err == nil {
		t.Fatalf("credentials = %+v: the pod's agent refused, so resolving credentials must fail", creds)
	}
	if asked.Load() != 0 {
		t.Fatal("the node's own role was asked for when the pod's refused")
	}
}

// stsRole stands in for STS answering AssumeRoleWithWebIdentity for a pod whose
// token is "projected", and counts every time anything asks it.
func stsRole(t *testing.T, refuse *atomic.Bool) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var asked atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		asked.Add(1)
		if refuse.Load() || r.PostFormValue("WebIdentityToken") != "projected" || r.PostFormValue("RoleArn") != "arn:aws:iam::111122223333:role/app" {
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`<ErrorResponse><Error><Code>AccessDenied</Code></Error></ErrorResponse>`))
			return
		}
		fmt.Fprintf(w, `<AssumeRoleWithWebIdentityResponse><AssumeRoleWithWebIdentityResult><Credentials>`+
			`<AccessKeyId>POD</AccessKeyId><SecretAccessKey>pod-secret</SecretAccessKey><SessionToken>ST</SessionToken>`+
			`<Expiration>%s</Expiration></Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>`,
			time.Now().Add(time.Hour).UTC().Format(time.RFC3339))
	}))
	t.Cleanup(srv.Close)
	return srv, &asked
}

func TestAnIRSAPodAssumesItsRoleAndNeverFallsBackToTheNode(t *testing.T) {
	imds, nodeAsked := nodeRole(t)
	var refuse atomic.Bool
	sts, stsAsked := stsRole(t, &refuse)
	token := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(token, []byte("projected"), 0o600); err != nil {
		t.Fatal(err)
	}
	env := map[string]string{
		"AWS_ROLE_ARN":                      "arn:aws:iam::111122223333:role/app",
		"AWS_WEB_IDENTITY_TOKEN_FILE":       token,
		"AWS_ENDPOINT_URL_STS":              sts.URL,
		"AWS_EC2_METADATA_SERVICE_ENDPOINT": imds.URL,
		"AWS_SHARED_CREDENTIALS_FILE":       filepath.Join(t.TempDir(), "none"),
	}
	resolve := func() (bedrock.Credentials, error) {
		return awsCredentials(Provider{}, func(k string) string { return env[k] })(context.Background())
	}

	if creds, err := resolve(); err != nil || creds.AccessKeyID != "POD" {
		t.Fatalf("credentials = %+v, %v: the pod's own role is assumed with its token", creds, err)
	}

	refuse.Store(true)
	if creds, err := resolve(); err == nil || !strings.Contains(err.Error(), "AccessDenied") {
		t.Fatalf("credentials = %+v, %v: STS refused the pod's role, so resolving credentials must fail and say why", creds, err)
	}
	refuse.Store(false)

	env["AWS_WEB_IDENTITY_TOKEN_FILE"] = filepath.Join(t.TempDir(), "gone")
	if creds, err := resolve(); err == nil {
		t.Fatalf("credentials = %+v: the pod's token cannot be read, so resolving credentials must fail", creds)
	}
	env["AWS_WEB_IDENTITY_TOKEN_FILE"] = token

	delete(env, "AWS_ROLE_ARN")
	if creds, err := resolve(); err == nil {
		t.Fatalf("credentials = %+v: a token with no role to assume is not a reason to use another", creds)
	}
	env["AWS_ROLE_ARN"] = "arn:aws:iam::111122223333:role/app"

	env["AWS_ENDPOINT_URL_STS"] = "http://sts.example.com"
	if creds, err := resolve(); err == nil {
		t.Fatalf("credentials = %+v: the token would have crossed the network in the clear", creds)
	}

	if nodeAsked.Load() != 0 {
		t.Fatal("the node's own role was asked for in place of the pod's")
	}
	if stsAsked.Load() != 2 {
		t.Fatalf("STS was asked %d times: only with a token, a role and an endpoint it could be trusted with", stsAsked.Load())
	}
}

func TestTheSTSEndpointIsTheRegionsUnlessNamed(t *testing.T) {
	for _, c := range []struct{ override, region, want string }{
		{"", "eu-west-2", "https://sts.eu-west-2.amazonaws.com"},
		{"", "", ""},
		{"https://sts-fips.us-east-1.amazonaws.com", "us-east-1", "https://sts-fips.us-east-1.amazonaws.com"},
		{"http://127.0.0.1:4566", "us-east-1", "http://127.0.0.1:4566"},
	} {
		if got, err := stsEndpoint(c.override, c.region); err != nil || got != c.want {
			t.Errorf("stsEndpoint(%q, %q) = %q, %v; want %q", c.override, c.region, got, err, c.want)
		}
	}
	for _, c := range []struct{ override, region string }{
		{"http://sts.example.com", "us-east-1"},
		{"ftp://127.0.0.1", "us-east-1"},
		{"", "evil.example/"},
		{"", "us-east-1.evil.example#"},
	} {
		if got, err := stsEndpoint(c.override, c.region); err == nil {
			t.Errorf("stsEndpoint(%q, %q) = %q: the web identity token would go somewhere it should not", c.override, c.region, got)
		}
	}
}

// The AWS SDKs take a profile the program names over keys in the environment,
// and AWS_PROFILE, which is the environment's own, under them.
func TestAConfiguredProfileIsPreferredToKeysInTheEnvironment(t *testing.T) {
	file := filepath.Join(t.TempDir(), "credentials")
	if err := os.WriteFile(file, []byte("[work]\naws_access_key_id = WORK\naws_secret_access_key = S\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	env := map[string]string{
		"AWS_ACCESS_KEY_ID":           "PERSONAL",
		"AWS_SECRET_ACCESS_KEY":       "P",
		"AWS_SHARED_CREDENTIALS_FILE": file,
		"AWS_EC2_METADATA_DISABLED":   "true",
	}
	getenv := func(k string) string { return env[k] }

	creds, err := awsCredentials(Provider{Options: map[string]string{"profile": "work"}}, getenv)(context.Background())
	if err != nil || creds.AccessKeyID != "WORK" {
		t.Fatalf("credentials = %+v, %v: the profile the configuration names signs, not the keys beside it", creds, err)
	}
	if creds, err := awsCredentials(Provider{Options: map[string]string{"profile": "missing"}}, getenv)(context.Background()); err == nil {
		t.Fatalf("credentials = %+v: a configured profile that cannot be used was passed over for the environment's keys", creds)
	}

	env["AWS_PROFILE"] = "work"
	creds, err = awsCredentials(Provider{}, getenv)(context.Background())
	if err != nil || creds.AccessKeyID != "PERSONAL" {
		t.Fatalf("credentials = %+v, %v: AWS_PROFILE yields to keys in the environment, as it does in the AWS SDKs", creds, err)
	}
}

func TestANamedVertexKeyIsUsedOrRefused(t *testing.T) {
	oauth := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"invalid_grant"}`))
	}))
	defer oauth.Close()
	var metaAsked atomic.Int64
	meta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		metaAsked.Add(1)
		_, _ = w.Write([]byte(`{"access_token":"VM-DEFAULT-SA","expires_in":3600}`))
	}))
	defer meta.Close()

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]string{
		"type": "service_account", "project_id": "keyproj",
		"client_email": "app@keyproj.iam.gserviceaccount.com",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"token_uri":    oauth.URL,
	})
	if err != nil {
		t.Fatal(err)
	}
	key := filepath.Join(t.TempDir(), "key.json")
	if err := os.WriteFile(key, body, 0o600); err != nil {
		t.Fatal(err)
	}

	host := strings.TrimPrefix(meta.URL, "http://")
	for _, c := range []struct {
		name string
		p    Provider
		env  map[string]string
	}{
		{"options", Provider{Options: map[string]string{"credentials_file": key}}, map[string]string{"GCE_METADATA_HOST": host}},
		{"GOOGLE_APPLICATION_CREDENTIALS", Provider{}, map[string]string{"GCE_METADATA_HOST": host, "GOOGLE_APPLICATION_CREDENTIALS": key}},
	} {
		src, project, err := googleToken(c.p, func(k string) string { return c.env[k] }, os.ReadFile)
		if err != nil {
			t.Fatal(err)
		}
		tok, err := src.Token(context.Background())
		if err == nil || !strings.Contains(err.Error(), "invalid_grant") {
			t.Fatalf("%s: token = %q, %v: the key's refusal is the answer for %s", c.name, tok, err, project)
		}
	}
	if metaAsked.Load() != 0 {
		t.Fatal("the metadata server's account was asked for in place of the key named")
	}
}

func TestInstanceMetadataIsOffHoweverTrueIsSpelt(t *testing.T) {
	for _, value := range []string{"true", "True", "TRUE"} {
		imds, asked := nodeRole(t)
		env := map[string]string{
			"AWS_EC2_METADATA_DISABLED":         value,
			"AWS_EC2_METADATA_SERVICE_ENDPOINT": imds.URL,
			"AWS_SHARED_CREDENTIALS_FILE":       filepath.Join(t.TempDir(), "none"),
		}
		creds, err := awsCredentials(Provider{}, func(k string) string { return env[k] })(context.Background())
		if err == nil || asked.Load() != 0 {
			t.Fatalf("%s: credentials = %+v, %v: the node's role was used where the AWS SDKs would not", value, creds, err)
		}
	}

	imds, _ := nodeRole(t)
	env := map[string]string{
		"AWS_EC2_METADATA_DISABLED":         "false",
		"AWS_EC2_METADATA_SERVICE_ENDPOINT": imds.URL,
		"AWS_SHARED_CREDENTIALS_FILE":       filepath.Join(t.TempDir(), "none"),
	}
	if creds, err := awsCredentials(Provider{}, func(k string) string { return env[k] })(context.Background()); err != nil || creds.AccessKeyID != "NODE" {
		t.Fatalf("credentials = %+v, %v: metadata left on is read", creds, err)
	}
}

func TestANamedProfileIsUsedOrRefusedButNeverPassedOver(t *testing.T) {
	file := filepath.Join(t.TempDir(), "credentials")
	if err := os.WriteFile(file, []byte("[default]\naws_access_key_id = DEFAULT\naws_secret_access_key = S\n"+
		"[work]\nsso_session = corp\nsso_account_id = 111122223333\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	imds, asked := nodeRole(t)
	base := map[string]string{
		"AWS_SHARED_CREDENTIALS_FILE":       file,
		"AWS_EC2_METADATA_SERVICE_ENDPOINT": imds.URL,
	}
	getenv := func(extra map[string]string) func(string) string {
		return func(k string) string {
			if v, ok := extra[k]; ok {
				return v
			}
			return base[k]
		}
	}

	for name, src := range map[string]bedrock.CredentialSource{
		"options":     awsCredentials(Provider{Options: map[string]string{"profile": "work"}}, getenv(nil)),
		"AWS_PROFILE": awsCredentials(Provider{}, getenv(map[string]string{"AWS_PROFILE": "work"})),
		"absent":      awsCredentials(Provider{Options: map[string]string{"profile": "missing"}}, getenv(nil)),
	} {
		creds, err := src(context.Background())
		if err == nil {
			t.Fatalf("%s: credentials = %+v: a profile named outright was passed over for another identity", name, creds)
		}
		if !strings.Contains(err.Error(), "profile") {
			t.Fatalf("%s: err = %v: the error must say the named profile is what failed", name, err)
		}
	}
	if asked.Load() != 0 {
		t.Fatal("the node's role was asked for in place of a profile named outright")
	}

	creds, err := awsCredentials(Provider{Options: map[string]string{"profile": "default"}}, getenv(nil))(context.Background())
	if err != nil || creds.AccessKeyID != "DEFAULT" {
		t.Fatalf("credentials = %+v, %v", creds, err)
	}
}

// The example is what an operator copies onto an AWS host, and it promises the
// runtime role last. A profile named in it would be used or refused instead.
func TestTheExampleBedrockReachesTheRuntimeRole(t *testing.T) {
	raw, err := os.ReadFile(examplePath)
	if err != nil {
		t.Fatal(err)
	}
	entry := regexp.MustCompile(`(?m)^# \[\[provider\]\]\n# id = "bedrock"\n(?:# .*\n)+`).Find(raw)
	if entry == nil {
		t.Fatal("the example has no Bedrock entry")
	}
	var cfg Config
	if _, err := toml.Decode(string(regexp.MustCompile(`(?m)^# `).ReplaceAll(entry, nil)), &cfg); err != nil || len(cfg.Providers) != 1 {
		t.Fatalf("the example's Bedrock entry does not decode: %v", err)
	}

	imds, _ := nodeRole(t)
	env := map[string]string{
		"AWS_EC2_METADATA_SERVICE_ENDPOINT": imds.URL,
		"AWS_SHARED_CREDENTIALS_FILE":       filepath.Join(t.TempDir(), "none"),
	}
	creds, err := awsCredentials(cfg.Providers[0], func(k string) string { return env[k] })(context.Background())
	if err != nil || creds.AccessKeyID != "NODE" {
		t.Fatalf("credentials = %+v, %v: the example's Bedrock entry never reaches the runtime role it documents", creds, err)
	}
}

func TestGCEMetadataHostIsAHostAndAPort(t *testing.T) {
	meta := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		_, _ = w.Write([]byte(`{"access_token":"from-metadata","expires_in":3600}`))
	}))
	defer meta.Close()

	host := strings.TrimPrefix(meta.URL, "http://")
	src, _, err := googleToken(Provider{}, func(k string) string {
		if k == "GCE_METADATA_HOST" {
			return host
		}
		return ""
	}, os.ReadFile)
	if err != nil {
		t.Fatal(err)
	}
	if tok, err := src.Token(context.Background()); err != nil || tok != "from-metadata" {
		t.Fatalf("token = %q, %v: GCE_METADATA_HOST=%s names a host and a port, as Google's libraries read it", tok, err, host)
	}
}
