// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net/netip"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/leelsey/goose/provider/bedrock"
	"github.com/leelsey/goose/provider/vertex"
)

func awsCredentials(p Provider, getenv func(string) string) bedrock.CredentialSource {
	explicit := bedrock.Credentials{
		AccessKeyID:     p.Options["access_key_id"],
		SecretAccessKey: p.Options["secret_access_key"],
		SessionToken:    p.Options["session_token"],
	}
	// A profile asked for by name is the identity the operator chose. The
	// runtime's role in its place signs as somebody else, so the AWS SDKs
	// refuse rather than fall through to it, and so does this. One the
	// configuration names is the program's choice, which they put ahead of
	// anything in the environment; AWS_PROFILE is the environment's own, and
	// yields to the keys or the web identity beside it.
	chosen := p.Options["profile"]
	named := getenv("AWS_PROFILE")
	file := firstNonEmpty(p.Options["credentials_file"], getenv("AWS_SHARED_CREDENTIALS_FILE"), defaultAWSFile())
	profile := func(name string) (bedrock.Credentials, error) {
		if file == "" {
			return bedrock.Credentials{}, fmt.Errorf("aws credentials: profile %q is named, but there is no shared credentials file to read it from", name)
		}
		creds, err := awsFileCredentials(file, name)
		if err != nil {
			return bedrock.Credentials{}, fmt.Errorf("aws credentials: the profile named cannot be used: %v (profile %q in %s)", err, name, file)
		}
		return creds, nil
	}

	relative := getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
	full, fullErr := containerURI(getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI"))

	// A pod given an EKS service account's role (IRSA) is handed a token file
	// and the role to assume with it. It is used or refused, as a container
	// endpoint is: the node's role behind it is no more the pod's.
	var webIdentity bedrock.CredentialSource
	if tokenFile := getenv("AWS_WEB_IDENTITY_TOKEN_FILE"); tokenFile != "" {
		region := firstNonEmpty(p.Options["region"], getenv("AWS_REGION"), getenv("AWS_DEFAULT_REGION"))
		if endpoint, err := stsEndpoint(getenv("AWS_ENDPOINT_URL_STS"), region); err != nil {
			err = fmt.Errorf("aws web identity credentials: %w", err)
			webIdentity = func(context.Context) (bedrock.Credentials, error) { return bedrock.Credentials{}, err }
		} else {
			webIdentity = bedrock.WebIdentityCredentials(nil, endpoint, getenv("AWS_ROLE_ARN"), tokenFile, getenv("AWS_ROLE_SESSION_NAME"))
		}
	}

	// A workload that names its own credentials endpoint is answered by that
	// endpoint or not at all, as the AWS SDKs do: when it refuses, the instance
	// role behind it is the node's, not the task's or the pod's, and signing with
	// it is the borrowing the endpoint exists to prevent.
	var runtime bedrock.CredentialSource
	switch {
	case relative != "" || full != "":
		if tokenFile := getenv("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE"); tokenFile != "" {
			runtime = bedrock.ContainerCredentialsFromFile(nil, relative, full, tokenFile)
		} else {
			runtime = bedrock.ContainerCredentials(nil, relative, full, getenv("AWS_CONTAINER_AUTHORIZATION_TOKEN"))
		}
	case fullErr == nil && !strings.EqualFold(getenv("AWS_EC2_METADATA_DISABLED"), "true"):
		runtime = bedrock.InstanceCredentials(nil, getenv("AWS_EC2_METADATA_SERVICE_ENDPOINT"))
	}

	return func(ctx context.Context) (bedrock.Credentials, error) {
		if explicit.AccessKeyID != "" && explicit.SecretAccessKey != "" {
			return explicit, nil
		}
		if chosen != "" {
			return profile(chosen)
		}
		env := bedrock.Credentials{
			AccessKeyID:     getenv("AWS_ACCESS_KEY_ID"),
			SecretAccessKey: getenv("AWS_SECRET_ACCESS_KEY"),
			SessionToken:    getenv("AWS_SESSION_TOKEN"),
		}
		if env.AccessKeyID != "" && env.SecretAccessKey != "" {
			return env, nil
		}
		if webIdentity != nil {
			return webIdentity(ctx)
		}
		if named != "" {
			return profile(named)
		}
		var reasons []string
		if fullErr != nil {
			reasons = append(reasons, fullErr.Error())
		}
		if file != "" {
			creds, err := awsFileCredentials(file, "default")
			if err == nil {
				return creds, nil
			}
			reasons = append(reasons, fmt.Sprintf("%v (profile %q in %s)", err, "default", file))
		}
		if runtime != nil {
			creds, err := runtime(ctx)
			if err == nil {
				return creds, nil
			}
			reasons = append(reasons, err.Error())
		}
		if len(reasons) == 0 {
			return bedrock.Credentials{}, errors.New("aws credentials: none are configured, in the file, the environment or the runtime")
		}
		return bedrock.Credentials{}, fmt.Errorf("aws credentials: nothing usable in the file, the environment or the runtime: %s",
			strings.Join(reasons, "; "))
	}
}

var containerAddrs = []netip.Addr{
	netip.MustParseAddr("169.254.170.2"),
	netip.MustParseAddr("169.254.170.23"),
	netip.MustParseAddr("fd00:ec2::23"),
}

func containerURI(full string) (string, error) {
	if full == "" {
		return "", nil
	}
	refused := fmt.Errorf("AWS_CONTAINER_CREDENTIALS_FULL_URI names %q, which is neither a container runtime's own address nor HTTPS, "+
		"so the token beside it was not sent there", full)
	u, err := url.Parse(full)
	if err != nil {
		return "", refused
	}
	if u.Scheme == "https" || (u.Scheme == "http" && runtimeAddress(u.Hostname())) {
		return full, nil
	}
	return "", refused
}

// The regional endpoint is the one the AWS SDKs now choose, and
// AWS_ENDPOINT_URL_STS replaces it as it does theirs. The web identity token is
// sent there, so only over HTTPS or to a local address, and a region only ever
// names the host rather than moving it.
func stsEndpoint(override, region string) (string, error) {
	if override != "" {
		u, err := url.Parse(override)
		if err != nil || !(u.Scheme == "https" || (u.Scheme == "http" && runtimeAddress(u.Hostname()))) {
			return "", fmt.Errorf("AWS_ENDPOINT_URL_STS names %q, which is neither a local address nor HTTPS, "+
				"so the web identity token was not sent there", override)
		}
		return override, nil
	}
	if region == "" {
		return "", nil
	}
	if !regionName(region) {
		return "", fmt.Errorf("region %q is not a region's name, so no STS endpoint was made from it", region)
	}
	return "https://sts." + region + ".amazonaws.com", nil
}

// regionName says s could name an AWS or Google Cloud region. A region is
// spliced into the host that a call and its credential are sent to, so a
// character any region's name lacks could move the call to another host.
func regionName(s string) bool {
	return s != "" && strings.Trim(s, "abcdefghijklmnopqrstuvwxyz0123456789-") == ""
}

func runtimeAddress(host string) bool {
	if host == "localhost" {
		return true
	}
	addr, err := netip.ParseAddr(host)
	if err != nil {
		return false
	}
	return addr.IsLoopback() || slices.Contains(containerAddrs, addr)
}

func defaultAWSFile() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".aws", "credentials")
}

func awsFileCredentials(path, profile string) (bedrock.Credentials, error) {
	f, err := os.Open(path)
	if err != nil {
		return bedrock.Credentials{}, err
	}
	defer f.Close()

	var creds bedrock.Credentials
	current := ""
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			current = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(line, "["), "]"))
			current = strings.TrimPrefix(current, "profile ")
			continue
		}
		if current != profile {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "aws_access_key_id":
			creds.AccessKeyID = strings.TrimSpace(value)
		case "aws_secret_access_key":
			creds.SecretAccessKey = strings.TrimSpace(value)
		case "aws_session_token":
			creds.SessionToken = strings.TrimSpace(value)
		}
	}
	if err := scanner.Err(); err != nil {
		return bedrock.Credentials{}, err
	}
	if creds.AccessKeyID == "" || creds.SecretAccessKey == "" {
		return bedrock.Credentials{}, fmt.Errorf("profile has no key pair")
	}
	return creds, nil
}

func googleToken(p Provider, getenv func(string) string, readFile func(string) ([]byte, error)) (vertex.TokenSource, string, error) {
	if tok := p.Options["access_token"]; tok != "" {
		return vertex.StaticToken(tok), "", nil
	}
	// A key named outright is the identity the operator chose, and Google's own
	// libraries never put the metadata server's account in its place: that one
	// would sign for the key's project as somebody else.
	keyFile := firstNonEmpty(p.Options["credentials_file"], getenv("GOOGLE_APPLICATION_CREDENTIALS"))
	if keyFile == "" {
		return vertex.MetadataToken(nil, getenv("GCE_METADATA_HOST")), "", nil
	}
	data, err := readFile(keyFile)
	if err != nil {
		return nil, "", fmt.Errorf("google credentials %s: %w", keyFile, err)
	}
	return vertex.ServiceAccountToken(data, nil)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
