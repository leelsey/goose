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
	profile := firstNonEmpty(p.Options["profile"], getenv("AWS_PROFILE"), "default")
	file := firstNonEmpty(p.Options["credentials_file"], getenv("AWS_SHARED_CREDENTIALS_FILE"), defaultAWSFile())

	relative := getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
	full, fullErr := containerURI(getenv("AWS_CONTAINER_CREDENTIALS_FULL_URI"))

	// Each source stands on its own. Tying the metadata switch to the absence of
	// a container URI left the instance role reachable as a fallback on exactly
	// the hosts where it is turned off to keep a task from borrowing the node's.
	var sources []bedrock.CredentialSource
	if relative != "" || full != "" {
		sources = append(sources, bedrock.ContainerCredentials(nil, relative, full,
			getenv("AWS_CONTAINER_AUTHORIZATION_TOKEN")))
	}
	if getenv("AWS_EC2_METADATA_DISABLED") != "true" {
		sources = append(sources, bedrock.InstanceCredentials(nil, getenv("AWS_EC2_METADATA_SERVICE_ENDPOINT")))
	}
	var runtime bedrock.CredentialSource
	if len(sources) > 0 {
		runtime = bedrock.FirstWorkingCredentials(sources...)
	}

	return func(ctx context.Context) (bedrock.Credentials, error) {
		if explicit.AccessKeyID != "" && explicit.SecretAccessKey != "" {
			return explicit, nil
		}
		env := bedrock.Credentials{
			AccessKeyID:     getenv("AWS_ACCESS_KEY_ID"),
			SecretAccessKey: getenv("AWS_SECRET_ACCESS_KEY"),
			SessionToken:    getenv("AWS_SESSION_TOKEN"),
		}
		if env.AccessKeyID != "" && env.SecretAccessKey != "" {
			return env, nil
		}
		var reasons []string
		if fullErr != nil {
			reasons = append(reasons, fullErr.Error())
		}
		if file != "" {
			creds, err := awsFileCredentials(file, profile)
			if err == nil {
				return creds, nil
			}
			reasons = append(reasons, fmt.Sprintf("%v (profile %q in %s)", err, profile, file))
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
	keyFile := firstNonEmpty(p.Options["credentials_file"], getenv("GOOGLE_APPLICATION_CREDENTIALS"))
	var sources []vertex.TokenSource
	project := ""
	if keyFile != "" {
		data, err := readFile(keyFile)
		if err != nil {
			return nil, "", fmt.Errorf("google credentials %s: %w", keyFile, err)
		}
		source, keyProject, err := vertex.ServiceAccountToken(data, nil)
		if err != nil {
			return nil, "", err
		}
		project = keyProject
		sources = append(sources, source)
	}
	sources = append(sources, vertex.MetadataToken(nil, getenv("GCE_METADATA_HOST")))
	return vertex.FirstWorkingToken(sources...), project, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
