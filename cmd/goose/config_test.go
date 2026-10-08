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
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/provider/openai"
	"github.com/leelsey/goose/route"
)

func write(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "goose.toml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestUnknownKeyIsRefused(t *testing.T) {
	path := write(t, `
[server]
port = 11234
formats = ["openai"]
typo_here = true
`)
	if _, _, err := load(path); err == nil || !strings.Contains(err.Error(), "unknown key") {
		t.Fatalf("err = %v", err)
	}
}

// A file a config names relatively — a TLS certificate or key, a provider's
// credentials_file — is read wherever goose was started, not wherever the
// config itself sits; goose mcp in particular starts wherever its client has
// a project open, so it must resolve beside the file that named it, the same
// way usage.log already does.
func TestRelativeFilePathsAreBesideTheConfigThatNamesThem(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "goose.toml")
	body := `
[server]
tls_cert = "cert.pem"
tls_key  = "key.pem"

[[provider]]
id = "v"
kind = "vertex"
model = "gemini-2.0-flash"
options = { credentials_file = "sa.json" }
`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	elsewhere := t.TempDir()
	t.Chdir(elsewhere)

	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(dir, "cert.pem"); cfg.Server.TLSCert != want {
		t.Errorf("tls_cert = %q, want %q", cfg.Server.TLSCert, want)
	}
	if want := filepath.Join(dir, "key.pem"); cfg.Server.TLSKey != want {
		t.Errorf("tls_key = %q, want %q", cfg.Server.TLSKey, want)
	}
	if want := filepath.Join(dir, "sa.json"); cfg.Providers[0].Options["credentials_file"] != want {
		t.Errorf("credentials_file = %q, want %q", cfg.Providers[0].Options["credentials_file"], want)
	}

	abs := filepath.Join(t.TempDir(), "abs.pem")
	if err := os.WriteFile(abs, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	absPath := filepath.Join(dir, "abs.toml")
	absBody := "[server]\ntls_cert = \"" + filepath.ToSlash(abs) + "\"\ntls_key = \"" + filepath.ToSlash(abs) + "\"\n"
	if err := os.WriteFile(absPath, []byte(absBody), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, _, err = load(absPath)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.TLSCert != abs {
		t.Errorf("an absolute tls_cert was rewritten: %q, want %q", cfg.Server.TLSCert, abs)
	}
}

func TestFormatsAcceptedAtRootAndUnderServer(t *testing.T) {
	root := write(t, `
formats = ["anthropic"]

[server]
port = 11234
`)
	cfg, _, err := load(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Formats) != 1 || cfg.Formats[0] != "anthropic" {
		t.Fatalf("formats = %v", cfg.Formats)
	}

	nested := write(t, `
[server]
port = 11234
formats = ["ollama"]
`)
	cfg, _, err = load(nested)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Formats) != 1 || cfg.Formats[0] != "ollama" {
		t.Fatalf("formats = %v", cfg.Formats)
	}
}

func TestSeveralProvidersNeedRoutes(t *testing.T) {
	path := write(t, `
[[provider]]
id = "a"
kind = "ollama"

[[provider]]
id = "b"
kind = "ollama"
`)
	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err == nil {
		t.Fatal("routing by accident is how a request reaches a backend nobody chose")
	}
}

func TestRouteNamingAnUnknownProviderIsRefused(t *testing.T) {
	path := write(t, `
[[provider]]
id = "a"
kind = "ollama"

[[route]]
chain = ["missing"]
`)
	cfg, _, _ := load(path)
	if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), "missing") {
		t.Fatalf("err = %v", err)
	}
}

func TestSecretsComeFromTheEnvironmentOnlyHere(t *testing.T) {
	path := write(t, `
[[provider]]
id = "claude"
kind = "anthropic"
`)
	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cfg.resolveSecrets(func(k string) string {
		if k == "ANTHROPIC_API_KEY" {
			return "from-env"
		}
		return ""
	}, true); err != nil {
		t.Fatal(err)
	}
	if cfg.Providers[0].APIKey != "from-env" {
		t.Fatalf("key = %q", cfg.Providers[0].APIKey)
	}
}

func TestAVendorsKeyIsSentOnlyToThatVendor(t *testing.T) {
	cfg := &Config{Providers: []Provider{
		{ID: "vendor", Kind: "anthropic"},
		{ID: "elsewhere", Kind: "anthropic", BaseURL: "https://collector.example"},
		{ID: "named", Kind: "anthropic", BaseURL: "https://gateway.example", APIKeyEnv: "ANTHROPIC_API_KEY"},
		{ID: "written-out", Kind: "anthropic", BaseURL: "https://api.anthropic.com"},
		{ID: "resource", Kind: "azure", BaseURL: "https://mine.openai.azure.com/"},
		{ID: "account", Kind: "cloudflare", BaseURL: "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"},
		{ID: "cleartext", Kind: "openai", BaseURL: "http://api.openai.com/v1"},
		{ID: "lookalike", Kind: "openai", BaseURL: "https://api.openai.com.collector.example/v1"},
		{ID: "typed", Kind: "google", BaseURL: "https://gateway.example", typed: true},
		{ID: "regional", Kind: "openai", BaseURL: "https://eu.api.openai.com/v1"},
		{ID: "sovereign", Kind: "azure", BaseURL: "https://mine.openai.azure.us/"},
		{ID: "named-resource", Kind: "azure", Options: map[string]string{"resource": "mine"}},
		{ID: "spliced", Kind: "azure", Options: map[string]string{"resource": "collector.example/"}},
	}}
	notes, err := cfg.resolveSecrets(func(k string) string {
		switch k {
		case "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_API_KEY", "CLOUDFLARE_API_TOKEN":
			return "sk-operator"
		}
		return ""
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []bool{true, false, true, true, true, true, false, false, true, true, true, true, false} {
		p := cfg.Providers[i]
		if got := p.APIKey == "sk-operator"; got != want {
			t.Errorf("%s: key sent = %v, want %v: a vendor's usual variable goes to that vendor's own host, to one the file names it for, or to one the operator typed", p.ID, got, want)
		}
	}
	if len(notes) != 4 || !strings.Contains(notes[0], `"elsewhere"`) || !strings.Contains(notes[0], "ANTHROPIC_API_KEY") ||
		!strings.Contains(notes[0], "api_key_env") {
		t.Fatalf("notes = %q: a key kept back leaves a backend that refuses every call, and nothing said why", notes)
	}
	if !strings.Contains(notes[3], `"spliced"`) || !strings.Contains(notes[3], "options.resource") {
		t.Fatalf("note = %q: an Azure resource is spliced into the host as written, so it is where the key would have gone", notes[3])
	}
}

// The binary's rule for OpenAI's own hosts is the adapter's: a base the
// adapter speaks OpenAI's full dialect to is one OPENAI_API_KEY is sent to.
func TestOpenAIsOwnHostsAreTheAdaptersOwn(t *testing.T) {
	for _, base := range []string{
		"https://api.openai.com/v1",
		"https://eu.api.openai.com/v1",
		"https://us.api.openai.com/v1",
		"https://API.OpenAI.com/v1",
		"http://api.openai.com/v1",
		"https://api.openai.com.collector.example/v1",
		"https://evilapi.openai.com/v1",
		"https://collector.example/v1",
	} {
		p, err := openai.New(openai.Options{BaseURL: base, APIKey: "k"})
		if err != nil {
			t.Fatal(err)
		}
		own := p.Capabilities().Sessions
		if got := vendorHost("openai", base); got != own {
			t.Errorf("%s: the binary takes it for OpenAI's own host = %v, the adapter = %v", base, got, own)
		}
	}
}

func TestANamedKeyVariableThatIsEmptyIsAnError(t *testing.T) {
	env := func(k string) string {
		if k == "SET_KEY" {
			return "k"
		}
		return ""
	}
	server := &Config{Server: Server{APIKeyEnv: "UNSET_KEY"}}
	if _, err := server.resolveSecrets(env, true); err == nil || !strings.Contains(err.Error(), "UNSET_KEY") {
		t.Errorf("err = %v: the server's variable found empty was read as no key, which leaves the proxy open", err)
	}
	cfg := &Config{
		Server:    Server{APIKeyEnv: "SET_KEY"},
		Providers: []Provider{{ID: "p", Kind: "anthropic", APIKeyEnv: "SET_KEY"}},
	}
	if _, err := cfg.resolveSecrets(env, true); err != nil {
		t.Fatal(err)
	}
	if cfg.Server.APIKey != "k" || cfg.Providers[0].APIKey != "k" {
		t.Fatalf("server = %q provider = %q", cfg.Server.APIKey, cfg.Providers[0].APIKey)
	}
	stdio := &Config{Server: Server{APIKeyEnv: "UNSET_KEY"}}
	if _, err := stdio.resolveSecrets(env, false); err != nil {
		t.Fatalf("err = %v: goose mcp binds no port, and the proxy's key is not its to need", err)
	}
}

// An MCP client starts goose mcp with little of the operator's environment, so
// a backend's variable missing there is one backend without its key, not a
// reason to serve nothing.
func TestAProviderKeyVariableThatIsEmptyIsSaidNotFatal(t *testing.T) {
	for _, serving := range []bool{true, false} {
		cfg := &Config{Providers: []Provider{
			{ID: "claude", Kind: "anthropic", APIKeyEnv: "UNSET_KEY"},
			{ID: "local", Kind: "ollama"},
		}}
		notes, err := cfg.resolveSecrets(func(string) string { return "" }, serving)
		if err != nil {
			t.Fatalf("serving = %v: err = %v: every route went unserved for one backend's key", serving, err)
		}
		if len(notes) != 1 || !strings.Contains(notes[0], `"claude"`) || !strings.Contains(notes[0], "UNSET_KEY") {
			t.Fatalf("serving = %v: notes = %q: the backend that will fail, and the variable it wants, went unnamed", serving, notes)
		}
		if cfg.Providers[0].APIKey != "" {
			t.Fatalf("key = %q", cfg.Providers[0].APIKey)
		}
	}
}

func TestAKeyOrRetryAKindNeverUsesIsRefused(t *testing.T) {
	sound := func(p Provider) *Config {
		return &Config{Server: Server{Host: "127.0.0.1", Port: 1}, Formats: []string{"openai"}, Providers: []Provider{p}}
	}
	for name, p := range map[string]Provider{
		"a key on bedrock":                   {ID: "b", Kind: "bedrock", APIKey: "k"},
		"a key variable on bedrock":          {ID: "b", Kind: "bedrock", APIKeyEnv: "AWS_BEARER_TOKEN_BEDROCK"},
		"a key on vertex":                    {ID: "v", Kind: "vertex", Model: "gemini-2.5-pro", APIKey: "k"},
		"a key a CLI is not handed":          {ID: "c", Kind: "claude", APIKey: "k"},
		"a key variable a CLI is not handed": {ID: "c", Kind: "claude", APIKeyEnv: "ANTHROPIC_API_KEY"},
		"retries on a CLI":                   {ID: "c", Kind: "codex", MaxRetries: 3},
	} {
		if err := sound(p).validate(true); err == nil {
			t.Errorf("%s was accepted, and would look applied and not be", name)
		}
	}
	for name, p := range map[string]Provider{
		"a key a CLI is handed": {ID: "c", Kind: "claude", APIKey: "k",
			Options: map[string]string{"api_key_env_name": "ANTHROPIC_API_KEY"}},
		"retries on bedrock": {ID: "b", Kind: "bedrock", MaxRetries: 3},
		"a key on anthropic": {ID: "a", Kind: "anthropic", APIKeyEnv: "ANTHROPIC_API_KEY"},
	} {
		if err := sound(p).validate(true); err != nil {
			t.Errorf("%s was refused: %v", name, err)
		}
	}
}

func TestBuildsAClientFromAFile(t *testing.T) {
	path := write(t, `
[[provider]]
id = "local"
kind = "ollama"
model = "llama3.1:8b"

[[provider]]
id = "claude"
kind = "anthropic"
api_key = "k"

[[route]]
name = "claude"
models = ["claude-*"]
chain = ["claude"]

[[route]]
name = "rest"
chain = ["local"]
`)
	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err != nil {
		t.Fatal(err)
	}
	client, err := build(cfg, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := client.Providers(); len(got) != 2 {
		t.Fatalf("providers = %v", got)
	}
}

func TestCompatibleBackendNeedsABaseURL(t *testing.T) {
	if _, err := newProvider(Provider{ID: "x", Kind: "compatible"}); err == nil {
		t.Fatal("a compatible backend with no URL has nowhere to send anything")
	}
	if _, err := newProvider(Provider{ID: "x", Kind: "compatible", BaseURL: "http://x", Dialect: "nonsense"}); err == nil {
		t.Fatal("an unknown dialect feature must be refused rather than ignored")
	}
}

func TestAWSCredentialsPreferExplicitThenEnvThenFile(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "credentials")
	if err := os.WriteFile(file, []byte("[work]\naws_access_key_id = FILE\naws_secret_access_key = FILESECRET\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(k string) string {
		switch k {
		case "AWS_SHARED_CREDENTIALS_FILE":
			return file
		case "AWS_PROFILE":
			return "work"
		default:
			return ""
		}
	}
	creds, err := awsCredentials(Provider{}, env)(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if creds.AccessKeyID != "FILE" {
		t.Fatalf("credentials = %+v", creds)
	}

	withEnv := func(k string) string {
		if k == "AWS_ACCESS_KEY_ID" {
			return "ENV"
		}
		if k == "AWS_SECRET_ACCESS_KEY" {
			return "ENVSECRET"
		}
		return env(k)
	}
	creds, err = awsCredentials(Provider{}, withEnv)(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if creds.AccessKeyID != "ENV" {
		t.Fatalf("the environment must win over the file: %+v", creds)
	}

	explicit := Provider{Options: map[string]string{"access_key_id": "CFG", "secret_access_key": "CFGSECRET"}}
	creds, err = awsCredentials(explicit, withEnv)(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if creds.AccessKeyID != "CFG" {
		t.Fatalf("the file must win over both: %+v", creds)
	}
}

func TestAWSCredentialsNameTheProfileThatFailed(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "credentials")
	_ = os.WriteFile(file, []byte("[other]\naws_access_key_id = X\naws_secret_access_key = Y\n"), 0o600)
	env := func(k string) string {
		switch k {
		case "AWS_SHARED_CREDENTIALS_FILE":
			return file
		case "AWS_EC2_METADATA_DISABLED":
			return "true"
		}
		return ""
	}
	_, err := awsCredentials(Provider{}, env)(context.Background())
	if err == nil || !strings.Contains(err.Error(), "default") {
		t.Fatalf("err = %v: the error must name the profile that was tried", err)
	}
}

func TestGoogleTokenPrefersAnExplicitOne(t *testing.T) {
	src, _, err := googleToken(Provider{Options: map[string]string{"access_token": "tok"}}, func(string) string { return "" }, os.ReadFile)
	if err != nil {
		t.Fatal(err)
	}
	got, err := src.Token(context.Background())
	if err != nil || got != "tok" {
		t.Fatalf("token = %q err = %v", got, err)
	}
}

func TestGoogleTokenReportsABadKeyFile(t *testing.T) {
	path := write(t, "not json")
	_, _, err := googleToken(Provider{Options: map[string]string{"credentials_file": path}}, func(string) string { return "" }, os.ReadFile)
	if err == nil {
		t.Fatal("a key file that cannot be read must fail at startup, not at the first request")
	}
}

func TestCloudKindsBuild(t *testing.T) {
	cases := []Provider{
		{ID: "az", Kind: "azure", APIKey: "k", Options: map[string]string{"resource": "r"}},
		{ID: "cf", Kind: "cloudflare", APIKey: "k", Options: map[string]string{"account_id": "a"}},
		{ID: "or", Kind: "openrouter", APIKey: "k"},
	}
	for _, p := range cases {
		if _, err := newProvider(p); err != nil {
			t.Fatalf("%s: %v", p.Kind, err)
		}
	}
	if _, err := newProvider(Provider{ID: "b", Kind: "bedrock"}); err == nil {
		t.Fatal("bedrock without a region cannot build a URL")
	}
}

// A provider that fails to build must stop goose with a message naming the
// provider, not just the library's bare error, so an operator with several
// providers of one kind knows which entry and setting to fix.
func TestAProviderBuildErrorNamesTheProvider(t *testing.T) {
	cfg := &Config{
		Server:  Server{Host: "127.0.0.1", Port: 1},
		Formats: []string{"openai"},
		Providers: []Provider{
			{ID: "sound", Kind: "ollama"},
			{ID: "no-region", Kind: "bedrock"},
		},
	}
	_, err := build(cfg, nil, nil, nil)
	if err == nil || !strings.Contains(err.Error(), `"no-region"`) {
		t.Fatalf("err = %v: the provider that failed to build went unnamed", err)
	}
}

func TestCLIKindsBuildAndScrubTheEnvironment(t *testing.T) {
	t.Setenv("GOOSE_PROXY_KEY", "secret")
	p, err := newProvider(Provider{ID: "claude", Kind: "claude", Options: map[string]string{"max_concurrent": "2"}})
	if err != nil {
		t.Fatal(err)
	}
	if p.ID() != "claude" {
		t.Fatalf("id = %q", p.ID())
	}
	if _, ok := p.(interface {
		Acquire(context.Context) (func(), error)
	}); !ok {
		t.Fatal("a subscription backend must be able to shed load")
	}
	if _, ok := p.(interface{ Close(context.Context) error }); !ok {
		t.Fatal("a provider that owns child processes must be closable")
	}
}

func TestToolGroupMustHaveAServer(t *testing.T) {
	path := write(t, `
[[provider]]
id = "a"
kind = "ollama"

[[route]]
chain = ["a"]
tools = ["mcp"]
`)
	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), "mcp") {
		t.Fatalf("err = %v: a route cannot advertise tools nothing serves", err)
	}
}

func TestMCPServerNeedsATransport(t *testing.T) {
	path := write(t, `
[[provider]]
id = "a"
kind = "ollama"

[[mcp]]
name = "files"
`)
	cfg, _, _ := load(path)
	if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), "command or a url") {
		t.Fatalf("err = %v", err)
	}
}

func TestValidateRefusesAServerItCannotHonour(t *testing.T) {
	base := func() *Config {
		return &Config{
			Server:    Server{Host: "127.0.0.1", Port: 1},
			Formats:   []string{"openai"},
			Providers: []Provider{{ID: "p", Kind: "ollama"}},
		}
	}
	for name, mangle := range map[string]func(*Config){
		"negative body limit":  func(c *Config) { c.Server.MaxBodyMB = -1 },
		"negative concurrency": func(c *Config) { c.Server.MaxConcurrent = -1 },
		"negative rate":        func(c *Config) { c.Server.MaxPerMinute = -1 },
		"negative frame limit": func(c *Config) { c.Server.MaxFrameMB = -1 },
		"certificate alone":    func(c *Config) { c.Server.TLSCert = "cert.pem" },
		"key alone":            func(c *Config) { c.Server.TLSKey = "key.pem" },
		"missing certificate": func(c *Config) {
			c.Server.TLSCert = filepath.Join(t.TempDir(), "absent.pem")
			c.Server.TLSKey = filepath.Join(t.TempDir(), "absent.key")
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := base()
			mangle(cfg)
			if err := cfg.validate(true); err == nil {
				t.Fatal("accepted a configuration it cannot honour")
			}
		})
	}
	if err := base().validate(true); err != nil {
		t.Fatalf("a sound configuration was refused: %v", err)
	}
}

func TestConnectToolsFailsTheStartOnAServerThatWillNotStart(t *testing.T) {
	cfg := &Config{MCP: []MCP{{Name: "files", Command: "/nonexistent/mcp-server"}}}
	m, err := connectTools(context.Background(), cfg)
	if err == nil {
		if m != nil {
			_ = m.Close()
		}
		t.Fatal("a server that cannot start was accepted")
	}
	if !strings.Contains(err.Error(), "files") {
		t.Fatalf("error = %v: the operator needs to know which one", err)
	}
}

func TestConnectToolsWithNothingConfiguredStartsNothing(t *testing.T) {
	m, err := connectTools(context.Background(), &Config{})
	if err != nil {
		t.Fatal(err)
	}
	if m != nil {
		t.Fatal("a file that names no tool server gets no manager")
	}
}

func TestToolServerEnvironmentIsScrubbed(t *testing.T) {
	t.Setenv("GOOSE_PROXY_KEY", "secret-proxy-key")
	t.Setenv("ANTHROPIC_API_KEY", "secret-upstream-key")
	t.Setenv("PATH", "/usr/bin")
	cfg := &Config{MCP: []MCP{{Name: "files", Command: "/nonexistent/mcp-server"}}}
	_, _ = connectTools(context.Background(), cfg)

	env := spawn.ScrubEnv(os.Environ(), defaultToolEnv, nil)
	joined := strings.Join(env, "\n")
	if strings.Contains(joined, "secret-proxy-key") || strings.Contains(joined, "secret-upstream-key") {
		t.Fatalf("a tool server would receive the proxy's own credentials: %v", env)
	}
	if !strings.Contains(joined, "PATH=") {
		t.Fatalf("a tool server needs a PATH to find anything: %v", env)
	}
}

func TestValidateRefusesADurationThatDoesNotParse(t *testing.T) {
	sound := func() *Config {
		return &Config{
			Server:    Server{Host: "127.0.0.1", Port: 1},
			Formats:   []string{"openai"},
			Providers: []Provider{{ID: "p", Kind: "ollama"}},
		}
	}
	for name, mangle := range map[string]func(*Config){
		"server cooldown": func(c *Config) { c.Server.BreakerCooldown = "30 seconds" },
		"provider":        func(c *Config) { c.Providers[0].Timeout = "2 minutes" },
		"mcp": func(c *Config) {
			c.MCP = []MCP{{Name: "files", Command: "x", Timeout: "forever"}}
		},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := sound()
			mangle(cfg)
			err := cfg.validate(true)
			if err == nil {
				t.Fatal("accepted a duration it cannot read")
			}
			if !strings.Contains(err.Error(), "not a duration") {
				t.Fatalf("error = %v", err)
			}
		})
	}
	for _, good := range []string{"", "30s", "2m", "1h30m"} {
		cfg := sound()
		cfg.Server.BreakerCooldown = good
		if err := cfg.validate(true); err != nil {
			t.Fatalf("%q was refused: %v", good, err)
		}
	}
}

func TestRouteAndPassthroughNamesAreChecked(t *testing.T) {
	base := func() *Config {
		return &Config{
			Server:    Server{Host: "127.0.0.1", Port: 11234},
			Formats:   []string{"openai"},
			Providers: []Provider{{ID: "up", Kind: "openai"}},
			Routes:    []Route{{Name: "r", Chain: []string{"up"}}},
		}
	}
	for name, mutate := range map[string]func(*Config){
		"unknown format": func(c *Config) { c.Routes[0].Formats = []string{"opnai"} },
		"empty header":   func(c *Config) { c.Routes[0].Header = []string{"=x"} },
		"unknown passthrough": func(c *Config) {
			c.Server.Passthrough = "nowhere"
		},
	} {
		cfg := base()
		mutate(cfg)
		if err := cfg.validate(true); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if err := base().validate(true); err != nil {
		t.Fatalf("a sound configuration was refused: %v", err)
	}
}

func TestNamingAFormatTwiceServesItOnce(t *testing.T) {
	for _, names := range [][]string{
		{"ollama", "ollama-generate"},
		{"ollama-generate", "ollama"},
		{"openai", "openai"},
	} {
		codecs, err := buildCodecs(names)
		if err != nil {
			t.Fatalf("%v: %v", names, err)
		}
		seen := map[string]bool{}
		for _, c := range codecs {
			if seen[c.Name()] {
				t.Errorf("%v: %s is served twice", names, c.Name())
			}
			seen[c.Name()] = true
		}
	}
}

func TestARoutePolicyNamesCodesThatExist(t *testing.T) {
	cfg := &Config{
		Server:    Server{Host: "127.0.0.1", Port: 11234},
		Formats:   []string{"openai"},
		Providers: []Provider{{ID: "up", Kind: "openai"}},
		Routes:    []Route{{Name: "r", Chain: []string{"up"}, Allow: []string{"drop.thinkng"}}},
	}
	if err := cfg.validate(true); err == nil {
		t.Error("a misspelt code was accepted")
	}
	cfg.Routes[0].Allow = []string{llm.DropThinking}
	cfg.Routes[0].Deny = []string{"wire.no_finsh_reason"}
	if err := cfg.validate(true); err == nil {
		t.Error("a misspelt denial was accepted")
	}
	cfg.Routes[0].Deny = []string{llm.WireNoFinish}
	if err := cfg.validate(true); err != nil {
		t.Errorf("a sound policy was refused: %v", err)
	}
}

func TestAConfiguredModelIsAPin(t *testing.T) {
	pinned := map[string]string{"up": "pinned"}

	rule, err := newRule(Route{Name: "r", Chain: []string{"up"}}, pinned, false)
	if err != nil {
		t.Fatal(err)
	}
	if rule.Chain[0].Model != "pinned" {
		t.Errorf("model = %q: a provider pinned to one model served another", rule.Chain[0].Model)
	}

	rule, err = newRule(Route{Name: "r", Chain: []string{"up:asked-for"}}, pinned, false)
	if err != nil {
		t.Fatal(err)
	}
	if rule.Chain[0].Model != "asked-for" {
		t.Errorf("model = %q", rule.Chain[0].Model)
	}

	rule, err = newRule(Route{Name: "r", Model: "rewritten", Chain: []string{"up"}}, pinned, false)
	if err != nil {
		t.Fatal(err)
	}
	if rule.Chain[0].Model != "" || rule.Model != "rewritten" {
		t.Errorf("chain = %q rule = %q", rule.Chain[0].Model, rule.Model)
	}
}

func TestVertexTakesTheProjectFromTheKeyItWasGiven(t *testing.T) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(priv)
	if err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]string{
		"type": "service_account", "project_id": "from-the-key",
		"client_email": "robot@example.iam.gserviceaccount.com",
		"private_key":  string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"token_uri":    "https://oauth2.googleapis.com/token",
	})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "key.json")
	if err := os.WriteFile(path, body, 0o600); err != nil {
		t.Fatal(err)
	}

	t.Setenv("GOOGLE_CLOUD_PROJECT", "")
	t.Setenv("GCP_PROJECT", "")
	p := Provider{ID: "v", Kind: "vertex", Model: "gemini-2.0-flash",
		Options: map[string]string{"credentials_file": path}}
	if _, err := newProvider(p); err != nil {
		t.Fatalf("vertex: %v: the key names the project", err)
	}
}

func TestOptionsAreCheckedLikeEveryOtherKey(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"a misspelt option", `
[[provider]]
id = "b"
kind = "bedrock"
options = { regoin = "us-east-1" }
`},
		{"an option for a kind that reads none", `
[[provider]]
id = "a"
kind = "anthropic"
options = { project = "p" }
`},
		{"a whole number that is not one", `
[[provider]]
id = "c"
kind = "claude"
options = { max_turns = "lots" }
`},
		{"a flag that is neither true nor false", `
[[provider]]
id = "o"
kind = "openrouter"
options = { allow_fallbacks = "no" }
`},
		{"a setting the kind never looks at", `
[[provider]]
id = "o"
kind = "openai"
betas = ["context-1m"]
`},
		{"an unknown tool transport", `
[[provider]]
id = "a"
kind = "anthropic"
[[mcp]]
name = "files"
transport = "stdioo"
command = "mcp-server-filesystem"
`},
		{"a stdio server with no command", `
[[provider]]
id = "a"
kind = "anthropic"
[[mcp]]
name = "files"
transport = "stdio"
url = "http://127.0.0.1:9"
`},
	}
	for _, c := range cases {
		cfg, _, err := load(write(t, c.body))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if err := cfg.validate(true); err == nil {
			t.Fatalf("%s was accepted", c.name)
		}
	}
}

func TestOptionsAKindDoesReadAreAccepted(t *testing.T) {
	cfg, _, err := load(write(t, `
[[provider]]
id = "b"
kind = "bedrock"
options = { region = "us-east-1", profile = "work" }

[[provider]]
id = "c"
kind = "claude"
options = { max_turns = "4", system_mode = "file" }

[[provider]]
id = "o"
kind = "openrouter"
options = { allow_fallbacks = "false", providers = "a,b" }

[[route]]
name = "all"
chain = ["b", "c", "o"]

[[mcp]]
name = "files"
command = "mcp-server-filesystem"
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err != nil {
		t.Fatal(err)
	}
}

// A base_url with no scheme parses without complaint and then fails every
// call at request time, a 400 that blames the caller for the file's mistake;
// -check must catch it instead, naming the provider and the setting.
func TestBaseURLWithNoSchemeIsRefused(t *testing.T) {
	for name, base := range map[string]string{
		"a bare host:port, ollama's own convention": "127.0.0.1:11434",
		"a bare host": "example.com",
	} {
		cfg, _, err := load(write(t, `
[[provider]]
id = "local"
kind = "ollama"
base_url = "`+base+`"
`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), `"local"`) ||
			!strings.Contains(err.Error(), "base_url") || !strings.Contains(err.Error(), "scheme") {
			t.Fatalf("%s: err = %v: must name the provider, base_url and its missing scheme", name, err)
		}
	}
	// An agent CLI's base_url is the binary's path, not a URL, and must not be
	// refused for having no scheme.
	cfg, _, err := load(write(t, `
[[provider]]
id = "cli"
kind = "claude"
base_url = "/usr/local/bin/claude"
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err != nil {
		t.Fatalf("an agent CLI's base_url is a binary path, not refused for having no scheme: %v", err)
	}
}

func TestClientKeyOnAKindThatIgnoresItIsRefused(t *testing.T) {
	cfg, _, err := load(write(t, `
[[provider]]
id = "local"
kind = "ollama"
allow_client_key = true
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err == nil {
		t.Fatal("a kind that never reads it accepted it")
	}
}

func TestHeadersReachTheUpstream(t *testing.T) {
	var seen http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		_, _ = w.Write([]byte(`{"id":"c1","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()

	p, err := newProvider(Provider{
		ID: "up", Kind: "compatible", BaseURL: srv.URL, APIKey: "k",
		Headers: map[string]string{"X-Tenant": "acme", "Authorization": "not-this"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.(interface {
		Complete(context.Context, *llm.Attempt) (*llm.Response, error)
	}).Complete(context.Background(), &llm.Attempt{
		Model: "m", Recorder: llm.NopRecorder{},
		Request: &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}}},
	}); err != nil {
		t.Fatal(err)
	}
	if seen.Get("X-Tenant") != "acme" {
		t.Fatalf("headers = %v: the file named one and it did not travel", seen)
	}
	if seen.Get("Authorization") != "Bearer k" {
		t.Fatalf("authorization = %q: a header from the file cannot replace the credential", seen.Get("Authorization"))
	}
}

// A caller's own beta flags travel under the operator's key unless the file
// keeps them back, and only the kind that sends them reads the setting.
func TestACallersBetasTravelUnlessTheFileKeepsThemBack(t *testing.T) {
	var seen http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		_, _ = w.Write([]byte(`{"id":"m","model":"m","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{}}`))
	}))
	defer srv.Close()

	off := false
	for name, c := range map[string]struct {
		set  *bool
		want string
	}{"left out": {nil, "files-api-2025-04-14"}, "false": {&off, ""}} {
		p, err := newProvider(Provider{ID: "a", Kind: "anthropic", BaseURL: srv.URL, APIKey: "k", ClientBeta: c.set})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := p.(interface {
			Complete(context.Context, *llm.Attempt) (*llm.Response, error)
		}).Complete(context.Background(), &llm.Attempt{
			Model: "m", Recorder: llm.NopRecorder{},
			Request: &llm.Request{
				Betas:    []string{"files-api-2025-04-14"},
				Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{&llm.Text{Text: "hi"}}}},
			},
		}); err != nil {
			t.Fatal(err)
		}
		if got := strings.Join(seen.Values("anthropic-beta"), ","); got != c.want {
			t.Fatalf("client_betas %s: anthropic-beta = %q, want %q", name, got, c.want)
		}
	}

	cfg, _, err := load(write(t, `
[[provider]]
id = "o"
kind = "openai"
client_betas = false
`))
	if err != nil {
		t.Fatal(err)
	}
	if err := cfg.validate(true); err == nil {
		t.Fatal("a kind that never sends a caller's betas accepted a setting for them")
	}
}

func TestANamelessHeaderIsRefused(t *testing.T) {
	for _, body := range []string{`
[[provider]]
id = "a"
kind = "anthropic"
headers = { "" = "x" }
`, `
[[provider]]
id = "a"
kind = "anthropic"
[[mcp]]
name = "files"
url = "https://example.test/mcp"
headers = { " " = "x" }
`, `
[[provider]]
id = "c"
kind = "claude"
headers = { "X-Foo" = "bar" }
`} {
		cfg, _, err := load(write(t, body))
		if err != nil {
			t.Fatal(err)
		}
		if err := cfg.validate(true); err == nil {
			t.Fatalf("accepted: %s", body)
		}
	}
}

func TestAContainerEndpointThatIsNotTheRuntimesIsRefused(t *testing.T) {
	for _, uri := range []string{
		"http://169.254.169.254/creds",
		"http://10.0.0.1/creds",
		"http://elsewhere.example/creds",
		"http://169.254.170.2@elsewhere/c",
		"ftp://169.254.170.2/creds",
	} {
		if got, err := containerURI(uri); err == nil {
			t.Fatalf("%s was allowed as %q", uri, got)
		}
	}
	for _, uri := range []string{
		"http://169.254.170.2/v2/credentials",
		"http://169.254.170.23/v1/credentials",
		"http://[fd00:ec2::23]/v1/credentials",
		"http://127.0.0.1:8080/creds",
		"http://localhost:8080/creds",
		"https://vault.example/creds",
	} {
		if got, err := containerURI(uri); err != nil || got != uri {
			t.Fatalf("%s = %q, %v", uri, got, err)
		}
	}
	if got, err := containerURI(""); err != nil || got != "" {
		t.Fatalf("an unset variable is not a refusal: %q, %v", got, err)
	}
}

func TestARefusedContainerEndpointIsReportedWithoutItsToken(t *testing.T) {
	env := func(k string) string {
		switch k {
		case "AWS_CONTAINER_CREDENTIALS_FULL_URI":
			return "http://elsewhere.example/creds"
		case "AWS_CONTAINER_AUTHORIZATION_TOKEN":
			return "the-tasks-token"
		case "AWS_SHARED_CREDENTIALS_FILE":
			return filepath.Join(t.TempDir(), "none")
		case "AWS_EC2_METADATA_DISABLED":
			return "true"
		}
		return ""
	}
	_, err := awsCredentials(Provider{}, env)(context.Background())
	if err == nil || !strings.Contains(err.Error(), "AWS_CONTAINER_CREDENTIALS_FULL_URI") {
		t.Fatalf("err = %v: the refused variable must be named", err)
	}
	if strings.Contains(err.Error(), "the-tasks-token") {
		t.Fatalf("the token reached the error: %v", err)
	}
}

func TestEveryAttemptGetsADeadlineAndACeiling(t *testing.T) {
	got := limitsOf(Server{})
	if got.Timeout <= 0 {
		t.Fatal("an attempt with no deadline holds its request, connection and slot until the client gives up")
	}
	if got.MaxResponseBytes <= 0 {
		t.Fatal("an upstream that never stops sending is not answering a question anyone asked")
	}

	set := limitsOf(Server{AttemptTimeout: "90s", MaxResponseMB: 8})
	if set.Timeout != 90*time.Second {
		t.Fatalf("timeout = %v", set.Timeout)
	}
	if set.MaxResponseBytes != 8<<20 {
		t.Fatalf("ceiling = %d", set.MaxResponseBytes)
	}

	if bad := limitsOf(Server{AttemptTimeout: "not a duration"}); bad.Timeout != defAttempt {
		t.Fatalf("timeout = %v: an unreadable duration falls back rather than disabling the deadline", bad.Timeout)
	}
}

func TestTheStreamLineCeilingCanBeRaised(t *testing.T) {
	if got := limitsOf(Server{}).MaxFrameBytes; got != 0 {
		t.Fatalf("frame = %d: left out, the provider's own ceiling applies", got)
	}
	if got := limitsOf(Server{MaxFrameMB: 16}).MaxFrameBytes; got != 16<<20 {
		t.Fatalf("frame = %d: a streamed image is one line, and the operator had no way to make room for it", got)
	}
}

func TestAnAgentGateIsOffOnlyWhenAskedFor(t *testing.T) {
	if got := agentGate(""); got != 0 {
		t.Fatalf("unset = %d: leaving the key out asks for the provider's default", got)
	}
	if got := agentGate("0"); got >= 0 {
		t.Fatalf("zero = %d: in a config file zero is how an operator removes a limit", got)
	}
	if got := agentGate("8"); got != 8 {
		t.Fatalf("eight = %d", got)
	}
}

func TestDisablingInstanceMetadataIsNotUndoneByAContainerRole(t *testing.T) {
	var reachedIMDS atomic.Bool
	imds := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reachedIMDS.Store(true)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer imds.Close()
	task := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer task.Close()

	env := map[string]string{
		"AWS_CONTAINER_CREDENTIALS_FULL_URI": task.URL + "/creds",
		"AWS_EC2_METADATA_DISABLED":          "true",
		"AWS_EC2_METADATA_SERVICE_ENDPOINT":  imds.URL,
	}
	src := awsCredentials(Provider{}, func(k string) string { return env[k] })
	if _, err := src(context.Background()); err == nil {
		t.Fatal("the task role refused, so resolving credentials must fail")
	}
	if reachedIMDS.Load() {
		t.Fatal("the node's own role was asked for on a host where the operator turned it off")
	}
}

func TestARouteCanBeLenientUnderAStrictServer(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "goose.toml")
	body := `
formats = ["openai"]
[server]
strict = true
[[provider]]
id = "p"
kind = "ollama"
[[route]]
name = "lenient"
models = ["*"]
strict = false
chain = ["p"]
`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Routes) != 1 || cfg.Routes[0].Strict == nil || *cfg.Routes[0].Strict {
		t.Fatalf("route = %+v: a route that says strict = false has said something", cfg.Routes)
	}
	rule, err := newRule(cfg.Routes[0], nil, cfg.Server.Strict)
	if err != nil {
		t.Fatal(err)
	}
	if !rule.PolicySet {
		t.Fatal("the route chose a policy, so the server default must not overrule it")
	}
	if rule.Policy.Strict {
		t.Fatalf("policy = %+v", rule.Policy)
	}
}

// One Vertex entry serves one publisher, and which one comes from the model
// name; a config that names neither fails at startup rather than on first call.
func TestVertexNeedsAModelOrAPublisher(t *testing.T) {
	cfg := defaults()
	cfg.Providers = []Provider{
		{ID: "vx", Kind: "vertex", Options: map[string]string{"project": "p", "region": "us-east5"}},
	}
	if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), "publisher") {
		t.Fatalf("err = %v", err)
	}
	cfg.Providers[0].Options["publisher"] = "anthropic"
	if err := cfg.validate(true); err != nil {
		t.Fatalf("a named publisher is enough: %v", err)
	}
	cfg.Providers[0].Options["publisher"] = ""
	cfg.Providers[0].Model = "gemini-2.5-pro"
	if err := cfg.validate(true); err != nil {
		t.Fatalf("a model names its publisher: %v", err)
	}
}

// A route that names codes to allow or deny has chosen a policy, so the
// server's is not put in its place; the strictness it says nothing of is
// still the server's, or a strict server serves a lossy call on that route.
func TestARouteThatOnlyNamesCodesKeepsAStrictServer(t *testing.T) {
	cfg, _, err := load(write(t, `
formats = ["openai"]
[server]
strict = true
[[provider]]
id = "p"
kind = "ollama"
[[route]]
name = "allowing"
models = ["a-*"]
chain = ["p"]
allow = ["wire.no_finish_reason"]
[[route]]
name = "denying"
chain = ["p"]
deny = ["drop.image"]
`))
	if err != nil {
		t.Fatal(err)
	}
	rules, err := rulesOf(cfg)
	if err != nil {
		t.Fatal(err)
	}
	table := &route.Table{Rules: rules, Defaults: route.Rule{Policy: llm.LossPolicy{Strict: true}}}
	lost := []llm.Diagnostic{{Code: llm.DropThinking, Class: llm.ClassContent, Severity: llm.SeverityLoss}}
	for _, model := range []string{"a-model", "b-model"} {
		dec, err := table.Route(&route.Call{Model: model})
		if err != nil {
			t.Fatal(err)
		}
		if !dec.Policy.Strict || dec.Policy.Check(lost) == nil {
			t.Fatalf("%s: policy = %+v: server.strict was dropped on a route that never said strict = false", dec.Rule, dec.Policy)
		}
	}

	cfg.Server.Strict = false
	rules, err = rulesOf(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if rules[0].Policy.Strict {
		t.Fatalf("policy = %+v: a lenient server made a route strict", rules[0].Policy)
	}
}

// goose mcp starts in whatever project its client has open, so a program the
// operator's file names by a relative path, and the directory it runs in, are
// beside that file; a bare name is still looked up on PATH.
func TestAProgramTheConfigRunsIsBesideIt(t *testing.T) {
	fake, err := filepath.Abs(filepath.Join("..", "..", "mcp", "testdata", "fake-server.sh"))
	if err != nil {
		t.Fatal(err)
	}
	server, err := os.ReadFile(fake)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "srv.sh"), server, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "goose.toml")
	if err := os.WriteFile(path, []byte(`
[[provider]]
id = "cli"
kind = "claude"
base_url = "./bin/claude"
options = { temp_dir = "scratch" }

[[provider]]
id = "cli-in-dir"
kind = "codex"
base_url = "bin/codex"
options = { dir = "work" }

[[provider]]
id = "cli-on-path"
kind = "copilot"
base_url = "copilot"

[[route]]
chain = ["cli"]

[[mcp]]
name = "files"
command = "./srv.sh"

[[mcp]]
name = "in-dir"
command = "./run.sh"
dir = "tools"

[[mcp]]
name = "on-path"
command = "mcp-server-filesystem"
`), 0o600); err != nil {
		t.Fatal(err)
	}
	project := t.TempDir()
	if err := os.WriteFile(filepath.Join(project, "srv.sh"), []byte("#!/bin/sh\nexit 3\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Chdir(project)

	cfg, _, err := load(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ name, got, want string }{
		{"mcp files command", cfg.MCP[0].Command, filepath.Join(dir, "srv.sh")},
		{"mcp in-dir dir", cfg.MCP[1].Dir, filepath.Join(dir, "tools")},
		{"mcp in-dir command", cfg.MCP[1].Command, filepath.Join(dir, "tools", "run.sh")},
		{"mcp on-path command", cfg.MCP[2].Command, "mcp-server-filesystem"},
		{"cli base_url", cfg.Providers[0].BaseURL, filepath.Join(dir, "bin", "claude")},
		{"cli temp_dir", cfg.Providers[0].Options["temp_dir"], filepath.Join(dir, "scratch")},
		{"cli-in-dir dir", cfg.Providers[1].Options["dir"], filepath.Join(dir, "work")},
		{"cli-in-dir base_url", cfg.Providers[1].BaseURL, filepath.Join(dir, "work", "bin", "codex")},
		{"cli-on-path base_url", cfg.Providers[2].BaseURL, "copilot"},
	} {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}

	cfg.MCP = cfg.MCP[:1]
	tools, err := connectTools(context.Background(), cfg)
	if err != nil {
		t.Fatalf("err = %v: the project's srv.sh ran in place of the one beside the file", err)
	}
	tools.Close()
}

// A remote tool server's url fails at startup without a scheme or a host, so
// -check must refuse it, naming the server but not repeating the url, which
// may carry the server's token.
func TestARemoteToolServerNeedsASchemeAndAHost(t *testing.T) {
	for url, want := range map[string]string{
		"tools.example:8080/mcp?token=ZZSECRET": "scheme",
		"tools.example/mcp":                     "scheme",
		"https:///mcp":                          "no host",
	} {
		cfg := defaults()
		cfg.Providers = []Provider{{ID: "p", Kind: "ollama"}}
		cfg.MCP = []MCP{{Name: "remote", Transport: "http", URL: url}}
		err := cfg.validate(true)
		if err == nil || !strings.Contains(err.Error(), `"remote"`) || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s: err = %v, want the server named and %q", url, err, want)
		}
		if strings.Contains(err.Error(), "ZZSECRET") {
			t.Fatalf("err = %v: the url's token reached the error", err)
		}
	}
	cfg := defaults()
	cfg.Providers = []Provider{{ID: "p", Kind: "ollama"}}
	cfg.MCP = []MCP{{Name: "remote", URL: "https://tools.example/mcp?workspace=w"}}
	if err := cfg.validate(true); err != nil {
		t.Fatalf("a sound url, query and all, was refused: %v", err)
	}
}

// Every call's path is written after base_url as it stands, so a query or a
// fragment there swallows it; and a URL with no host is missing a host, not a
// scheme.
func TestABaseURLThatCannotTakeAPathIsRefused(t *testing.T) {
	for base, want := range map[string]string{
		"http://127.0.0.1:18401?x=1":  "query",
		"http://127.0.0.1:18401/?":    "query",
		"http://127.0.0.1:18401#here": "fragment",
		"http://":                     "no host",
		"https:///v1":                 "no host",
	} {
		cfg := defaults()
		cfg.Providers = []Provider{{ID: "local", Kind: "ollama", BaseURL: base}}
		err := cfg.validate(true)
		if err == nil || !strings.Contains(err.Error(), `"local"`) || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s: err = %v, want %q", base, err, want)
		}
		if want == "no host" && strings.Contains(err.Error(), "scheme") {
			t.Fatalf("%s: err = %v: the scheme is there; the host is what is missing", base, err)
		}
	}
}

// A region is spliced into the host a call and its credential are sent to,
// so anything a region's name lacks could send them somewhere else.
func TestARegionIsARegionsName(t *testing.T) {
	for _, region := range []string{"127.0.0.1:18409/#", "x@127.0.0.1:18409/#", "us-east-1.example.com", "US-EAST-1"} {
		for _, p := range []Provider{
			{ID: "b", Kind: "bedrock", Options: map[string]string{"region": region}},
			{ID: "v", Kind: "vertex", Model: "gemini-2.5-pro", Options: map[string]string{"project": "p", "region": region}},
		} {
			cfg := defaults()
			cfg.Providers = []Provider{p}
			if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), "region") {
				t.Fatalf("%s %q: err = %v", p.Kind, region, err)
			}
		}
	}
	for _, p := range []Provider{
		{ID: "b", Kind: "bedrock", Options: map[string]string{"region": "us-east-1"}},
		{ID: "v", Kind: "vertex", Model: "gemini-2.5-pro", Options: map[string]string{"project": "p", "region": "global"}},
		{ID: "v", Kind: "vertex", Model: "gemini-2.5-pro", Options: map[string]string{"project": "p", "region": "europe-west4"}},
	} {
		cfg := defaults()
		cfg.Providers = []Provider{p}
		if err := cfg.validate(true); err != nil {
			t.Fatalf("%s %q: %v", p.Kind, p.Options["region"], err)
		}
	}

	t.Setenv("AWS_REGION", "x@127.0.0.1:18409/#")
	if _, err := newProvider(Provider{ID: "b", Kind: "bedrock"}); err == nil || !strings.Contains(err.Error(), "region") {
		t.Fatalf("err = %v: AWS_REGION moved bedrock's host", err)
	}
	t.Setenv("GOOGLE_CLOUD_LOCATION", "x@127.0.0.1:18409/#")
	vx := Provider{ID: "v", Kind: "vertex", Model: "gemini-2.5-pro", Options: map[string]string{"project": "p", "access_token": "t"}}
	if _, err := newProvider(vx); err == nil || !strings.Contains(err.Error(), "region") {
		t.Fatalf("err = %v: GOOGLE_CLOUD_LOCATION moved vertex's host", err)
	}
	t.Setenv("GOOGLE_CLOUD_LOCATION", "global")
	if _, err := newProvider(vx); err != nil {
		t.Fatalf("global was refused: %v", err)
	}
}

// An Azure resource is spliced into the host the key goes to, the same way a
// region is, so it is held to a resource's name.
func TestAnAzureResourceIsAResourcesName(t *testing.T) {
	for _, resource := range []string{"x@127.0.0.1:18409/#", "evil.example.com/#", "my_res"} {
		cfg := defaults()
		cfg.Providers = []Provider{{ID: "a", Kind: "azure", Options: map[string]string{"resource": resource}}}
		if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), "resource") {
			t.Fatalf("%q: err = %v", resource, err)
		}
	}
	cfg := defaults()
	cfg.Providers = []Provider{{ID: "a", Kind: "azure", Options: map[string]string{"resource": "my-Resource-01"}}}
	if err := cfg.validate(true); err != nil {
		t.Fatalf("a resource's name was refused: %v", err)
	}
}

// A negative duration would otherwise be swapped for the default without a
// word, while a negative limit beside it is refused.
func TestValidateRefusesANegativeDuration(t *testing.T) {
	for name, mangle := range map[string]func(*Config){
		"server.attempt_timeout":  func(c *Config) { c.Server.AttemptTimeout = "-5m" },
		"server.breaker_cooldown": func(c *Config) { c.Server.BreakerCooldown = "-30s" },
		"provider p timeout":      func(c *Config) { c.Providers[0].Timeout = "-1s" },
		"mcp files timeout":       func(c *Config) { c.MCP = []MCP{{Name: "files", Command: "x", Timeout: "-1s"}} },
	} {
		cfg := defaults()
		cfg.Providers = []Provider{{ID: "p", Kind: "ollama"}}
		mangle(cfg)
		if err := cfg.validate(true); err == nil || !strings.Contains(err.Error(), name+" is negative") {
			t.Errorf("%s: err = %v", name, err)
		}
	}
}

// The library's error names an API caller's kind and the library's own field;
// the operator needs the keys in the file, or the variables, that supply it.
func TestAProviderBuildErrorNamesTheKeysToSet(t *testing.T) {
	for _, name := range []string{"AWS_REGION", "AWS_DEFAULT_REGION", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"} {
		t.Setenv(name, "")
	}
	for _, c := range []struct {
		p    Provider
		want string
	}{
		{Provider{ID: "p1", Kind: "bedrock"}, "options.region or AWS_REGION"},
		{Provider{ID: "p2", Kind: "vertex", Model: "gemini-2.5-pro", Options: map[string]string{"access_token": "t"}}, "options.project or GOOGLE_CLOUD_PROJECT"},
		{Provider{ID: "p3", Kind: "azure"}, "options.resource or base_url"},
		{Provider{ID: "p4", Kind: "cloudflare"}, "options.account_id or base_url"},
		{Provider{ID: "p5", Kind: "azure", Options: map[string]string{"resource": "r", "api_version": "2024-10-21"}}, "api_version needs a deployment"},
	} {
		cfg := &Config{Formats: []string{"openai"}, Providers: []Provider{c.p}}
		_, err := build(cfg, nil, nil, nil)
		if err == nil || !strings.Contains(err.Error(), `provider "`+c.p.ID+`"`) || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err = %v, want %q", c.p.Kind, err, c.want)
			continue
		}
		if strings.Contains(err.Error(), string(llm.KindInvalidRequest)) {
			t.Errorf("%s: err = %v: an API kind tells the operator nothing", c.p.Kind, err)
		}
	}
}

// The manager names the server in its own error, so naming it again only says
// it twice.
func TestAToolServerThatWillNotStartIsNamedOnce(t *testing.T) {
	cfg := &Config{MCP: []MCP{{Name: "files", Command: "/nonexistent/mcp-server"}}}
	m, err := connectTools(context.Background(), cfg)
	if err == nil {
		m.Close()
		t.Fatal("a server that cannot start was accepted")
	}
	if n := strings.Count(err.Error(), "files"); n != 1 {
		t.Fatalf("error = %v: the server is named %d times", err, n)
	}
}
