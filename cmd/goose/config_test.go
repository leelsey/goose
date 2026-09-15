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
	"testing"

	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/llm"
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
	if err := cfg.validate(); err == nil {
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
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "missing") {
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
	cfg.resolveSecrets(func(k string) string {
		if k == "ANTHROPIC_API_KEY" {
			return "from-env"
		}
		return ""
	})
	if cfg.Providers[0].APIKey != "from-env" {
		t.Fatalf("key = %q", cfg.Providers[0].APIKey)
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
	if err := cfg.validate(); err != nil {
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
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "mcp") {
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
	if err := cfg.validate(); err == nil || !strings.Contains(err.Error(), "command or a url") {
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
			if err := cfg.validate(); err == nil {
				t.Fatal("accepted a configuration it cannot honour")
			}
		})
	}
	if err := base().validate(); err != nil {
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
			err := cfg.validate()
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
		if err := cfg.validate(); err != nil {
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
		if err := cfg.validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if err := base().validate(); err != nil {
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
	if err := cfg.validate(); err == nil {
		t.Error("a misspelt code was accepted")
	}
	cfg.Routes[0].Allow = []string{llm.DropThinking}
	cfg.Routes[0].Deny = []string{"wire.no_finsh_reason"}
	if err := cfg.validate(); err == nil {
		t.Error("a misspelt denial was accepted")
	}
	cfg.Routes[0].Deny = []string{llm.WireNoFinish}
	if err := cfg.validate(); err != nil {
		t.Errorf("a sound policy was refused: %v", err)
	}
}

func TestAConfiguredModelIsAPin(t *testing.T) {
	pinned := map[string]string{"up": "pinned"}

	rule, err := newRule(Route{Name: "r", Chain: []string{"up"}}, pinned)
	if err != nil {
		t.Fatal(err)
	}
	if rule.Chain[0].Model != "pinned" {
		t.Errorf("model = %q: a provider pinned to one model served another", rule.Chain[0].Model)
	}

	rule, err = newRule(Route{Name: "r", Chain: []string{"up:asked-for"}}, pinned)
	if err != nil {
		t.Fatal(err)
	}
	if rule.Chain[0].Model != "asked-for" {
		t.Errorf("model = %q", rule.Chain[0].Model)
	}

	rule, err = newRule(Route{Name: "r", Model: "rewritten", Chain: []string{"up"}}, pinned)
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
		if err := cfg.validate(); err == nil {
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
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
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
	if err := cfg.validate(); err == nil {
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
		if err := cfg.validate(); err == nil {
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
