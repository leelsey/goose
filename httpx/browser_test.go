// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func browse(t *testing.T, srv *httptest.Server, path, body string, header map[string]string) (int, map[string]any) {
	t.Helper()
	req, err := http.NewRequest(http.MethodPost, srv.URL+path, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	// A page can send this without asking first, which is what makes it the
	// shape of the attack.
	req.Header.Set("Content-Type", "text/plain")
	for k, v := range header {
		if k == "Host" {
			req.Host = v
			continue
		}
		req.Header.Set(k, v)
	}
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

// A proxy on loopback with no key of its own is reachable by every page the
// operator visits, and spends the operator's credentials for any of them that
// asks. /mcp refuses them for the same reason.
func TestAPageTheOperatorVisitsCannotSpendTheirKey(t *testing.T) {
	chat := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	for _, c := range []struct {
		name   string
		header map[string]string
	}{
		{"another origin", map[string]string{"Origin": "https://evil.example"}},
		{"a cross-site request with no origin", map[string]string{"Sec-Fetch-Site": "cross-site"}},
		{"a name of its own pointed at loopback", map[string]string{"Host": "rebind.evil.example:11234"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := &stub{}
			srv := server(t, p, Options{Host: "127.0.0.1"})
			status, out := browse(t, srv, "/v1/chat/completions", chat, c.header)
			if status != http.StatusForbidden {
				t.Fatalf("status = %d, want 403", status)
			}
			if p.seen != nil {
				t.Fatal("the request reached the backend")
			}
			if inner, _ := out["error"].(map[string]any); inner == nil || inner["message"] == nil {
				t.Fatalf("refusal = %v: the caller's own wire", out)
			}
		})
	}

	t.Run("in the caller's wire", func(t *testing.T) {
		srv := server(t, &stub{}, Options{Host: "127.0.0.1"})
		_, out := browse(t, srv, "/v1/messages",
			`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`,
			map[string]string{"Origin": "https://evil.example"})
		if out["type"] != "error" {
			t.Fatalf("refusal = %v: an Anthropic client reads its own shape", out)
		}
	})

	t.Run("the operator's own client", func(t *testing.T) {
		p := &stub{}
		srv := server(t, p, Options{Host: "127.0.0.1"})
		for _, host := range []string{"", "LOCALHOST:11234", "[::1]:11234", "0.0.0.0:11234", "goose.localhost:11234"} {
			header := map[string]string{"Sec-Fetch-Site": "none"}
			if host != "" {
				header["Host"] = host
			}
			if status, out := browse(t, srv, "/v1/chat/completions", chat, header); status != http.StatusOK {
				t.Fatalf("host %q: status = %d: %v", host, status, out)
			}
		}
	})

	t.Run("an origin the operator named", func(t *testing.T) {
		srv := server(t, &stub{}, Options{Host: "127.0.0.1", CORS: true, CORSOrigin: "https://app.example"})
		header := map[string]string{"Origin": "https://app.example", "Sec-Fetch-Site": "cross-site"}
		if status, out := browse(t, srv, "/v1/chat/completions", chat, header); status != http.StatusOK {
			t.Fatalf("status = %d: %v", status, out)
		}
		header["Origin"] = "https://other.example"
		if status, _ := browse(t, srv, "/v1/chat/completions", chat, header); status != http.StatusForbidden {
			t.Fatalf("status = %d: only the origin the operator named", status)
		}
	})

	t.Run("a listener that is not on loopback", func(t *testing.T) {
		srv := server(t, &stub{}, Options{Host: "0.0.0.0"})
		header := map[string]string{"Host": "proxy.example:11234"}
		if status, out := browse(t, srv, "/v1/chat/completions", chat, header); status != http.StatusOK {
			t.Fatalf("status = %d: %v", status, out)
		}
	})

	t.Run("the listener named with its port", func(t *testing.T) {
		for _, host := range []string{"127.0.0.1:11234", "[::1]", "[::1]:11234", "localhost:11234", "LOCALHOST"} {
			p := &stub{}
			srv := server(t, p, Options{Host: host})
			header := map[string]string{"Host": "rebind.evil.example:11234"}
			if status, _ := browse(t, srv, "/v1/chat/completions", chat, header); status != http.StatusForbidden || p.seen != nil {
				t.Fatalf("Host %q: status = %d: the check is on whatever way the address is written", host, status)
			}
		}
	})

	// A front proxy hands on the public name it serves under, which goose
	// cannot know; and where there is a key, a page cannot present it, so the
	// name guards nothing the key does not.
	for _, c := range []struct {
		name   string
		opts   Options
		header map[string]string
	}{
		{"behind a front proxy the operator named", Options{Host: "127.0.0.1", ForwardedHeader: "X-Forwarded-For"},
			map[string]string{"X-Forwarded-For": "203.0.113.7"}},
		{"with a key of its own", Options{Host: "127.0.0.1", APIKey: "secret"},
			map[string]string{"Authorization": "Bearer secret"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			srv := server(t, &stub{}, c.opts)
			c.header["Host"] = "goose.example.com"
			if status, out := browse(t, srv, "/v1/chat/completions", chat, c.header); status != http.StatusOK {
				t.Fatalf("status = %d: %v", status, out)
			}
			req, _ := http.NewRequest(http.MethodGet, srv.URL+"/health", nil)
			req.Host = "goose.example.com"
			resp, err := srv.Client().Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("GET /health = %d", resp.StatusCode)
			}
		})
	}
}

// An extension's or a desktop app's origin is not something its user can see,
// so the refusal names it and the setting that admits it, and the operator is
// told once, whether or not every request is logged.
func TestARefusalNamesWhatItRefusedAndWhatAdmitsIt(t *testing.T) {
	chat := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	var buf bytes.Buffer
	srv := server(t, &stub{}, Options{Host: "127.0.0.1", Logger: slog.New(slog.NewTextHandler(&buf, nil))})
	for _, c := range []struct {
		header map[string]string
		names  []string
		logged string
	}{
		{map[string]string{"Origin": "app://obsidian.md"},
			[]string{`"app://obsidian.md"`, "cors_origin"}, "origin=app://obsidian.md"},
		{map[string]string{"Host": "host.docker.internal:11234"},
			[]string{`"host.docker.internal:11234"`, "server.api_key"}, "host=host.docker.internal:11234"},
		// The wildcard is refused without a key, so advice that left the key out
		// would stop the proxy starting.
		{map[string]string{"Sec-Fetch-Site": "cross-site"},
			[]string{`"cross-site"`, `cors_origin = "*"`, "server.api_key"}, "sec_fetch_site=cross-site"},
		// A sandboxed frame on any site sends this one, so it is the wildcard's
		// equal and admitted only as the wildcard is.
		{map[string]string{"Origin": "null"},
			[]string{`the origin "null"`, `cors_origin = "null"`, "server.api_key"}, "origin=null"},
	} {
		for range 3 {
			status, out := browse(t, srv, "/v1/chat/completions", chat, c.header)
			inner, _ := out["error"].(map[string]any)
			message, _ := inner["message"].(string)
			if status != http.StatusForbidden {
				t.Fatalf("status = %d, want 403", status)
			}
			for _, name := range c.names {
				if !strings.Contains(message, name) {
					t.Fatalf("message = %q: it names neither what was refused nor what admits it", message)
				}
			}
		}
		if n := strings.Count(buf.String(), c.logged); n != 1 {
			t.Fatalf("%s logged %d times, want once:\n%s", c.logged, n, buf.String())
		}
	}
}

func TestARefusalLogsOnlySoMuch(t *testing.T) {
	chat := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
	var buf bytes.Buffer
	srv := server(t, &stub{}, Options{Host: "127.0.0.1", Logger: slog.New(slog.NewTextHandler(&buf, nil))})
	for i := range turnedAwayLogged + 8 {
		browse(t, srv, "/v1/chat/completions", chat, map[string]string{"Host": fmt.Sprintf("n%d.rebind.evil.example", i)})
	}
	if n := strings.Count(buf.String(), "refused a request"); n != turnedAwayLogged {
		t.Fatalf("%d refusals logged: a page can send as many names as it likes", n)
	}
	// Anyone can fill it, so the operator is told once that it is full, and
	// that the answer is now the place to look.
	if n := strings.Count(buf.String(), "no further refused"); n != 1 {
		t.Fatalf("the log says %d times that it stopped, want once:\n%s", n, buf.String())
	}

	long := "https://" + strings.Repeat("a", 8<<10) + ".example"
	_, out := browse(t, srv, "/v1/chat/completions", chat, map[string]string{"Origin": long})
	inner, _ := out["error"].(map[string]any)
	if message, _ := inner["message"].(string); len(message) > 1024 {
		t.Fatalf("message is %d bytes: the caller's own header, repeated back whole", len(message))
	}
}

func TestTheBrowserSDKsHeadersAreAllowed(t *testing.T) {
	srv := server(t, &stub{}, Options{CORS: true, CORSOrigin: "https://app.example"})
	asked := []string{
		"authorization", "content-type", "x-api-key", "x-goog-api-key", "x-goog-api-client",
		"anthropic-version", "anthropic-beta", "anthropic-dangerous-direct-browser-access",
		"x-stainless-os", "x-stainless-lang", "x-stainless-retry-count", "x-stainless-timeout",
		"x-goose-provider",
	}
	req, _ := http.NewRequest(http.MethodOptions, srv.URL+"/v1/messages", nil)
	req.Header.Set("Origin", "https://app.example")
	req.Header.Set("Access-Control-Request-Method", "POST")
	req.Header.Set("Access-Control-Request-Headers", strings.Join(append(asked, "x-evil"), ","))
	resp, err := srv.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	allowed := map[string]bool{}
	for _, name := range strings.Split(resp.Header.Get("Access-Control-Allow-Headers"), ",") {
		allowed[strings.ToLower(strings.TrimSpace(name))] = true
	}
	for _, name := range asked {
		if !allowed[name] {
			t.Errorf("%s is not allowed: the browser never sends the request", name)
		}
	}
	if allowed["x-evil"] {
		t.Fatal("an allow-list that admits anything is not one")
	}
}
