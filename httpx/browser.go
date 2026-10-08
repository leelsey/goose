// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"fmt"
	"net"
	"net/http"
	"slices"
	"strings"

	"github.com/leelsey/goose/llm"
)

// browser refuses what a web page can have a browser send here without the
// operator's leave: a request from an origin the CORS setting does not name,
// and, on a loopback listener no key guards, one addressed to a name that is
// not the machine's own. The model endpoints spend the operator's credentials
// as /mcp runs their tools, and a proxy on loopback often holds no key of its
// own, so any page the operator visits could otherwise drive it. /mcp keeps its
// own check, which refuses every origin in its own wire.
func (s *Server) browser(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch pattern := s.served(r); {
		case pattern == "/mcp" || pattern == "/mcp/":
		case s.foreignHost(r.Host):
			s.turnAway(w, r, "host", clip(r.Host), "server.api_key",
				"this proxy does not answer to the name %q; call it as localhost, or set %s to admit it")
			return
		case s.foreignOrigin(r):
			if origin := clip(r.Header.Get("Origin")); origin != "" {
				s.turnAway(w, r, "origin", origin, s.admitting(origin),
					"this proxy does not serve pages from the origin %q; set %s to admit it")
				return
			}
			s.turnAway(w, r, "sec_fetch_site", clip(r.Header.Get("Sec-Fetch-Site")), s.admitting("*"),
				"this proxy does not serve a %q request that names no origin; set %s to admit it")
			return
		}
		next.ServeHTTP(w, r)
	})
}

const (
	// Longer than any real origin or host name; no more of what a caller chose
	// is repeated to it or written to the log.
	turnedAwayLength = 256
	// A page can have a browser send as many names and origins as it likes, and
	// the log is the operator's, so only so many are written to it.
	turnedAwayLogged = 64
)

func clip(value string) string {
	if len(value) > turnedAwayLength {
		return value[:turnedAwayLength]
	}
	return value
}

// admitting is everything that has to be set for pages from origin to be
// served. The wildcard and "null", which a sandboxed frame on any site sends,
// let any site drive the proxy, so New takes either only with a key.
func (s *Server) admitting(origin string) string {
	if (origin == "*" || origin == "null") && s.opts.APIKey == "" {
		return fmt.Sprintf("cors = true, cors_origin = %q and server.api_key", origin)
	}
	return fmt.Sprintf("cors = true and cors_origin = %q", origin)
}

// turnAway names what was refused and the setting that admits it, to the
// caller and, once for each value, to the operator, who is seldom the one
// reading the answer and has no other way to learn an app's origin.
func (s *Server) turnAway(w http.ResponseWriter, r *http.Request, name, value, admit, format string) {
	if first, filled := s.firstTurnedAway(name + " " + value); first {
		s.log().Warn("refused a request", name, value, "admit", admit)
		if filled {
			s.log().Warn("no further refused names or origins are logged; each refusal still names its own to the caller",
				"logged", turnedAwayLogged)
		}
	}
	s.refuse(w, r, llm.Errorf(llm.KindPermission, format, value, admit))
}

// firstTurnedAway reports whether key is new to the log, and whether it is the
// last the log takes, so that the operator is told the log has stopped.
func (s *Server) firstTurnedAway(key string) (first, filled bool) {
	s.turnedMu.Lock()
	defer s.turnedMu.Unlock()
	if s.turned[key] || len(s.turned) >= turnedAwayLogged {
		return false, false
	}
	s.turned[key] = true
	return true, len(s.turned) == turnedAwayLogged
}

func (s *Server) foreignHost(host string) bool {
	if host == "" || !s.rebindable() {
		return false
	}
	return !ownName(bare(host))
}

// rebindable is a listener a page can take for its own origin by pointing a
// name of its own at it: one on loopback with nothing else guarding it. A key
// is something no such page can present, and a forwarded header says a front
// proxy is in the way, handing on whatever public name it serves under.
func (s *Server) rebindable() bool {
	return s.opts.APIKey == "" && s.opts.ForwardedHeader == "" &&
		Loopback(strings.ToLower(bare(s.opts.Host)))
}

func bare(host string) string {
	if h, _, err := net.SplitHostPort(host); err == nil {
		return h
	}
	return strings.Trim(host, "[]")
}

// ownName is a name that only ever means this machine, so no page from
// elsewhere can be served under it, whatever that page's owner does with DNS.
func ownName(host string) bool {
	host = strings.ToLower(strings.TrimSuffix(host, "."))
	if host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && (ip.IsLoopback() || ip.IsUnspecified())
}

// A request a browser marks as coming from another site carries no Origin when
// it is a navigation or a no-cors fetch, so the mark is read as well. Ports do
// not separate sites, so a page on another local port is same-site.
func (s *Server) foreignOrigin(r *http.Request) bool {
	origin := r.Header.Get("Origin")
	if origin == "" {
		switch r.Header.Get("Sec-Fetch-Site") {
		case "cross-site", "same-site":
		default:
			return false
		}
	}
	return !s.opts.CORS || (s.opts.CORSOrigin != "*" && origin != s.opts.CORSOrigin)
}

// The headers the official browser SDKs send, and the one a client pins a
// session's provider with. The Stainless-generated SDKs add a family of their
// own that grows between releases, so that is admitted by its prefix.
var corsHeaders = []string{
	"Authorization", "Content-Type", "x-api-key", "x-goog-api-key", "x-goog-api-client",
	"anthropic-version", "anthropic-beta", "anthropic-dangerous-direct-browser-access",
	ProviderHeader,
}

const stainless = "x-stainless-"

func allowedHeaders(requested string) string {
	allowed := slices.Clone(corsHeaders)
	for _, name := range strings.Split(requested, ",") {
		name = strings.ToLower(strings.TrimSpace(name))
		if len(name) > len(stainless) && strings.HasPrefix(name, stainless) {
			allowed = append(allowed, name)
		}
	}
	return strings.Join(allowed, ", ")
}
