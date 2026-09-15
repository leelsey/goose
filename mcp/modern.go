// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
)

const modernVersion = "2026-07-28"

const (
	metaVersion    = "io.modelcontextprotocol/protocolVersion"
	metaClientInfo = "io.modelcontextprotocol/clientInfo"
	metaClientCaps = "io.modelcontextprotocol/clientCapabilities"
	metaServerInfo = "io.modelcontextprotocol/serverInfo"
)

const (
	codeHeaderMismatch     = -32020
	codeUnsupportedVersion = -32022
)

var modernSpoken = []string{modernVersion}

type meta struct {
	Version string          `json:"io.modelcontextprotocol/protocolVersion"`
	Client  json.RawMessage `json:"io.modelcontextprotocol/clientInfo"`
	Caps    json.RawMessage `json:"io.modelcontextprotocol/clientCapabilities"`
}

func readMeta(params json.RawMessage) (meta, bool) {
	if len(params) == 0 {
		return meta{}, false
	}
	var envelope struct {
		Meta *meta `json:"_meta"`
	}
	if json.Unmarshal(params, &envelope) != nil || envelope.Meta == nil {
		return meta{}, false
	}
	if envelope.Meta.Version == "" {
		return meta{}, false
	}
	return *envelope.Meta, true
}

func requestMeta(version string) map[string]any {
	return map[string]any{
		metaVersion:    version,
		metaClientInfo: implementation(),
		metaClientCaps: map[string]any{},
	}
}

func answer(result map[string]any) map[string]any {
	result["resultType"] = "complete"
	result["_meta"] = map[string]any{metaServerInfo: implementation()}
	return result
}

func unsupportedVersion(requested string) *rpcError {
	data, _ := json.Marshal(map[string]any{
		"supported": append(append([]string(nil), modernSpoken...), spoken...),
		"requested": requested,
	})
	return &rpcError{
		Code: codeUnsupportedVersion, Message: "unsupported protocol version " + requested,
		Data: data, status: 400,
	}
}

func speaksModern(version string) bool {
	for _, v := range modernSpoken {
		if v == version {
			return true
		}
	}
	return false
}

type mirrored struct {
	Present bool
	Version string
	Method  string
	Name    string
	Params  map[string]string
}

func (m mirrored) check(version, method, name string) *rpcError {
	if !m.Present {
		return nil
	}
	switch {
	case m.Version == "":
		return headerMismatch("MCP-Protocol-Version is required and was not sent")
	case m.Version != version:
		return headerMismatch("MCP-Protocol-Version does not match the version in the request")
	case m.Method == "":
		return headerMismatch("Mcp-Method is required and was not sent")
	case m.Method != method:
		return headerMismatch("Mcp-Method does not match the method in the request")
	}
	if name == "" {
		return nil
	}
	switch {
	case m.Name == "":
		return headerMismatch("Mcp-Name is required for this method and was not sent")
	case headerText(m.Name) != name:
		return headerMismatch("Mcp-Name does not match the name in the request")
	}
	return nil
}

func (m mirrored) checkParams(expected map[string]string) *rpcError {
	if !m.Present {
		return nil
	}
	for name, want := range expected {
		got, ok := m.Params[http.CanonicalHeaderKey(name)]
		switch {
		case !ok:
			return headerMismatch(name + " is required for this call and was not sent")
		case headerText(got) != headerText(want):
			return headerMismatch(name + " does not match the argument in the request")
		}
	}
	return nil
}

func headerMismatch(message string) *rpcError {
	return &rpcError{Code: codeHeaderMismatch, Message: message, status: 400}
}

func headerValue(s string) string {
	if plainASCII(s) && !strings.HasPrefix(s, "=?base64?") {
		return s
	}
	return "=?base64?" + base64.StdEncoding.EncodeToString([]byte(s)) + "?="
}

func headerText(v string) string {
	if !strings.HasPrefix(v, "=?base64?") || !strings.HasSuffix(v, "?=") {
		return v
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSuffix(strings.TrimPrefix(v, "=?base64?"), "?="))
	if err != nil {
		return v
	}
	return string(raw)
}

func plainASCII(s string) bool {
	if s != strings.TrimSpace(s) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x21 || c > 0x7e {
			return false
		}
	}
	return true
}
