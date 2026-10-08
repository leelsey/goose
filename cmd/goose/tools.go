// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"context"
	"os"
	"time"

	"github.com/leelsey/goose/internal/spawn"
	"github.com/leelsey/goose/mcp"
)

var defaultToolEnv = []string{
	"HOME", "PATH", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TMPDIR",
	"XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
	"APPDATA", "LOCALAPPDATA", "USERPROFILE", "SystemRoot", "TEMP", "TMP",
	"http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
	"NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR",
}

func connectTools(ctx context.Context, cfg *Config) (*mcp.Manager, error) {
	if len(cfg.MCP) == 0 {
		return nil, nil
	}
	m := newToolManager()
	for _, entry := range cfg.MCP {
		allow := append(append([]string(nil), defaultToolEnv...), entry.EnvAllow...)
		server := mcp.ServerConfig{
			Name:      entry.Name,
			Transport: mcp.TransportKind(entry.Transport),
			Command:   entry.Command,
			Args:      entry.Args,
			Dir:       entry.Dir,
			Env:       spawn.ScrubEnv(os.Environ(), allow, nil),
			URL:       entry.URL,
			Headers:   entry.Headers,
			Timeout:   toolTimeout(entry.Timeout),
		}
		// Add names the server in its error already.
		if err := m.Add(ctx, server, nil); err != nil {
			_ = m.Close()
			return nil, err
		}
	}
	return m, nil
}

func newToolManager() *mcp.Manager {
	mcp.SetVersion(version)
	return mcp.NewManager()
}

func toolTimeout(s string) time.Duration {
	if s == "" {
		return 30 * time.Second
	}
	d, err := time.ParseDuration(s)
	if err != nil || d <= 0 {
		return 30 * time.Second
	}
	return d
}
