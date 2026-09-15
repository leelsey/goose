// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"os"
	"path/filepath"
	"testing"
)

func FuzzLoadConfig(f *testing.F) {
	for _, seed := range []string{
		"formats = [\"openai\"]\n[server]\nport = 1\n[[provider]]\nid = \"p\"\nkind = \"ollama\"\n",
		"[server]\nport = 70000\n",
		"[[provider]]\n",
		"[[route]]\nchain = [\"a:b\"]\n",
		"[[mcp]]\nname = \"x\"\n",
		"= broken",
		"",
	} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, body string) {
		dir := t.TempDir()
		path := filepath.Join(dir, "goose.toml")
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		cfg, _, err := load(path)
		if err != nil {
			return
		}
		if cfg == nil {
			t.Fatal("load returned no configuration and no error")
		}
		cfg.resolveSecrets(func(string) string { return "" })
		_ = cfg.validate()
	})
}
