// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package main

import (
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

const examplePath = "../../goose.example.toml"

// An unknown key is a fatal error, so the example is the only place an operator
// can find out that a key exists at all. Nothing else checks it, which is how
// it drifted from the code in the first place.
func TestTheExampleNamesEveryKeyTheCodeReads(t *testing.T) {
	raw, err := os.ReadFile(examplePath)
	if err != nil {
		t.Fatal(err)
	}
	written := map[string]bool{}
	key := regexp.MustCompile(`(?m)^[\t ]*#?[\t ]*([a-z_0-9]+)[\t ]*=`)
	for _, m := range key.FindAllStringSubmatch(string(raw), -1) {
		written[m[1]] = true
	}
	inline := regexp.MustCompile(`([a-z_0-9]+)[\t ]*=`)
	for _, table := range regexp.MustCompile(`\{[^}]*\}`).FindAllString(string(raw), -1) {
		for _, m := range inline.FindAllStringSubmatch(table, -1) {
			written[m[1]] = true
		}
	}
	for _, m := range regexp.MustCompile(`(?m)^#? ?\[+([a-z_.]+)\]+`).FindAllStringSubmatch(string(raw), -1) {
		for _, part := range strings.Split(m[1], ".") {
			written[part] = true
		}
	}

	var walk func(reflect.Type)
	seen := map[reflect.Type]bool{}
	var missing []string
	walk = func(rt reflect.Type) {
		for rt.Kind() == reflect.Pointer || rt.Kind() == reflect.Slice {
			rt = rt.Elem()
		}
		if rt.Kind() != reflect.Struct || seen[rt] {
			return
		}
		seen[rt] = true
		for i := 0; i < rt.NumField(); i++ {
			f := rt.Field(i)
			name, _, _ := strings.Cut(f.Tag.Get("toml"), ",")
			if name == "" || name == "-" {
				continue
			}
			if !written[name] {
				missing = append(missing, rt.Name()+"."+name)
			}
			walk(f.Type)
		}
	}
	walk(reflect.TypeOf(Config{}))
	if len(missing) > 0 {
		t.Fatalf("goose.example.toml does not mention %s; an unknown key is fatal, so a key nobody wrote down is a key nobody can use",
			strings.Join(missing, ", "))
	}
}

func TestTheExampleLoads(t *testing.T) {
	raw, err := os.ReadFile(examplePath)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "goose.toml")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := load(path)
	if err != nil {
		t.Fatalf("the example an operator is told to copy does not load: %v", err)
	}
	if err := cfg.validate(true); err != nil {
		t.Fatalf("the example an operator is told to copy does not validate: %v", err)
	}
}

// Every commented setting is one an operator may uncomment, so with each of
// them uncommented the example must still pass a check. Left out are the
// listener's certificate, a file of the operator's own, a placeholder that
// stands for a list, and the second of two lines showing one key's values.
func TestTheExampleUncommentedPassesACheck(t *testing.T) {
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
	raw, err := os.ReadFile(examplePath)
	if err != nil {
		t.Fatal(err)
	}
	table := regexp.MustCompile(`^(# ?)?\[\[?[a-z_.]+\]\]?\s*$`)
	setting := regexp.MustCompile(`^(# ?)?([a-z_0-9]+)\s*=`)
	var out []string
	keys := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		if m := table.FindStringSubmatch(line); m != nil {
			keys = map[string]bool{}
			line = strings.TrimPrefix(line, m[1])
		} else if m := setting.FindStringSubmatch(line); m != nil {
			key := m[2]
			if keys[key] || key == "tls_cert" || key == "tls_key" || strings.Contains(line, "[...]") {
				continue
			}
			keys[key] = true
			line = strings.TrimPrefix(line, m[1])
		}
		out = append(out, line)
	}
	path := filepath.Join(t.TempDir(), "goose.toml")
	if err := os.WriteFile(path, []byte(strings.Join(out, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"-c", path, "-check"}, {"mcp", "-c", path, "-check"}} {
		if _, _, err := runWith(t, args...); err != nil {
			t.Fatalf("%v: %v\n%s", args, err, strings.Join(out, "\n"))
		}
	}
}
