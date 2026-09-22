// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package mcp

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

const headerAnnotation = "x-mcp-header"

func paramHeaders(schema, args json.RawMessage) (map[string]string, error) {
	named, err := annotated(schema)
	if err != nil || len(named) == 0 {
		return nil, err
	}
	var values map[string]json.RawMessage
	if len(args) > 0 {
		if err := json.Unmarshal(args, &values); err != nil {
			// The mirror check reads these. Answering "nothing to check"
			// for arguments nobody could read would skip it instead.
			return nil, fmt.Errorf("mcp: the arguments are not an object: %w", err)
		}
	}
	out := map[string]string{}
	for _, p := range named {
		raw, ok := walk(values, p.path)
		if !ok {
			continue
		}
		text, ok := headerScalar(raw)
		if !ok {
			continue
		}
		out["Mcp-Param-"+p.name] = headerValue(text)
	}
	return out, nil
}

type annotation struct {
	name string
	path []string
}

func annotated(schema json.RawMessage) ([]annotation, error) {
	if len(schema) == 0 {
		return nil, nil
	}
	var root map[string]json.RawMessage
	if json.Unmarshal(schema, &root) != nil {
		return nil, nil
	}
	var out []annotation
	seen := map[string]bool{}
	var walkProperties func(node map[string]json.RawMessage, path []string, depth int) error
	walkProperties = func(node map[string]json.RawMessage, path []string, depth int) error {
		if depth > 16 {
			return nil
		}
		raw, ok := node["properties"]
		if !ok {
			return nil
		}
		var properties map[string]json.RawMessage
		if json.Unmarshal(raw, &properties) != nil {
			return nil
		}
		for name, rawChild := range properties {
			var child map[string]json.RawMessage
			if json.Unmarshal(rawChild, &child) != nil {
				continue
			}
			here := append(append([]string(nil), path...), name)
			if marker, ok := child[headerAnnotation]; ok {
				var header string
				if json.Unmarshal(marker, &header) != nil {
					return fmt.Errorf("%s on %q is not a name", headerAnnotation, name)
				}
				if err := validHeaderName(header); err != nil {
					return fmt.Errorf("%s on %q: %w", headerAnnotation, name, err)
				}
				if seen[strings.ToLower(header)] {
					return fmt.Errorf("%s %q is used twice", headerAnnotation, header)
				}
				if err := headerable(child); err != nil {
					return fmt.Errorf("%s on %q: %w", headerAnnotation, name, err)
				}
				seen[strings.ToLower(header)] = true
				out = append(out, annotation{name: header, path: here})
			}
			if err := walkProperties(child, here, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := walkProperties(root, nil, 0); err != nil {
		return nil, err
	}
	return out, nil
}

func headerable(node map[string]json.RawMessage) error {
	var kind string
	if raw, ok := node["type"]; ok {
		_ = json.Unmarshal(raw, &kind)
	}
	switch kind {
	case "string", "integer", "boolean":
		return nil
	case "":
		return fmt.Errorf("a property with no type cannot be mirrored")
	default:
		return fmt.Errorf("a %s cannot be mirrored", kind)
	}
}

func validHeaderName(name string) error {
	if name == "" {
		return fmt.Errorf("the name is empty")
	}
	const token = "!#$%&'*+-.^_`|~"
	for i := 0; i < len(name); i++ {
		c := name[i]
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case strings.IndexByte(token, c) >= 0:
		default:
			return fmt.Errorf("%q is not a header name", name)
		}
	}
	return nil
}

func walk(values map[string]json.RawMessage, path []string) (json.RawMessage, bool) {
	for i, step := range path {
		raw, ok := values[step]
		if !ok {
			return nil, false
		}
		if i == len(path)-1 {
			return raw, true
		}
		var next map[string]json.RawMessage
		if json.Unmarshal(raw, &next) != nil {
			return nil, false
		}
		values = next
	}
	return nil, false
}

func headerScalar(raw json.RawMessage) (string, bool) {
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return "", false
	}
	switch v := value.(type) {
	case string:
		return v, true
	case bool:
		return strconv.FormatBool(v), true
	case float64:
		if v == float64(int64(v)) {
			return strconv.FormatInt(int64(v), 10), true
		}
		return "", false
	default:
		return "", false
	}
}
