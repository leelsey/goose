// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package cloudauth signs requests for the cloud runtimes.
package cloudauth

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	algorithm  = "AWS4-HMAC-SHA256"
	terminator = "aws4_request"
	amzDateFmt = "20060102T150405Z"
	shortDate  = "20060102"
)

type Creds struct {
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

func (c Creds) valid() bool { return c.AccessKeyID != "" && c.SecretAccessKey != "" }

func SignV4(r *http.Request, body []byte, c Creds, region, service string, now time.Time) error {
	if !c.valid() {
		return errMissingCredentials
	}
	now = now.UTC()
	amzDate := now.Format(amzDateFmt)
	date := now.Format(shortDate)

	payloadHash := hex.EncodeToString(sha256sum(body))
	r.Header.Set("X-Amz-Date", amzDate)
	if c.SessionToken != "" {
		r.Header.Set("X-Amz-Security-Token", c.SessionToken)
	}
	if r.Header.Get("Host") == "" && r.Host == "" {
		r.Host = r.URL.Host
	}

	signed, canonicalHeaders := canonicalHeaders(r)
	canonical := strings.Join([]string{
		r.Method,
		canonicalURI(r),
		canonicalQuery(r),
		canonicalHeaders,
		signed,
		payloadHash,
	}, "\n")

	scope := strings.Join([]string{date, region, service, terminator}, "/")
	stringToSign := strings.Join([]string{
		algorithm,
		amzDate,
		scope,
		hex.EncodeToString(sha256sum([]byte(canonical))),
	}, "\n")

	key := signingKey(c.SecretAccessKey, date, region, service)
	signature := hex.EncodeToString(hmacSHA256(key, []byte(stringToSign)))

	r.Header.Set("Authorization", algorithm+
		" Credential="+c.AccessKeyID+"/"+scope+
		", SignedHeaders="+signed+
		", Signature="+signature)
	return nil
}

func canonicalURI(r *http.Request) string {
	path := r.URL.EscapedPath()
	if path == "" {
		return "/"
	}
	return escapePath(path)
}

func canonicalQuery(r *http.Request) string {
	q := r.URL.Query()
	keys := make([]string, 0, len(q))
	for k := range q {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var parts []string
	for _, k := range keys {
		values := append([]string(nil), q[k]...)
		sort.Strings(values)
		for _, v := range values {
			parts = append(parts, escape(k)+"="+escape(v))
		}
	}
	return strings.Join(parts, "&")
}

func canonicalHeaders(r *http.Request) (signed, canonical string) {
	host := r.Host
	if host == "" {
		host = r.URL.Host
	}
	values := map[string]string{"host": host}
	for k, vs := range r.Header {
		lower := strings.ToLower(k)
		switch lower {
		case "authorization", "user-agent", "content-length", "connection", "expect":
			continue
		}
		trimmed := make([]string, 0, len(vs))
		for _, v := range vs {
			trimmed = append(trimmed, strings.Join(strings.Fields(v), " "))
		}
		values[lower] = strings.Join(trimmed, ",")
	}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for _, k := range keys {
		b.WriteString(k)
		b.WriteByte(':')
		b.WriteString(values[k])
		b.WriteByte('\n')
	}
	return strings.Join(keys, ";"), b.String()
}

func signingKey(secret, date, region, service string) []byte {
	k := hmacSHA256([]byte("AWS4"+secret), []byte(date))
	k = hmacSHA256(k, []byte(region))
	k = hmacSHA256(k, []byte(service))
	return hmacSHA256(k, []byte(terminator))
}

func hmacSHA256(key, data []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(data)
	return h.Sum(nil)
}

func sha256sum(b []byte) []byte {
	sum := sha256.Sum256(b)
	return sum[:]
}

func escapePath(path string) string {
	var b strings.Builder
	for i := 0; i < len(path); i++ {
		ch := path[i]
		if ch == '/' {
			b.WriteByte(ch)
			continue
		}
		b.WriteString(escapeByte(ch))
	}
	return b.String()
}

func escape(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		b.WriteString(escapeByte(s[i]))
	}
	return b.String()
}

func escapeByte(ch byte) string {
	if unreserved(ch) {
		return string(ch)
	}
	const hexDigits = "0123456789ABCDEF"
	return string([]byte{'%', hexDigits[ch>>4], hexDigits[ch&0x0f]})
}

func unreserved(ch byte) bool {
	switch {
	case ch >= 'A' && ch <= 'Z', ch >= 'a' && ch <= 'z', ch >= '0' && ch <= '9':
		return true
	case ch == '-', ch == '_', ch == '.', ch == '~':
		return true
	default:
		return false
	}
}

type sigError string

func (e sigError) Error() string { return string(e) }

const errMissingCredentials = sigError("aws: no credentials")
