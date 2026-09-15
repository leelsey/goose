// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

// Package httpcore is the HTTP plumbing every remote adapter shares.
package httpcore

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/leelsey/goose/llm"
)

const errBodyCap = 16 << 10

const IdleConnsPerHost = 64

func NewClient(headerTimeout time.Duration) *http.Client {
	tr, ok := http.DefaultTransport.(*http.Transport)
	if ok {
		tr = tr.Clone()
	} else {
		tr = &http.Transport{}
	}
	if headerTimeout <= 0 {
		headerTimeout = 5 * time.Minute
	}
	tr.ResponseHeaderTimeout = headerTimeout
	tr.MaxIdleConnsPerHost = IdleConnsPerHost
	if tr.MaxIdleConns < IdleConnsPerHost {
		tr.MaxIdleConns = IdleConnsPerHost * 4
	}
	return &http.Client{Transport: tr}
}

type Attempted func(err *llm.Error, elapsed time.Duration)

type Caller struct {
	Provider   string
	HTTP       *http.Client
	MaxRetries int
	Backoff    time.Duration
	OnDiscard  Attempted
}

type Request struct {
	Method    string
	URL       string
	Body      []byte
	Header    http.Header
	OnDiscard Attempted
	Sign      func(ctx context.Context, r *http.Request, body []byte) error
	Limits    llm.Limits
}

func (c *Caller) Do(ctx context.Context, req Request) (*http.Response, error) {
	retries := c.MaxRetries
	if retries < 0 {
		retries = 0
	}
	backoff := c.Backoff
	if backoff <= 0 {
		backoff = time.Second
	}

	if max := req.Limits.MaxRequestBytes; max > 0 && int64(len(req.Body)) > max {
		return nil, &llm.Error{
			Kind: llm.KindInvalidRequest, Code: llm.PolicyBodyTooLarge, Provider: c.Provider,
			Message: "the request body is larger than this route allows",
			Sent:    false, Retryable: true,
		}
	}

	var last *llm.Error
	for attempt := 0; ; attempt++ {
		started := time.Now()
		resp, err := c.once(ctx, req)
		switch {
		case err == nil:
			return resp, nil
		case !err.Retryable || attempt >= retries:
			return nil, err
		}
		last = err
		elapsed := time.Since(started)
		select {
		case <-ctx.Done():
			return nil, cancelled(ctx.Err(), c.Provider)
		case <-time.After(wait(backoff, attempt, err.RetryAfter)):
		}
		if last.Sent {
			if req.OnDiscard != nil {
				req.OnDiscard(last, elapsed)
			} else if c.OnDiscard != nil {
				c.OnDiscard(last, elapsed)
			}
		}
	}
}

func (c *Caller) once(ctx context.Context, req Request) (*http.Response, *llm.Error) {
	method := req.Method
	if method == "" {
		method = http.MethodPost
	}
	r, err := http.NewRequestWithContext(ctx, method, req.URL, bytes.NewReader(req.Body))
	if err != nil {
		return nil, &llm.Error{Kind: llm.KindInvalidRequest, Provider: c.Provider, Message: err.Error(), Err: err}
	}
	if req.Body != nil && r.Header.Get("Content-Type") == "" {
		r.Header.Set("Content-Type", "application/json")
	}
	for k, vs := range req.Header {
		for _, v := range vs {
			r.Header.Add(k, v)
		}
	}
	if req.Sign != nil {
		if err := req.Sign(ctx, r, req.Body); err != nil {
			return nil, &llm.Error{Kind: llm.KindAuth, Provider: c.Provider, Message: err.Error(), Err: err}
		}
	}

	client := c.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(r)
	if err != nil {
		if ctx.Err() != nil {
			return nil, cancelled(ctx.Err(), c.Provider)
		}
		return nil, &llm.Error{
			Kind: llm.KindTransport, Provider: c.Provider, Message: err.Error(),
			Sent: true, Retryable: true, Err: err,
		}
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		if max := req.Limits.MaxResponseBytes; max > 0 {
			resp.Body = &capped{rc: resp.Body, left: max, provider: c.Provider}
		}
		return resp, nil
	}
	e := StatusError(c.Provider, resp)
	resp.Body.Close()
	return nil, e
}

func cancelled(err error, provider string) *llm.Error {
	kind := llm.KindCancelled
	if errors.Is(err, context.DeadlineExceeded) {
		kind = llm.KindTimeout
	}
	return &llm.Error{Kind: kind, Provider: provider, Message: err.Error(), Sent: true, Err: err}
}

func StatusError(provider string, resp *http.Response) *llm.Error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, errBodyCap))
	message, code := Upstream(body)
	e := &llm.Error{
		Kind:      llm.KindForStatus(resp.StatusCode),
		Provider:  provider,
		Status:    resp.StatusCode,
		Code:      code,
		Message:   message,
		Body:      string(body),
		Sent:      true,
		Retryable: llm.RetryableStatus(resp.StatusCode),
	}
	if e.Message == "" {
		e.Message = http.StatusText(resp.StatusCode)
	}
	if ra := resp.Header.Get("Retry-After"); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil {
			e.RetryAfter = time.Duration(secs) * time.Second
		} else if when, err := http.ParseTime(ra); err == nil {
			e.RetryAfter = time.Until(when)
		}
	}
	return e
}

const maxBackoff = 30 * time.Second

func wait(base time.Duration, attempt int, retryAfter time.Duration) time.Duration {
	if retryAfter > 0 {
		if retryAfter > maxBackoff {
			retryAfter = maxBackoff
		}
		return retryAfter + jitter(retryAfter/10)
	}
	shift := attempt
	if shift > 5 {
		shift = 5
	}
	d := base << uint(shift)
	if d > maxBackoff {
		d = maxBackoff
	}
	return jitter(d)
}

func jitter(d time.Duration) time.Duration {
	if d <= 0 {
		return 0
	}
	return time.Duration(rand.Int64N(int64(d) + 1))
}

func JoinURL(base, path string) string {
	if base == "" {
		return path
	}
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(path, "/")
}

type capped struct {
	rc       io.ReadCloser
	r        io.Reader
	left     int64
	provider string
}

func (c *capped) reader() io.Reader {
	if c.rc != nil {
		return c.rc
	}
	return c.r
}

func (c *capped) Read(p []byte) (int, error) {
	if c.left < 0 {
		return 0, c.err()
	}
	if int64(len(p)) > c.left+1 {
		p = p[:c.left+1]
	}
	n, err := c.reader().Read(p)
	c.left -= int64(n)
	if c.left < 0 {
		return n, c.err()
	}
	return n, err
}

func (c *capped) Close() error {
	if c.rc == nil {
		return nil
	}
	return c.rc.Close()
}

func Capped(r io.Reader, max int64, provider string) io.Reader {
	if max <= 0 {
		return r
	}
	return &capped{r: r, left: max, provider: provider}
}

func (c *capped) err() error {
	return &llm.Error{
		Kind: llm.KindUpstream, Code: llm.PolicyBodyTooLarge, Provider: c.provider,
		Message: "the response body is larger than this route allows",
		Sent:    true,
	}
}

func HopByHop(header http.Header, key string) bool {
	switch http.CanonicalHeaderKey(key) {
	case "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
		"Te", "Trailer", "Transfer-Encoding", "Upgrade":
		return true
	}
	for _, nominated := range header.Values("Connection") {
		for _, name := range strings.Split(nominated, ",") {
			if strings.EqualFold(strings.TrimSpace(name), key) {
				return true
			}
		}
	}
	return false
}

var credentials = []string{
	"Authorization", "X-Api-Key", "X-Goog-Api-Key", "Api-Key",
	"X-Goog-User-Project", "Cookie",
}

func ForwardPassthrough(dst, src http.Header) {
	ForwardHeader(dst, src, append([]string{"Host", "Content-Length"}, credentials...)...)
}

func ForwardHeader(dst, src http.Header, drop ...string) {
	skip := make(map[string]bool, len(drop))
	for _, name := range drop {
		skip[http.CanonicalHeaderKey(name)] = true
	}
	for k, vs := range src {
		if HopByHop(src, k) || skip[http.CanonicalHeaderKey(k)] {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}
