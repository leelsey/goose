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
	"net/url"
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
	// Refine reads a refusal's body for what its status does not say, for a
	// vendor that answers a bad key or an account it will not serve with the
	// status it gives a malformed request.
	Refine func(e *llm.Error)
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
		timer := time.NewTimer(wait(backoff, attempt, err.RetryAfter))
		select {
		case <-ctx.Done():
			// Not booked here: the caller's own error carries Sent, so this
			// attempt is on the row the chain writes for it. Booking it again
			// would put one upstream request on the ledger twice.
			timer.Stop()
			return nil, cancelled(why(ctx), c.Provider)
		case <-timer.C:
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
			// A credential still being fetched when the time ran out is the wall
			// or the caller leaving, as it is anywhere else, not a key the backend
			// refused. Nothing has left the process either way.
			if ctx.Err() != nil {
				e := cancelled(why(ctx), c.Provider)
				e.Sent = false
				return nil, e
			}
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
			return nil, cancelled(why(ctx), c.Provider)
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
	if c.Refine != nil {
		c.Refine(e)
	}
	return nil, e
}

// ReadError names what went wrong reading a streamed answer. A context error is
// the proxy's own wall or the caller leaving, and reporting either as the
// backend failing sends the wrong status and counts a fault against a backend
// that has none.
func ReadError(ctx context.Context, provider string, err error) *llm.Error {
	if e := contextual(ctx, err, provider); e != nil {
		return e
	}
	return &llm.Error{Kind: llm.KindTransport, Provider: provider, Sent: true, Message: err.Error(), Err: err}
}

// BodyError is ReadError for an answer read in one piece.
func BodyError(ctx context.Context, provider string, err error) *llm.Error {
	if e := contextual(ctx, err, provider); e != nil {
		return e
	}
	return &llm.Error{
		Kind: llm.KindUpstream, Provider: provider, Sent: true,
		Message: "response body: " + err.Error(), Err: err,
	}
}

func contextual(ctx context.Context, err error, provider string) *llm.Error {
	switch {
	case ctx.Err() != nil:
		return cancelled(why(ctx), provider)
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return cancelled(err, provider)
	}
	return nil
}

// why prefers the cause, so a deadline the proxy set for itself is not read as
// the caller having hung up.
func why(ctx context.Context) error {
	if cause := context.Cause(ctx); cause != nil {
		return cause
	}
	return ctx.Err()
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

// ErrorBody is what an error keeps of a body that reported a failure inside a
// 200, or of a stream's error frame: as much as is read of a refusal's body, so
// an upstream's words are held, and redacted, the same wherever they came from.
func ErrorBody(b []byte) string {
	if len(b) > errBodyCap {
		b = b[:errBodyCap]
	}
	return string(b)
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

// The last three choose whom a key bills rather than proving anything, and are
// no more a caller's to choose for the operator's key than the key is.
var credentials = []string{
	"Authorization", "X-Api-Key", "X-Goog-Api-Key", "Api-Key", "Cookie",
	"X-Goog-User-Project", "OpenAI-Organization", "OpenAI-Project",
}

func ForwardPassthrough(dst, src http.Header) {
	ForwardHeader(dst, src, append([]string{"Host", "Content-Length"}, credentials...)...)
}

// Overlay lays the headers configured for a backend over those a caller sent.
// One a gateway in front of the backend relies on is the operator's to set,
// and a caller naming the same header does not replace it.
func Overlay(dst, configured http.Header) {
	for k := range configured {
		dst.Del(k)
	}
	for k, vs := range configured {
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

// credentialParams are the query parameters a key can travel in. The proxy's
// gate takes one there, and a backend handed it keeps it in its logs along
// with the rest of the address.
var credentialParams = []string{"key", "api-key", "api_key", "access_token"}

// PassQuery is a caller's query with every credential taken out of it. A query
// that cannot be read is refused rather than sent on, since nobody can say what
// it carries.
func PassQuery(provider, raw string) (string, error) {
	if raw == "" {
		return "", nil
	}
	q, err := url.ParseQuery(raw)
	if err != nil {
		return "", &llm.Error{Kind: llm.KindInvalidRequest, Provider: provider,
			Message: "the query string could not be read: " + err.Error(), Err: err}
	}
	dropped := false
	for name := range q {
		for _, c := range credentialParams {
			if strings.EqualFold(name, c) {
				delete(q, name)
				dropped = true
			}
		}
	}
	if !dropped {
		return raw, nil
	}
	return q.Encode(), nil
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

// Reach makes a request only to learn whether the backend answers, and closes
// whatever came back.
func (c *Caller) Reach(ctx context.Context, req Request) error {
	resp, err := c.Do(ctx, req)
	if err != nil {
		return err
	}
	return resp.Body.Close()
}

// MalformedList is what a model list that cannot be read returns.
func MalformedList(provider string, err error) error {
	return &llm.Error{
		Kind: llm.KindUpstream, Provider: provider, Sent: true,
		Message: "malformed model list: " + err.Error(), Err: err,
	}
}
