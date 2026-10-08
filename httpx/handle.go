// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpx

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/leelsey/goose"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type responseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	opened  bool
}

func (rw *responseWriter) Open(status int, header http.Header) error {
	if rw.opened {
		return nil
	}
	rw.opened = true
	overlay(rw.w.Header(), header)
	rw.w.WriteHeader(status)
	rw.Flush()
	return nil
}

func (rw *responseWriter) Write(p []byte) (int, error) { return rw.w.Write(p) }

func (rw *responseWriter) Flush() error {
	if rw.flusher != nil {
		rw.flusher.Flush()
	}
	return nil
}

func (s *Server) handle(codec wire.Codec) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			s.fail(w, codec, wire.Options{}, &llm.Error{
				Kind: llm.KindInvalidRequest, Status: statusForRead(err), Message: "request body: " + err.Error(),
			})
			return
		}
		in := &wire.Inbound{Method: r.Method, Path: r.URL.Path, Query: r.URL.Query(), Header: r.Header, Body: body}

		req, opts, diags, derr := codec.Decode(in)
		if derr != nil {
			e := llm.AsError(derr)
			if e.Kind == llm.KindNotFound && s.opts.Passthrough != "" {
				r.Body = io.NopCloser(bytes.NewReader(body))
				s.passthrough(w, r)
				return
			}
			s.fail(w, codec, opts, e)
			return
		}
		renders := codec.Renders()

		callOpts := []goose.CallOption{
			goose.WithFormat(codec.Name()),
			goose.WithRenders(renders),
			goose.WithPath(r.URL.Path),
			goose.WithHeader(r.Header),
			// What decoding lost is the request's own loss, so the call holds
			// it to the policy before anything is sent.
			goose.WithDiagnostics(diags),
		}
		if opts.Session != "" {
			callOpts = append(callOpts,
				goose.WithSession(opts.Session),
				goose.WithSessionProvider(r.Header.Get(ProviderHeader)),
			)
		}
		if key := s.handedOn(r, opts.Key); key != "" {
			callOpts = append(callOpts, goose.WithCredential(llm.Credential{
				Key: key, Scheme: schemeOf(r, key),
			}))
		}

		if opts.Stream {
			s.serveStream(w, r, codec, req, opts, callOpts)
			return
		}
		s.serveUnary(w, r, codec, req, opts, callOpts)
	}
}

func (s *Server) serveUnary(w http.ResponseWriter, r *http.Request, codec wire.Codec,
	req *llm.Request, opts wire.Options, callOpts []goose.CallOption) {

	resp, err := s.client.Chat(r.Context(), req, callOpts...)
	if err != nil {
		e := llm.AsError(err)
		if resp != nil && e.Usage == nil {
			e.Usage = &resp.Usage
		}
		s.fail(w, codec, opts, e)
		return
	}
	out, encDiags, encErr := codec.EncodeResponse(resp, opts)
	if encErr != nil {
		s.fail(w, codec, opts, llm.AsError(encErr))
		return
	}
	// What the encoding lost is held to the policy the answer was, and nothing
	// has been written yet, so a loss it refuses is refused whole. The call was
	// served and billed all the same, and its row says so.
	if lerr := resp.Policy.Check(encDiags); lerr != nil {
		e := llm.AsError(lerr)
		e.Provider, e.Model, e.Sent, e.Usage = resp.Provider, resp.Model, true, &resp.Usage
		s.fail(w, codec, opts, e)
		return
	}
	all := append(append(llm.Diagnostics{}, resp.Diagnostics...), encDiags...)
	s.decorate(w, resp.Usage, resp.Provider, resp.Session, all)
	overlay(w.Header(), out.Header)
	w.WriteHeader(out.Status)
	_, _ = w.Write(out.Body)
}

// overlay writes a codec's headers over the ones the server set first, so a
// header both of them carry, such as the session, is sent once.
func overlay(dst, src http.Header) {
	for k, vs := range src {
		dst.Del(k)
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

func (s *Server) serveStream(w http.ResponseWriter, r *http.Request, codec wire.Codec,
	req *llm.Request, opts wire.Options, callOpts []goose.CallOption) {

	stream, err := s.client.Stream(r.Context(), req, callOpts...)
	if err != nil {
		s.fail(w, codec, opts, llm.AsError(err))
		return
	}
	defer stream.Close()
	// A stream the policy refused as it was handed over has delivered nothing,
	// so it is answered as the failed call it is rather than as a 200 carrying
	// only the failure.
	if e := llm.AsError(stream.Err()); e != nil && !stream.Committed() {
		refused, u := *e, stream.Usage()
		refused.Usage = &u
		refused.Diagnostics = append(append(llm.Diagnostics{}, stream.Diagnostics()...), e.Diagnostics...)
		s.fail(w, codec, opts, &refused)
		return
	}

	meta := stream.Meta()
	rw := &responseWriter{w: w}
	if f, ok := w.(http.Flusher); ok {
		rw.flusher = f
	} else {
		s.log().Warn("the response writer cannot flush, so this stream is buffered until it ends",
			"path", r.URL.Path, "provider", meta.Provider)
		stream.Note(llm.Diagnostic{
			Code: llm.WireBufferFull, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityInfo, Provider: meta.Provider, Format: codec.Name(),
			Detail: "the response writer cannot flush, so the stream is delivered in one piece",
		})
	}
	w.Header().Set("Trailer", strings.Join([]string{UsageHeader, DiagnosticsHeader}, ", "))
	w.Header().Set(ProviderHeader, meta.Provider)
	if meta.Session != "" {
		w.Header().Set(SessionHeader, meta.Session)
	}
	// A failure after this point is written into the body rather than returned,
	// so redaction has to reach it where the codec reads it.
	stream.Redact(s.hide)

	encDiags, encErr := codec.EncodeStream(r.Context(), stream, opts, rw)

	if b, err := json.Marshal(stream.Usage()); err == nil {
		w.Header().Set(UsageHeader, string(b))
	}
	all := append(append(llm.Diagnostics{}, stream.Diagnostics()...), encDiags...)
	if codes := all.Codes(); len(codes) > 0 {
		w.Header().Set(DiagnosticsHeader, strings.Join(codes, ","))
	}
	if len(encDiags) > 0 {
		s.log().Debug("stream diagnostics", "codes", encDiags.Codes(), "provider", meta.Provider)
	}
	if encErr != nil {
		e := llm.AsError(encErr)
		s.log().Warn("stream failed after the headers were committed",
			"provider", meta.Provider, "model", meta.Model,
			"kind", string(e.Kind), "code", e.Code, "error", e.Message)
	}
}

func (s *Server) decorate(w http.ResponseWriter, u llm.Usage, provider, session string, diags llm.Diagnostics) {
	if b, err := json.Marshal(u); err == nil {
		w.Header().Set(UsageHeader, string(b))
	}
	if provider != "" {
		w.Header().Set(ProviderHeader, provider)
	}
	if session != "" {
		w.Header().Set(SessionHeader, session)
	}
	if codes := diags.Codes(); len(codes) > 0 {
		w.Header().Set(DiagnosticsHeader, strings.Join(codes, ","))
	}
}

func (s *Server) fail(w http.ResponseWriter, codec wire.Codec, opts wire.Options, e *llm.Error) {
	e = s.hide(e)
	out := codec.EncodeError(e, opts)
	if out == nil {
		writeJSON(w, http.StatusBadGateway, map[string]any{"error": map[string]any{"message": e.Message}})
		return
	}
	for k, vs := range out.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	if e.Usage != nil {
		if b, err := json.Marshal(*e.Usage); err == nil {
			w.Header().Set(UsageHeader, string(b))
		}
	}
	// A refusal is the answer that most needs saying why, and the codes name
	// only goose's own findings, never an upstream's words.
	if codes := llm.Diagnostics(e.Diagnostics).Codes(); len(codes) > 0 {
		w.Header().Set(DiagnosticsHeader, strings.Join(codes, ","))
	}
	w.WriteHeader(out.Status)
	_, _ = w.Write(out.Body)
}

func statusForRead(err error) int {
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

func schemeOf(r *http.Request, key string) llm.CredentialScheme {
	switch {
	case key == "":
		return ""
	case r.Header.Get("x-api-key") == key:
		return llm.SchemeAnthropic
	case r.Header.Get("x-goog-api-key") == key, r.URL.Query().Get("key") == key:
		return llm.SchemeGoogle
	}
	return ""
}

// hide is what an upstream's failure may be repeated as. Both the message and
// the code are the vendor's own text, parsed out of a body that routinely
// quotes the request back, so both are replaced rather than only the sentence.
func (s *Server) hide(e *llm.Error) *llm.Error {
	if e == nil || !s.opts.RedactUpstream || e.Provider == "" || refusal(e) {
		return e
	}
	s.log().Debug("upstream failure", "provider", e.Provider, "status", e.Status,
		"code", e.Code, "error", e.Message, "body", e.Body)
	redacted := *e
	redacted.Message, redacted.Code, redacted.Body = generic(e.Kind), "", ""
	return &redacted
}

// refusal says a failure is goose's own, naming what it refused by goose's own
// code: a loss the policy would not take, whether before sending or in the
// answer, or a capability a backend lacks, refused before sending. Nothing in
// it came from a backend, and a caller told only that the upstream failed
// cannot fix what it asked.
func refusal(e *llm.Error) bool {
	if e.Body != "" || !llm.KnownCode(e.Code) {
		return false
	}
	return !e.Sent || e.Kind == llm.KindLoss
}

func generic(kind llm.ErrorKind) string {
	switch kind {
	case llm.KindAuth:
		return "the upstream rejected the credentials"
	case llm.KindPermission:
		return "the upstream refused this request"
	case llm.KindNotFound:
		return "the upstream has no such model"
	case llm.KindRateLimit:
		return "the upstream is rate limiting this proxy"
	case llm.KindOverloaded:
		return "the upstream is overloaded"
	case llm.KindTimeout:
		return "the upstream did not answer in time"
	case llm.KindInvalidRequest:
		return "the upstream rejected the request as malformed"
	default:
		return "the upstream failed to serve this request"
	}
}
