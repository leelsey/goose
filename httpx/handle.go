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
	for k, vs := range header {
		for _, v := range vs {
			rw.w.Header().Add(k, v)
		}
	}
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
		}
		if opts.Session != "" {
			callOpts = append(callOpts,
				goose.WithSession(opts.Session),
				goose.WithSessionProvider(r.Header.Get(ProviderHeader)),
			)
		}
		if opts.Key != "" && opts.Key != s.opts.APIKey {
			callOpts = append(callOpts, goose.WithCredential(llm.Credential{
				Key: opts.Key, Scheme: schemeOf(r, opts.Key),
			}))
		}

		if opts.Stream {
			s.serveStream(w, r, codec, req, opts, diags, callOpts)
			return
		}
		s.serveUnary(w, r, codec, req, opts, diags, callOpts)
	}
}

func (s *Server) serveUnary(w http.ResponseWriter, r *http.Request, codec wire.Codec,
	req *llm.Request, opts wire.Options, diags llm.Diagnostics, callOpts []goose.CallOption) {

	resp, err := s.client.Chat(r.Context(), req, callOpts...)
	if err != nil {
		e := llm.AsError(err)
		if resp != nil && e.Usage == nil {
			e.Usage = &resp.Usage
		}
		s.fail(w, codec, opts, e)
		return
	}
	resp.Diagnostics = append(resp.Diagnostics, diags...)
	out, encDiags, encErr := codec.EncodeResponse(resp, opts)
	if encErr != nil {
		s.fail(w, codec, opts, llm.AsError(encErr))
		return
	}
	all := append(append(llm.Diagnostics{}, resp.Diagnostics...), encDiags...)
	s.decorate(w, resp.Usage, resp.Provider, resp.Session, all)
	for k, vs := range out.Header {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	w.WriteHeader(out.Status)
	_, _ = w.Write(out.Body)
}

func (s *Server) serveStream(w http.ResponseWriter, r *http.Request, codec wire.Codec,
	req *llm.Request, opts wire.Options, diags llm.Diagnostics, callOpts []goose.CallOption) {

	stream, err := s.client.Stream(r.Context(), req, callOpts...)
	if err != nil {
		s.fail(w, codec, opts, llm.AsError(err))
		return
	}
	defer stream.Close()

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
	for _, d := range diags {
		stream.Note(d)
	}

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
	if s.opts.RedactUpstream && e.Provider != "" {
		s.log().Debug("upstream failure", "provider", e.Provider, "status", e.Status,
			"code", e.Code, "error", e.Message, "body", e.Body)
		redacted := *e
		redacted.Message = generic(e.Kind)
		e = &redacted
	}
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
