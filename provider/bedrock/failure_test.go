// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/leelsey/goose/internal/awsstream"
	"github.com/leelsey/goose/llm"
)

// Converse refuses a model it does not serve with the 400 a malformed request
// gets. Only a malformed request fails the same on every backend, so the
// chain walks on past the rest.
func TestAModelThisBackendDoesNotServeIsNotAMalformedRequest(t *testing.T) {
	for message, want := range map[string]llm.ErrorKind{
		"The provided model identifier is invalid.": llm.KindNotFound,
		"Invocation of model ID anthropic.claude-sonnet-4-5-20250929-v1:0 with on-demand throughput isn't supported. Retry your request with the ID or ARN of an inference profile that contains this model.": llm.KindNotFound,
		"messages.0.content: text content blocks must be non-empty": llm.KindInvalidRequest,
	} {
		p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Amzn-ErrorType", "ValidationException:http://internal.amazon.com/coral/com.amazon.bedrock/")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"message":"` + message + `"}`))
		})
		_, err := p.Complete(context.Background(), attempt(hello()))
		if e := llm.AsError(err); e.Kind != want || e.Message != message {
			t.Errorf("unary %q: err = %+v, want %s", message, e, want)
		}
		_, err = p.Stream(context.Background(), attempt(hello()))
		if e := llm.AsError(err); e.Kind != want {
			t.Errorf("stream %q: err = %+v, want %s", message, e, want)
		}
	}
}

// Converse refuses a model the account has not been granted, or cannot
// subscribe to, with the 403 a credential without rights gets. Counted against
// the provider, a caller naming one such model would open the breaker for
// every other model the account is served, so only that refusal reads as a
// model this backend does not serve. Regression test for r4-prov-3.
func TestAModelTheAccountMayNotUseIsNotTheProvidersFault(t *testing.T) {
	for message, want := range map[string]llm.ErrorKind{
		"You don't have access to the model with the specified model ID.": llm.KindNotFound,
		"Model access is denied due to IAM user or service role is not authorized to perform the required AWS Marketplace actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe) to enable access to this model.": llm.KindNotFound,
		"User: arn:aws:iam::123456789012:user/u is not authorized to perform: bedrock:InvokeModel":                                                                                                                                llm.KindPermission,
	} {
		p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("X-Amzn-ErrorType", "AccessDeniedException:http://internal.amazon.com/coral/com.amazon.bedrock/")
			w.WriteHeader(http.StatusForbidden)
			_, _ = w.Write([]byte(`{"message":"` + message + `"}`))
		})
		_, err := p.Complete(context.Background(), attempt(hello()))
		if e := llm.AsError(err); e.Kind != want || e.Status != http.StatusForbidden {
			t.Errorf("unary %q: err = %+v, want %s", message, e, want)
		}
		_, err = p.Stream(context.Background(), attempt(hello()))
		if e := llm.AsError(err); e.Kind != want {
			t.Errorf("stream %q: err = %+v, want %s", message, e, want)
		}
	}
}

// A failure reported inside a 200 or as a stream's exception is the upstream's
// words as much as a refusal's body is, and the ledger is kept free of both
// alike. goose's own words for a failure that came with none are kept.
func TestAFailureInsideA200IsRedactedLikeARefusal(t *testing.T) {
	const secret = "SECRET upstream words quoting the prompt"
	redacted := func(t *testing.T, e *llm.Error) {
		t.Helper()
		if e == nil || !strings.Contains(e.Message, "SECRET") {
			t.Fatalf("err = %+v", e)
		}
		if got := e.Reason(true); got != string(e.Kind) {
			t.Fatalf("redacted reason = %q: the upstream's words reached the ledger", got)
		}
		if e.Detail(true) != "" || !strings.Contains(e.Detail(false), "SECRET") {
			t.Fatalf("detail = %q / %q", e.Detail(true), e.Detail(false))
		}
	}

	t.Run("unary", func(t *testing.T) {
		p, _, _ := serve(t, func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"__type":"ModelErrorException","message":"` + secret + `"}`))
		})
		_, err := p.Complete(context.Background(), attempt(hello()))
		redacted(t, llm.AsError(err))
	})

	b := &parser{provider: "bedrock", blocks: map[int]string{}}
	t.Run("exception", func(t *testing.T) {
		redacted(t, b.exception(awsstream.Message{
			Headers: map[string]string{":message-type": "exception", ":exception-type": "modelStreamErrorException"},
			Payload: []byte(`{"message":"` + secret + `"}`),
		}))
	})
	t.Run("error headers", func(t *testing.T) {
		redacted(t, b.exception(awsstream.Message{Headers: map[string]string{
			":message-type": "error", ":error-code": "internalServerException", ":error-message": secret,
		}}))
	})
	t.Run("no words", func(t *testing.T) {
		e := b.exception(awsstream.Message{Headers: map[string]string{":message-type": "error"}})
		if got := e.Reason(true); got != e.Message || got == "" {
			t.Fatalf("redacted reason = %q: goose's own words hold nobody's request", got)
		}
	})
}
