// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"sort"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type candidate struct {
	Content      *content `json:"content,omitempty"`
	FinishReason string   `json:"finishReason,omitempty"`
	Index        int      `json:"index"`
}

type content struct {
	Role  string            `json:"role"`
	Parts []json.RawMessage `json:"parts"`
}

type outBody struct {
	ResponseID     string         `json:"responseId,omitempty"`
	Candidates     []candidate    `json:"candidates,omitempty"`
	PromptFeedback *promptFeed    `json:"promptFeedback,omitempty"`
	UsageMetadata  *usageMetadata `json:"usageMetadata,omitempty"`
	ModelVersion   string         `json:"modelVersion,omitempty"`
}

type promptFeed struct {
	BlockReason string `json:"blockReason,omitempty"`
}

func (c *Codec) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	if resp == nil {
		return nil, nil, llm.Errorf(llm.KindInternal, "nil response")
	}
	var diags llm.Diagnostics
	body := outBody{ResponseID: resp.ID, UsageMetadata: encodeUsage(resp.Usage), ModelVersion: resp.Model}
	diags = append(diags, resp.Usage.Uncarried(Name, llm.FigureCacheRead, llm.FigureReasoning)...)
	for _, ch := range resp.Choices {
		parts, d := encodeParts(ch.Message.Content, true)
		diags = append(diags, d...)
		name, clean := finishName(ch.Stop)
		if !clean {
			diags = append(diags, unknownStop(ch.Stop))
		}
		body.Candidates = append(body.Candidates, candidate{
			Content:      &content{Role: "model", Parts: list(parts)},
			FinishReason: name,
			Index:        ch.Index,
		})
	}
	if resp.Block != nil {
		body.PromptFeedback = &promptFeed{BlockReason: resp.Block.Reason}
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode: %v", err)
	}
	raw, err = wire.WithExtra(raw, resp.Extra)
	if err != nil {
		return nil, diags, llm.Errorf(llm.KindInternal, "encode: %v", err)
	}
	return &wire.Outbound{Status: http.StatusOK, Header: jsonHeader(resp.Session), Body: raw}, diags, nil
}

// EncodeParts writes parts for a Gemini backend, each signature exactly as it
// was issued.
func EncodeParts(list []llm.Part) ([]json.RawMessage, llm.Diagnostics) {
	return encodeParts(list, false)
}

// encodeParts writes parts for a backend or, with client set, for a client,
// who is handed another family's signature marked as that family's.
func encodeParts(list []llm.Part, client bool) ([]json.RawMessage, llm.Diagnostics) {
	var diags llm.Diagnostics
	out := []json.RawMessage{}
	for _, p := range list {
		sig := signatureFor(p, client)
		switch v := p.(type) {
		case *llm.Text:
			out = append(out, signed(map[string]any{"text": v.Text}, sig))
		case *llm.Refusal:
			diags = append(diags, drop(llm.DropRefusal, "this format has no field for a refusal, so it travelled as ordinary text"))
			out = append(out, mustJSON(map[string]any{"text": v.Text}))
		case *llm.Thinking:
			if len(v.Redacted) > 0 {
				diags = append(diags, drop(llm.DropThinking, "redacted thinking has no part on this wire"))
				continue
			}
			// A reasoning item another backend stored by name alone holds
			// nothing this wire can carry.
			if v.Text == "" && sig == "" {
				continue
			}
			out = append(out, signed(map[string]any{"text": v.Text, "thought": true}, sig))
		case *llm.ToolUse:
			out = append(out, functionCall(v, sig, &diags))
		case *llm.ToolResult:
			out = append(out, mustJSON(map[string]any{"functionResponse": map[string]any{
				"name": v.Name, "response": responseObject(v.Content),
			}}))
			diags = append(diags, llm.Flattened(v.Content, "",
				"this wire takes a tool result as an object of text, so what was not text did not travel")...)
		case *llm.Image:
			if len(v.Data) > 0 {
				out = append(out, signed(map[string]any{"inlineData": map[string]any{
					"mimeType": v.MIME, "data": base64.StdEncoding.EncodeToString(v.Data),
				}}, sig))
				continue
			}
			out = append(out, signed(map[string]any{"fileData": map[string]any{"mimeType": v.MIME, "fileUri": v.URL}}, sig))
		case *llm.Raw:
			if v.Vendor == Name && json.Valid(v.JSON) {
				out = append(out, v.JSON)
				continue
			}
			diags = append(diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
		}
	}
	return out, diags
}

func signed(part map[string]any, signature string) json.RawMessage {
	if signature != "" {
		part["thoughtSignature"] = signature
	}
	return mustJSON(part)
}

// maxHeldThought is how much thinking may wait for its signature, across
// every candidate of a stream at once.
const maxHeldThought = wirecore.DefaultMaxBuffer

// thoughtHold keeps, for each candidate, the thinking block still waiting on
// its signature, so a signature that streams apart from its text (as the
// Claude-family wires split it) lands on the text it belongs to instead of
// travelling as an empty part of its own. How much waits is the upstream's to
// choose, so it is bounded, and each delta is added to it rather than copying
// all that came before.
type thoughtHold struct {
	held map[int]*heldThought
	size int
	max  int
	full bool
}

type heldThought struct {
	id     string
	text   []byte
	sig    string
	signer llm.Family
}

func newThoughtHold(max int) *thoughtHold {
	return &thoughtHold{held: map[int]*heldThought{}, max: max}
}

// join folds a streamed thinking delta into the block pending for its
// candidate. It returns the parts the caller must place in the stream now: an
// earlier block v does not continue, a block that just gained its signature,
// or one that outgrew the hold and goes out unsigned rather than wait. full
// reports, the first time only, that one did.
func (t *thoughtHold) join(choice int, v *llm.Thinking) (ready []llm.Part, full bool) {
	h := t.held[choice]
	if h != nil && h.id != "" && v.ID != "" && h.id != v.ID {
		ready = append(ready, t.flush(choice))
		h = nil
	}
	if h != nil && t.size+len(v.Text) > t.max {
		// What the block holds so far goes out, and the rest of it waits on its
		// own: the signature that ends it will not sign the part sent ahead.
		id := h.id
		ready = append(ready, t.flush(choice))
		full = t.overflow()
		h = &heldThought{id: id}
		t.held[choice] = h
	}
	if h == nil {
		h = &heldThought{}
		t.held[choice] = h
	}
	if h.id == "" {
		h.id = v.ID
	}
	h.text = append(h.text, v.Text...)
	t.size += len(v.Text)
	if v.Signature != "" {
		h.sig, h.signer = v.Signature, v.Signer
	}
	switch {
	case h.sig != "":
		ready = append(ready, t.flush(choice))
	case len(h.text) == 0:
		delete(t.held, choice)
	case t.size > t.max:
		ready = append(ready, t.flush(choice))
		full = t.overflow() || full
	}
	return ready, full
}

// flush takes the block pending for a candidate out of the hold, or returns
// nil when there is none.
func (t *thoughtHold) flush(choice int) llm.Part {
	h := t.held[choice]
	if h == nil {
		return nil
	}
	delete(t.held, choice)
	t.size -= len(h.text)
	return &llm.Thinking{ID: h.id, Text: string(h.text), Signature: h.sig, Signer: h.signer}
}

func (t *thoughtHold) overflow() bool {
	first := !t.full
	t.full = true
	return first
}

func heldThoughtFull() llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.WireBufferFull, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name,
		Detail: "a thinking block outgrew what may wait for its signature, so part of it went out without one and the signature that followed does not sign it",
	}
}

func signatureFor(p llm.Part, client bool) string {
	sig, signer := wirecore.Signed(p)
	if !client {
		return sig
	}
	return wirecore.SealBytesSignature(llm.FamilyGoogle, signer, sig)
}

func functionCall(t *llm.ToolUse, sig string, diags *llm.Diagnostics) json.RawMessage {
	args, broken := wirecore.ObjectArgsBroken(t.Args)
	if broken {
		d := drop(llm.DropToolCall, "this call's arguments did not survive, so it travelled with none rather than with what was meant")
		d.Field = t.Name
		diags.Note(d)
	}
	return signed(map[string]any{"functionCall": map[string]any{"name": t.Name, "args": args}}, sig)
}

func responseObject(parts []llm.Part) json.RawMessage {
	text := llm.TextOf(parts)
	if json.Valid([]byte(text)) && len(text) > 0 && (text[0] == '{') {
		return json.RawMessage(text)
	}
	return mustJSON(map[string]any{"result": text})
}

func (c *Codec) EncodeStream(ctx context.Context, s *llm.Stream, opts wire.Options, w wire.Writer) (llm.Diagnostics, error) {
	var diags llm.Diagnostics
	sse := wire.NewSSE(w)
	meta := s.Meta()
	if err := sse.Open(streamHeader(meta.Session)); err != nil {
		return diags, err
	}

	tools := map[int]*wirecore.ToolMerger{}
	stop := map[int]llm.StopReason{}
	hold := newThoughtHold(maxHeldThought)
	var candidates []int
	seen := func(choice int) {
		if _, ok := tools[choice]; ok {
			return
		}
		m := wirecore.NewToolMerger()
		if len(candidates) > 0 {
			m.Share(tools[candidates[0]])
		}
		tools[choice] = m
		candidates = append(candidates, choice)
	}
	var block *llm.Block
	var streamErr error
	checked := 0
	emitted, refused := false, false

	send := func(b outBody, extra map[string]json.RawMessage) error {
		raw, err := json.Marshal(b)
		if err != nil {
			return err
		}
		if raw, err = wire.WithExtra(raw, extra); err != nil {
			return err
		}
		emitted = true
		return sse.Send("", raw)
	}

	for fr := range s.Frames() {
		if err := ctx.Err(); err != nil {
			streamErr = err
			break
		}
		if fr.Err != nil {
			streamErr = fr.Err
			break
		}
		if fr.Block != nil {
			block = fr.Block
		}
		choice := fr.ChoiceIndex()
		seen(choice)
		var live []llm.Part
		for _, p := range fr.Delta {
			if th, ok := p.(*llm.Thinking); ok && len(th.Redacted) == 0 {
				// A backend that streams a thinking block's signature apart from
				// its text must not have the two land as separate parts; hold
				// the block until it is signed or interrupted by other content.
				ready, full := hold.join(choice, th)
				if full {
					diags = append(diags, heldThoughtFull())
				}
				live = append(live, ready...)
				continue
			}
			// A tool call (or any other part) interrupts a still-unsigned held
			// block just as much as a text delta would; flush it first so it
			// never absorbs a later, unrelated thinking block's signature.
			if held := hold.flush(choice); held != nil {
				live = append(live, held)
			}
			if t, ok := p.(*llm.ToolUse); ok {
				// The merger keeps a call's name and arguments; the signature is
				// this wire's and has to be kept with the call it came on.
				call := tools[choice].Merge(t, nil)
				if t.Signature != "" {
					call.Signature, call.Signer = t.Signature, t.Signer
				}
				continue
			}
			live = append(live, p)
		}
		if len(live) > 0 {
			parts, d := encodeParts(live, true)
			for _, x := range d {
				// A refusal streams as many deltas, and it is one refusal.
				if x.Code == llm.DropRefusal {
					if refused {
						continue
					}
					refused = true
				}
				diags = append(diags, x)
			}
			// What a frame loses here is known before any of it is written, so
			// a loss the stream's policy will not take is refused in its place.
			if streamErr = s.Check(diags[checked:]); streamErr != nil {
				break
			}
			checked = len(diags)
			body := outBody{Candidates: []candidate{{Content: &content{Role: "model", Parts: list(parts)}, Index: choice}}}
			if len(parts) > 0 || len(fr.Extra) > 0 {
				if err := send(body, fr.Extra); err != nil {
					return diags, err
				}
			}
		}
		if fr.Stop != llm.StopNone {
			stop[choice] = fr.Stop
		}
	}

	if streamErr == nil {
		streamErr = s.Err()
	}

	final := outBody{ResponseID: meta.ID, UsageMetadata: encodeUsage(s.Usage()), ModelVersion: meta.Model}
	diags = append(diags, s.Usage().Uncarried(Name, llm.FigureCacheRead, llm.FigureReasoning)...)
	if len(candidates) == 0 {
		// A stream can fail, be abandoned or end without ever naming a
		// candidate; seen() is what puts one on the list.
		seen(0)
	}
	sort.Ints(candidates)
	tailParts := map[int][]json.RawMessage{}
	tailNames := map[int]string{}
	for _, choice := range candidates {
		var parts []json.RawMessage
		if held := hold.flush(choice); held != nil {
			p, d := encodeParts([]llm.Part{held}, true)
			parts = append(parts, p...)
			diags = append(diags, d...)
		}
		for _, t := range tools[choice].Tools() {
			parts = append(parts, functionCall(t, signatureFor(t, true), &diags))
		}
		name, clean := finishName(stop[choice])
		if !clean {
			diags = append(diags, unknownStop(stop[choice]))
		}
		tailParts[choice], tailNames[choice] = parts, name
	}
	for _, choice := range candidates {
		if tools[choice].Overflowed() {
			diags = append(diags, llm.Diagnostic{
				Code: llm.WireBufferFull, Class: llm.ClassContent, Stage: llm.StageEncode,
				Severity: llm.SeverityLoss, Format: Name,
				Detail: "the tool arguments outgrew what one turn may hold, so the call is short of what was streamed",
			})
			break
		}
	}
	if block != nil {
		final.PromptFeedback = &promptFeed{BlockReason: block.Reason}
	}
	if streamErr == nil {
		streamErr = s.Check(diags[checked:])
	}
	// A refusal found only here, or one already open, means the content built
	// above is exactly what was refused (or came after what was); it must not
	// follow the error onto the wire, though the candidate that closes the
	// stream still does.
	withheld := streamErr != nil
	for _, choice := range candidates {
		parts, name := tailParts[choice], tailNames[choice]
		if withheld {
			parts = nil
		}
		if len(parts) > 0 || name != "" || !emitted {
			final.Candidates = append(final.Candidates, candidate{
				Content: &content{Role: "model", Parts: list(parts)}, FinishReason: name, Index: choice,
			})
		}
	}
	if streamErr != nil {
		e := llm.AsError(streamErr)
		_ = sse.Send("", mustJSON(errorBody(e)))
		diags = append(diags, llm.Diagnostic{
			Code: llm.WireErrorInBody, Class: llm.ClassTransport, Stage: llm.StageEncode,
			Severity: llm.SeverityLoss, Format: Name, Detail: e.Message,
		})
	}
	if err := send(final, nil); err != nil {
		return diags, err
	}
	diags = append(diags, s.Diagnostics()...)
	return diags, streamErr
}

func (c *Codec) EncodeError(err *llm.Error, _ wire.Options) *wire.Outbound {
	e := err
	if e == nil {
		e = llm.Errorf(llm.KindInternal, "unknown error")
	}
	status := e.Status
	if status == 0 || status < 400 {
		status = wire.StatusFor(e.Kind)
	}
	body, _ := json.Marshal(errorBody(e))
	h := jsonHeader("")
	wire.SetRetryAfter(h, e)
	return &wire.Outbound{Status: status, Header: h, Body: body}
}

func errorBody(e *llm.Error) map[string]any {
	status := e.Status
	if status == 0 || status < 400 {
		status = wire.StatusFor(e.Kind)
	}
	return map[string]any{"error": map[string]any{
		"code": status, "message": e.Message, "status": statusName(e.Kind),
	}}
}

func statusName(k llm.ErrorKind) string {
	switch k {
	case llm.KindInvalidRequest, llm.KindLoss, llm.KindUnsupported:
		return "INVALID_ARGUMENT"
	case llm.KindAuth:
		return "UNAUTHENTICATED"
	case llm.KindPermission:
		return "PERMISSION_DENIED"
	case llm.KindNotFound, llm.KindNoRoute:
		return "NOT_FOUND"
	case llm.KindRateLimit:
		return "RESOURCE_EXHAUSTED"
	case llm.KindOverloaded, llm.KindSaturated:
		return "UNAVAILABLE"
	case llm.KindTimeout:
		return "DEADLINE_EXCEEDED"
	case llm.KindCancelled:
		return "CANCELLED"
	default:
		return "INTERNAL"
	}
}

func unknownStop(s llm.StopReason) llm.Diagnostic {
	return llm.Diagnostic{
		Code: llm.DropUnknownBlock, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Field: string(s),
		Detail: "stop reason has no spelling on this wire",
	}
}

func drop(code, detail string) llm.Diagnostic {
	return llm.Diagnostic{
		Code: code, Class: llm.ClassContent, Stage: llm.StageEncode,
		Severity: llm.SeverityLoss, Format: Name, Detail: detail,
	}
}

func jsonHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"application/json"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func streamHeader(session string) http.Header {
	h := http.Header{"Content-Type": []string{"text/event-stream"}}
	if session != "" {
		h.Set("X-Goose-Session", session)
	}
	return h
}

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(`{"text":""}`)
	}
	return b
}

func list(parts []json.RawMessage) []json.RawMessage {
	if parts == nil {
		return []json.RawMessage{}
	}
	return parts
}
