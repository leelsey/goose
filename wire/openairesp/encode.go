// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package openairesp

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/leelsey/goose/internal/wirecore"
	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

type item struct {
	Type    string        `json:"type"`
	ID      string        `json:"id,omitempty"`
	Role    string        `json:"role,omitempty"`
	Status  string        `json:"status,omitempty"`
	Content []contentPart `json:"content,omitempty"`
	Summary []summaryPart `json:"summary,omitempty"`
	CallID  string        `json:"call_id,omitempty"`
	Name    string        `json:"name,omitempty"`
	Args    string        `json:"arguments,omitempty"`

	Result        string `json:"result,omitempty"`
	OutputFormat  string `json:"output_format,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`

	EncryptedContent string `json:"encrypted_content,omitempty"`

	// raw is an item this wire does not model, written as the backend wrote
	// it: read into the fields above it would lose every one they do not name.
	raw json.RawMessage
}

// rawItem is a backend's own item on this wire, carried whole.
func rawItem(v *llm.Raw) (item, bool) {
	if v.Vendor != Name {
		return item{}, false
	}
	var it item
	if json.Unmarshal(v.JSON, &it) != nil || it.Type == "" {
		return item{}, false
	}
	it.raw = v.JSON
	return it, true
}

// mintedReasoning starts the id goose gives a reasoning item the backend did
// not name. Read back as the backend's, it would reach OpenAI as the name of an
// item it never stored.
const mintedReasoning = "rs_goose"

// sameReasoning says whether a thinking part belongs to the reasoning item
// already open. A signature closes its block, and a backend that names its
// items says where one ends; merging two would keep only one signature.
func sameReasoning(named, signature string, v *llm.Thinking) bool {
	return signature == "" && (v.ID == "" || named == "" || v.ID == named)
}

func imageFormat(mime string) string {
	if mime == "" {
		return "png"
	}
	return strings.TrimPrefix(mime, "image/")
}

func imageItem(v *llm.Image, id string, status string) item {
	return item{
		Type: "image_generation_call", ID: id, Status: status,
		Result:        base64.StdEncoding.EncodeToString(v.Data),
		OutputFormat:  imageFormat(v.MIME),
		RevisedPrompt: v.Prompt,
	}
}

type contentPart struct {
	Type        string   `json:"type"`
	Text        string   `json:"text,omitempty"`
	Refusal     string   `json:"refusal,omitempty"`
	Annotations []string `json:"annotations,omitempty"`
}

// MarshalJSON writes what this part's own type requires. One struct serves two
// shapes, so omitting a field because it is empty would drop one the schema
// calls for and a strict client would refuse the answer.
func (c contentPart) MarshalJSON() ([]byte, error) {
	type plain contentPart
	out := plain(c)
	switch c.Type {
	case "output_text":
		if out.Annotations == nil {
			out.Annotations = []string{}
		}
		return json.Marshal(struct {
			plain
			Text        string   `json:"text"`
			Annotations []string `json:"annotations"`
		}{plain: out, Text: c.Text, Annotations: out.Annotations})
	case "refusal":
		return json.Marshal(struct {
			plain
			Refusal string `json:"refusal"`
		}{plain: out, Refusal: c.Refusal})
	}
	return json.Marshal(out)
}

// MarshalJSON writes what this item's own type requires, for the same reason.
func (i item) MarshalJSON() ([]byte, error) {
	if i.raw != nil {
		return i.raw, nil
	}
	type plain item
	out := plain(i)
	switch i.Type {
	case "reasoning":
		if out.Summary == nil {
			out.Summary = []summaryPart{}
		}
		return json.Marshal(struct {
			plain
			Summary []summaryPart `json:"summary"`
		}{plain: out, Summary: out.Summary})
	case "message":
		if out.Content == nil {
			out.Content = []contentPart{}
		}
		return json.Marshal(struct {
			plain
			Content []contentPart `json:"content"`
		}{plain: out, Content: out.Content})
	case "function_call":
		return json.Marshal(struct {
			plain
			CallID string `json:"call_id"`
			Name   string `json:"name"`
			Args   string `json:"arguments"`
		}{plain: out, CallID: i.CallID, Name: i.Name, Args: i.Args})
	case "image_generation_call":
		return json.Marshal(struct {
			plain
			Result string `json:"result"`
		}{plain: out, Result: i.Result})
	}
	return json.Marshal(out)
}

type summaryPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type responseObject struct {
	ID                 string          `json:"id"`
	Object             string          `json:"object"`
	CreatedAt          int64           `json:"created_at"`
	Status             string          `json:"status"`
	Model              string          `json:"model"`
	Output             []item          `json:"output"`
	Usage              *wireUsage      `json:"usage,omitempty"`
	Tier               string          `json:"service_tier,omitempty"`
	IncompleteDetails  *incompleteInfo `json:"incomplete_details,omitempty"`
	Error              *errorInfo      `json:"error,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
}

type incompleteInfo struct {
	Reason string `json:"reason"`
}

type errorInfo struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
}

func (c *Codec) items(parts []llm.Part, idFor func(string) string) ([]item, llm.Diagnostics) {
	var diags llm.Diagnostics
	var out []item
	var msg *item
	var reason *item
	var named string
	unsigned := 0

	closeAll := func() {
		if reason != nil {
			out = append(out, *reason)
			reason = nil
		}
		if msg != nil {
			out = append(out, *msg)
			msg = nil
		}
	}

	for _, p := range parts {
		if sig, _ := wirecore.Signed(p); sig != "" && p.Kind() != llm.KindThinking {
			unsigned++
		}
		switch v := p.(type) {
		case *llm.Thinking:
			if msg != nil {
				closeAll()
			}
			if len(v.Redacted) > 0 {
				diags = append(diags, drop(llm.DropThinking, "redacted thinking has no item on this wire"))
				continue
			}
			if reason != nil && !sameReasoning(named, reason.EncryptedContent, v) {
				out = append(out, *reason)
				reason = nil
			}
			if reason == nil {
				named = v.ID
				id := v.ID
				if id == "" {
					id = idFor(mintedReasoning)
				}
				reason = &item{Type: "reasoning", ID: id, Summary: []summaryPart{}}
			}
			if v.Text != "" {
				reason.Summary = append(reason.Summary, summaryPart{Type: "summary_text", Text: v.Text})
			}
			if v.Signature != "" {
				reason.EncryptedContent = wirecore.SealSignature(llm.FamilyOpenAI, v.Signer, v.Signature)
			}
		case *llm.Text:
			// A part carrying nothing but a signature has nothing for a message
			// to hold, and the stream leaves it out too.
			if v.Text == "" {
				continue
			}
			if reason != nil {
				out = append(out, *reason)
				reason = nil
			}
			if msg == nil {
				msg = &item{Type: "message", ID: idFor("msg"), Role: "assistant", Status: "completed"}
			}
			msg.Content = append(msg.Content, contentPart{Type: "output_text", Text: v.Text, Annotations: []string{}})
		case *llm.Refusal:
			if reason != nil {
				out = append(out, *reason)
				reason = nil
			}
			if msg == nil {
				msg = &item{Type: "message", ID: idFor("msg"), Role: "assistant", Status: "completed"}
			}
			msg.Content = append(msg.Content, contentPart{Type: "refusal", Refusal: v.Text})
		case *llm.ToolUse:
			closeAll()
			args := string(v.Args)
			if args == "" {
				args = "{}"
			}
			out = append(out, item{Type: "function_call", ID: idFor("fc"), CallID: v.ID, Name: v.Name, Args: args, Status: "completed"})
		case *llm.Image:
			closeAll()
			id := v.ID
			if id == "" {
				id = idFor("ig")
			}
			out = append(out, imageItem(v, id, "completed"))
		case *llm.ToolResult:
			diags = append(diags, drop(llm.DropToolResult, "a tool result cannot ride on an assistant turn here"))
		case *llm.Raw:
			if it, ok := rawItem(v); ok {
				closeAll()
				out = append(out, it)
				continue
			}
			diags = append(diags, drop(llm.DropUnknownBlock, "content block "+v.Type+" has no shape on this wire"))
		}
	}
	closeAll()
	if unsigned > 0 {
		diags = append(diags, unsignedParts(unsigned))
	}
	return out, diags
}

// unsignedParts says Gemini's signatures on calls, text or images did not
// travel: this wire signs only a reasoning item, and the model's reasoning
// before those parts is lost to the next turn.
func unsignedParts(n int) llm.Diagnostic {
	d := drop(llm.DropSignature, "this wire signs only a reasoning item, so the reasoning signed onto a call, a text or an image did not travel with it")
	d.Count = n
	return d
}

func (c *Codec) EncodeResponse(resp *llm.Response, opts wire.Options) (*wire.Outbound, llm.Diagnostics, error) {
	if resp == nil {
		return nil, nil, llm.Errorf(llm.KindInternal, "nil response")
	}
	var diags llm.Diagnostics
	n := 0
	idFor := func(prefix string) string { n++; return prefix + "_" + strconv.Itoa(n) }

	u := encodeUsage(resp.Usage)
	diags = append(diags, resp.Usage.Uncarried(Name, llm.FigureCacheRead, llm.FigureReasoning, llm.FigureTier)...)
	body := responseObject{
		ID:                 c.id(resp.ID),
		Object:             "response",
		CreatedAt:          created(resp.Created),
		Model:              modelName(resp, opts),
		Output:             []item{},
		Usage:              &u,
		Tier:               resp.Usage.Tier,
		PreviousResponseID: opts.Session,
	}

	stop := resp.Stop
	if ch := resp.First(); ch != nil {
		stop = ch.Stop
		out, d := c.items(ch.Message.Content, idFor)
		diags = append(diags, d...)
		body.Output = out
	}
	if len(resp.Choices) > 1 {
		diags = append(diags, llm.Diagnostic{
			Code: llm.DropCandidate, Class: llm.ClassContent, Stage: llm.StageEncode,
			Severity: llm.SeverityFatal, Format: Name, Count: len(resp.Choices) - 1,
			Detail: "this wire carries one candidate",
		})
	}
	state, incomplete, clean := status(stop, false, false)
	if !clean {
		diags = append(diags, unknownStop(stop))
	}
	if resp.Block != nil {
		state, incomplete = "incomplete", "content_filter"
	}
	body.Status = state
	if incomplete != "" {
		body.IncompleteDetails = &incompleteInfo{Reason: incomplete}
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

func errorCode(code string) any {
	if code == "" {
		return nil
	}
	return code
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
	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{"message": e.Message, "type": string(e.Kind), "code": errorCode(e.Code)},
	})
	h := jsonHeader("")
	wire.SetRetryAfter(h, e)
	return &wire.Outbound{Status: status, Header: h, Body: body}
}

func modelName(resp *llm.Response, opts wire.Options) string {
	if opts.RequestModel != "" {
		return opts.RequestModel
	}
	if resp.Requested != "" {
		return resp.Requested
	}
	return resp.Model
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

func created(t time.Time) int64 {
	if t.IsZero() {
		return time.Now().Unix()
	}
	return t.Unix()
}
