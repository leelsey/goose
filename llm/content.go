// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import "encoding/json"

type PartKind string

const (
	KindText       PartKind = "text"
	KindThinking   PartKind = "thinking"
	KindToolUse    PartKind = "tool_use"
	KindToolResult PartKind = "tool_result"
	KindImage      PartKind = "image"
	KindRefusal    PartKind = "refusal"
	KindRaw        PartKind = "raw"
)

type Part interface {
	Kind() PartKind
	part()
}

// Family names the vendors whose signatures verify one another's. Anthropic
// documents its signatures as good on its own API, on Bedrock and on Vertex
// alike, so Claude on any of them is one family; OpenAI and Azure OpenAI are
// another, and Gemini on its own API or on Vertex a third. A signature minted
// by one family is refused by the others, whatever it is attached to.
type Family string

const (
	FamilyAnthropic Family = "anthropic"
	FamilyOpenAI    Family = "openai"
	FamilyGoogle    Family = "google"
)

type Text struct {
	Text string
	// Signature is Gemini's. It signs the reasoning behind the part it arrived
	// on, which is text or a call as often as a thought, and a replayed turn
	// that has lost it is refused. A text part may carry nothing else.
	Signature string
	// Signer is the family that minted Signature. Empty says nobody recorded
	// it, and such a signature is left for the backend to judge.
	Signer Family
}

type Thinking struct {
	// ID is the backend's own name for the block, where it gives one, so a
	// client on that backend's wire is handed the item it was issued.
	ID        string
	Text      string
	Signature string
	Signer    Family
	// Redacted is the block exactly as the backend spelled it. It is opaque and
	// a client hands it straight back, so nothing re-spells it on the way
	// through; a backend whose own wire spells it differently converts at its
	// own edge.
	Redacted []byte
}

type ToolUse struct {
	ID        string
	Name      string
	Args      json.RawMessage
	Index     *int
	Signature string
	Signer    Family
}

type ToolResult struct {
	ID      string
	Name    string
	Content []Part
	IsError bool
}

type Image struct {
	ID      string
	MIME    string
	Data    []byte
	URL     string
	Detail  string
	Prompt  string
	Index   *int
	Partial bool
	// Signature is Gemini's, on an image it generated: an edit of that image
	// on the next turn is refused without it.
	Signature string
	Signer    Family
}

type Refusal struct {
	Text string
}

type Raw struct {
	Vendor string
	Type   string
	JSON   json.RawMessage
}

func (*Text) Kind() PartKind       { return KindText }
func (*Thinking) Kind() PartKind   { return KindThinking }
func (*ToolUse) Kind() PartKind    { return KindToolUse }
func (*ToolResult) Kind() PartKind { return KindToolResult }
func (*Image) Kind() PartKind      { return KindImage }
func (*Refusal) Kind() PartKind    { return KindRefusal }
func (*Raw) Kind() PartKind        { return KindRaw }

func (*Text) part()       {}
func (*Thinking) part()   {}
func (*ToolUse) part()    {}
func (*ToolResult) part() {}
func (*Image) part()      {}
func (*Refusal) part()    {}
func (*Raw) part()        {}

func TextOf(parts []Part) string {
	var b []byte
	for _, p := range parts {
		if t, ok := p.(*Text); ok {
			b = append(b, t.Text...)
		}
	}
	return string(b)
}

func RefusalOf(parts []Part) string {
	for _, p := range parts {
		if r, ok := p.(*Refusal); ok && r.Text != "" {
			return r.Text
		}
	}
	return ""
}

func ToolUses(parts []Part) []*ToolUse {
	var out []*ToolUse
	for _, p := range parts {
		if t, ok := p.(*ToolUse); ok {
			out = append(out, t)
		}
	}
	return out
}

func HasKind(parts []Part, k PartKind) bool {
	for _, p := range parts {
		if p.Kind() == k {
			return true
		}
	}
	return false
}

func clonePart(p Part) Part {
	switch v := p.(type) {
	case *Text:
		c := *v
		return &c
	case *Thinking:
		c := *v
		c.Redacted = append([]byte(nil), v.Redacted...)
		return &c
	case *ToolUse:
		c := *v
		c.Args = append(json.RawMessage(nil), v.Args...)
		if v.Index != nil {
			i := *v.Index
			c.Index = &i
		}
		return &c
	case *ToolResult:
		c := *v
		c.Content = CloneParts(v.Content)
		return &c
	case *Image:
		c := *v
		c.Data = append([]byte(nil), v.Data...)
		if v.Index != nil {
			i := *v.Index
			c.Index = &i
		}
		return &c
	case *Refusal:
		c := *v
		return &c
	case *Raw:
		c := *v
		c.JSON = append(json.RawMessage(nil), v.JSON...)
		return &c
	default:
		return p
	}
}

func CloneParts(parts []Part) []Part {
	if parts == nil {
		return nil
	}
	out := make([]Part, len(parts))
	for i, p := range parts {
		out[i] = clonePart(p)
	}
	return out
}

func dropCode(k PartKind) string {
	switch k {
	case KindImage:
		return DropImage
	case KindThinking:
		return DropThinking
	case KindToolUse:
		return DropToolCall
	case KindToolResult:
		return DropToolResult
	default:
		return DropUnknownBlock
	}
}

func Flattened(parts []Part, provider, detail string) Diagnostics {
	var out Diagnostics
	seen := map[PartKind]bool{}
	for _, p := range parts {
		k := p.Kind()
		if k == KindText || seen[k] {
			continue
		}
		seen[k] = true
		out = append(out, Diagnostic{
			Code: dropCode(k), Class: ClassContent, Stage: StageRequest,
			Severity: SeverityLoss, Provider: provider, Field: string(k), Detail: detail,
		})
	}
	return out
}
