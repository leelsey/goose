// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package google

import (
	"strconv"

	"github.com/leelsey/goose/llm"
)

// skipSignature is the signature Google documents for a call whose own was not
// kept, such as one another model made or one a client could not hand back.
const skipSignature = "skip_thought_signature_validator"

// currentTurn is where the turn Gemini validates begins: after the caller's
// last message that answers no call. The first call of every step the model has
// taken since must carry its signature, or the request is refused. A message
// that answers a call and says more besides is counted inside the turn, since a
// placeholder where none was needed costs nothing and a missing one fails.
func currentTurn(msgs []llm.Message) int {
	for i := len(msgs) - 1; i >= 0; i-- {
		m := msgs[i]
		if m.Role != llm.RoleAssistant && len(m.Content) > 0 && !llm.HasKind(m.Content, llm.KindToolResult) {
			return i + 1
		}
	}
	return 0
}

// callNames pairs each call with the function it named. Gemini names the
// function a result answers, where the other wires name only the call.
type callNames map[string]string

// fill readies one message's parts for Gemini. A result that names no function
// takes the name of the call it answers. While pending, the next call is the
// first of its step, and one that has lost its signature is sent with Google's
// placeholder. Only what changes is copied, since the parts are the caller's.
func (c callNames) fill(parts []llm.Part, pending *bool) ([]llm.Part, int, error) {
	var out []llm.Part
	set := func(i int, p llm.Part) {
		if out == nil {
			out = append([]llm.Part(nil), parts...)
		}
		out[i] = p
	}
	placeheld := 0
	for i, p := range parts {
		switch v := p.(type) {
		case *llm.ToolUse:
			if v.ID != "" && v.Name != "" {
				c[v.ID] = v.Name
			}
			if pending == nil || !*pending {
				continue
			}
			*pending = false
			if v.Signature != "" {
				continue
			}
			call := *v
			call.Signature, call.Signer = skipSignature, llm.FamilyGoogle
			set(i, &call)
			placeheld++
		case *llm.ToolResult:
			if v.Name != "" {
				continue
			}
			name, ok := c[v.ID]
			if !ok {
				return nil, 0, &llm.Error{
					Kind: llm.KindInvalidRequest, Sent: false,
					Message: "tool result " + strconv.Quote(v.ID) + " answers no call earlier in the conversation, " +
						"and this backend needs the name of the function a result answers",
				}
			}
			result := *v
			result.Name = name
			set(i, &result)
		}
	}
	if out == nil {
		return parts, placeheld, nil
	}
	return out, placeheld, nil
}
