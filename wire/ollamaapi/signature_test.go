// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package ollamaapi

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// Neither endpoint has a field for a signature, so the reasoning one stood for
// is gone from the next turn, and that is said on every path.
func TestASignatureThisWireCannotCarryIsReported(t *testing.T) {
	i0 := 0
	parts := []llm.Part{
		&llm.Thinking{Text: "plan", Signature: "EqQB", Signer: llm.FamilyAnthropic},
		&llm.ToolUse{ID: "c1", Name: "f", Args: json.RawMessage(`{}`), Index: &i0, Signature: "SIG-1", Signer: llm.FamilyGoogle},
	}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: parts}, Stop: llm.StopToolUse}}}
	for _, c := range []wire.Codec{NewChat(Options{}), NewGenerate(Options{})} {
		_, diags, err := c.EncodeResponse(resp, wire.Options{})
		if err != nil {
			t.Fatal(err)
		}
		if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 2 {
			t.Errorf("%s: diagnostics = %+v", c.Name(), diags)
		}
		s := llm.StaticStream(llm.StreamMeta{Model: "m"}, []*llm.Frame{{Delta: parts}, {Stop: llm.StopToolUse}})
		diags, err = c.EncodeStream(context.Background(), s, wire.Options{Stream: true}, &wire.BufferWriter{})
		if err != nil {
			t.Fatal(err)
		}
		if d := diags.Find(llm.DropSignature); len(d) != 1 || d[0].Count != 2 {
			t.Errorf("%s stream: diagnostics = %+v", c.Name(), diags)
		}
	}
}
