// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package geminiapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/leelsey/goose/llm"
	"github.com/leelsey/goose/wire"
)

// Gemini signs the first part after its thoughts, whatever kind it is. One this
// wire does not model is carried as it came, signature and all, rather than
// read as a text that has nothing in it.
func TestASignedPartThatIsNotTextIsCarriedWhole(t *testing.T) {
	code := `{"executableCode":{"code":"print(6*7)","language":"PYTHON"},"thoughtSignature":"CODE-SIG"}`
	parts, _, err := DecodeParts([]json.RawMessage{json.RawMessage(code), json.RawMessage(`{"text":"","thoughtSignature":"SIG-2"}`)})
	if err != nil {
		t.Fatal(err)
	}
	if raw, ok := parts[0].(*llm.Raw); !ok || string(raw.JSON) != code {
		t.Fatalf("part = %+v: the code it ran is the part", parts[0])
	}
	if tx, ok := parts[1].(*llm.Text); !ok || tx.Signature != "SIG-2" {
		t.Fatalf("part = %+v: a part with nothing but a signature is still a text", parts[1])
	}
	out, _ := EncodeParts(parts)
	if string(out[0]) != code {
		t.Fatalf("replayed = %s: the backend gets back the part it sent", out[0])
	}

	body := `{"contents":[{"role":"user","parts":[{"text":"run it"}]},{"role":"model","parts":[` + code + `]}]}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	if raw, ok := r.Messages[1].Content[0].(*llm.Raw); !ok || string(raw.JSON) != code {
		t.Fatalf("client part = %+v: a client replaying it is heard the same", r.Messages[1].Content[0])
	}
}

// Gemini's image models sign each image they make, and an edit of one on the
// next turn is refused without that signature.
func TestAGeneratedImageKeepsItsSignature(t *testing.T) {
	image := `{"inlineData":{"data":"iVBORw0KGgo=","mimeType":"image/png"},"thoughtSignature":"IMG-SIG"}`
	parts, _, err := DecodeParts([]json.RawMessage{json.RawMessage(image)})
	if err != nil {
		t.Fatal(err)
	}
	img, ok := parts[0].(*llm.Image)
	if !ok || img.Signature != "IMG-SIG" || img.Signer != llm.FamilyGoogle {
		t.Fatalf("image = %+v", parts[0])
	}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: parts}, Stop: llm.StopEnd}}}
	out, diags, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), image) || len(diags) != 0 {
		t.Fatalf("answer = %s, diagnostics = %+v: the client is handed the signature", out.Body, diags)
	}

	body := `{"contents":[{"role":"user","parts":[{"text":"draw"}]},{"role":"model","parts":[` + image + `]},
		{"role":"user","parts":[{"text":"make it blue"}]}]}`
	r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
	if err != nil {
		t.Fatal(err)
	}
	back, _ := EncodeParts(r.Messages[1].Content)
	if string(back[0]) != image {
		t.Fatalf("replayed = %s: the edit reaches the backend with the image's signature", back[0])
	}
}

// A reasoning item OpenAI stored by name alone reaches this wire with nothing
// in it, and a thought part with nothing in it is not one to hand over.
func TestAThoughtWithNothingInItIsNotAPart(t *testing.T) {
	stored := &llm.Thinking{ID: "rs_stored", Signer: llm.FamilyOpenAI}
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
		stored, &llm.Text{Text: "hi"},
	}}, Stop: llm.StopEnd}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{{Delta: []llm.Part{stored}}, {Delta: []llm.Part{&llm.Text{Text: "hi"}}, Stop: llm.StopEnd}})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	for _, body := range []string{string(out.Body), w.Body.String()} {
		if strings.Contains(body, `"thought"`) {
			t.Fatalf("answer = %s", body)
		}
	}
	if n := strings.Count(w.Body.String(), "data: "); n != 2 {
		t.Fatalf("stream = %s: a frame that carried nothing is not a chunk", w.Body.String())
	}
}

// A client on this wire has no field for who minted a signature, so one from
// another family goes out marked as that family's and comes back attributed.
// The field is bytes, which typed SDKs decode from base64 and hand back in
// either alphabet, so the mark is base64 too. The backend is sent a signature
// exactly as it was issued.
func TestAnotherFamilysSignatureIsMarkedForTheClient(t *testing.T) {
	const claude = "EqQBCkgIARABGAIiQPla+/8="
	resp := &llm.Response{Choices: []llm.Choice{{Message: llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{
		&llm.Thinking{Text: "plan", Signature: claude, Signer: llm.FamilyAnthropic},
		&llm.Text{Text: "ok", Signature: "CiQB", Signer: llm.FamilyGoogle},
	}}, Stop: llm.StopEnd}}}
	out, _, err := New(Options{}).EncodeResponse(resp, wire.Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(out.Body), `"thoughtSignature":"CiQB"`) {
		t.Fatalf("answer = %s: Google's own signature goes out as Google issued it", out.Body)
	}
	type typed struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text             string `json:"text"`
					ThoughtSignature []byte `json:"thoughtSignature"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
	}
	var answer typed
	if err := json.Unmarshal(out.Body, &answer); err != nil {
		t.Fatalf("answer = %s: a typed client cannot read it: %v", out.Body, err)
	}
	sent := answer.Candidates[0].Content.Parts

	s := llm.StaticStream(llm.StreamMeta{}, []*llm.Frame{{Delta: resp.Choices[0].Message.Content[:1], Stop: llm.StopEnd}})
	w := &wire.BufferWriter{}
	if _, err := New(Options{}).EncodeStream(context.Background(), s, wire.Options{Stream: true}, w); err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(w.Body.String(), "\n") {
		if chunk, ok := strings.CutPrefix(line, "data: "); ok {
			if err := json.Unmarshal([]byte(chunk), &typed{}); err != nil {
				t.Fatalf("chunk = %s: a typed client cannot read it: %v", chunk, err)
			}
		}
	}

	for name, spell := range map[string]func([]byte) string{
		"standard": base64.StdEncoding.EncodeToString, "url-safe": base64.RawURLEncoding.EncodeToString,
	} {
		turn, _ := json.Marshal([]map[string]any{
			{"text": "plan", "thought": true, "thoughtSignature": spell(sent[0].ThoughtSignature)},
			{"text": "ok", "thoughtSignature": spell(sent[1].ThoughtSignature)},
		})
		body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]},{"role":"model","parts":` + string(turn) + `}]}`
		r, _, _, err := New(Options{}).Decode(&wire.Inbound{Path: "/v1beta/models/g:generateContent", Body: []byte(body)})
		if err != nil {
			t.Fatal(err)
		}
		th := r.Messages[1].Content[0].(*llm.Thinking)
		tx := r.Messages[1].Content[1].(*llm.Text)
		if th.Signer != llm.FamilyAnthropic || th.Signature != claude || tx.Signer != llm.FamilyGoogle || tx.Signature != "CiQB" {
			t.Fatalf("%s: thinking = %+v, text = %+v", name, th, tx)
		}
	}
	back, _ := EncodeParts([]llm.Part{&llm.Text{Text: "x", Signature: "goose.v1.anthropic.EqQB", Signer: llm.FamilyGoogle}})
	if !strings.Contains(string(back[0]), `"thoughtSignature":"goose.v1.anthropic.EqQB"`) {
		t.Fatalf("to the backend = %s: what it issued goes back as it was", back[0])
	}
}
