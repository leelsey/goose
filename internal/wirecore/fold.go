// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import (
	"encoding/json"
	"sort"

	"github.com/leelsey/goose/llm"
)

const DefaultMaxBuffer = 32 << 20

type Folder struct {
	max      int
	recorder llm.Recorder

	choices map[int]*choice
	order   []int

	stop  llm.StopReason
	block *llm.Block
	err   *llm.Error
	diags llm.Diagnostics
	extra map[string]json.RawMessage
}

type FoldOptions struct {
	MaxBufferBytes int
	Recorder       llm.Recorder
}

type choice struct {
	parts    []llm.Part
	tools    map[int]*llm.ToolUse
	current  int
	nextTool int
	images   map[string]*llm.Image
	openImg  *llm.Image
	role     llm.Role
	stop     llm.StopReason
	stopSeq  string
	bytes    int
	full     bool
}

func NewFolder(opts FoldOptions) *Folder {
	if opts.MaxBufferBytes <= 0 {
		opts.MaxBufferBytes = DefaultMaxBuffer
	}
	return &Folder{max: opts.MaxBufferBytes, recorder: opts.Recorder, choices: map[int]*choice{}}
}

func (f *Folder) note(d llm.Diagnostic) {
	f.diags = append(f.diags, d)
	if f.recorder != nil {
		f.recorder.Note(d)
	}
}

func (f *Folder) Add(fr *llm.Frame) {
	if fr == nil {
		return
	}
	if fr.Err != nil && f.err == nil {
		f.err = fr.Err
	}
	if len(fr.Diagnostics) > 0 {
		f.diags = append(f.diags, fr.Diagnostics...)
	}
	if fr.Block != nil {
		f.block = fr.Block
	}
	if len(fr.Extra) > 0 && f.extra == nil {
		f.extra = fr.Extra
	}
	if fr.Block != nil && len(fr.Delta) == 0 && fr.Choice == nil {
		if fr.Stop != llm.StopNone {
			f.stop = fr.Stop
		}
		return
	}
	if fr.Stop == llm.StopNone && len(fr.Delta) == 0 && fr.Role == "" {
		return
	}

	c := f.choice(fr.ChoiceIndex())
	if fr.Role != "" {
		c.role = fr.Role
	}
	for _, p := range fr.Delta {
		f.merge(c, p)
	}
	if fr.Stop != llm.StopNone {
		c.stop = fr.Stop
		if fr.StopSequence != "" {
			c.stopSeq = fr.StopSequence
		}
	}
}

func (f *Folder) choice(i int) *choice {
	c, ok := f.choices[i]
	if !ok {
		c = &choice{tools: map[int]*llm.ToolUse{}, images: map[string]*llm.Image{}, current: -1, role: llm.RoleAssistant}
		f.choices[i] = c
		f.order = append(f.order, i)
	}
	return c
}

func (f *Folder) grow(c *choice, n int, what string) bool {
	if c.full {
		return false
	}
	if c.bytes+n > f.max {
		c.full = true
		f.note(llm.Diagnostic{
			Code: llm.WireBufferFull, Class: llm.ClassContent, Stage: llm.StageResponse,
			Severity: llm.SeverityLoss, Field: what,
			Detail: "accumulated response exceeded the buffer limit",
		})
		return false
	}
	c.bytes += n
	return true
}

func (f *Folder) merge(c *choice, p llm.Part) {
	switch v := p.(type) {
	case *llm.Text:
		if v.Text == "" || !f.grow(c, len(v.Text), "text") {
			return
		}
		if last, ok := lastOf[*llm.Text](c.parts); ok {
			last.Text += v.Text
			return
		}
		c.parts = append(c.parts, &llm.Text{Text: v.Text})

	case *llm.Thinking:
		f.mergeThinking(c, v)

	case *llm.Refusal:
		if v.Text == "" || !f.grow(c, len(v.Text), "refusal") {
			return
		}
		if last, ok := lastOf[*llm.Refusal](c.parts); ok {
			last.Text += v.Text
			return
		}
		c.parts = append(c.parts, &llm.Refusal{Text: v.Text})

	case *llm.ToolUse:
		f.mergeTool(c, v)

	case *llm.Image:
		f.mergeImage(c, v)

	case *llm.ToolResult:
		c.parts = append(c.parts, v)

	case *llm.Raw:
		c.parts = append(c.parts, v)
	}
}

func (f *Folder) mergeThinking(c *choice, v *llm.Thinking) {
	if len(v.Redacted) > 0 {
		c.parts = append(c.parts, &llm.Thinking{Redacted: append([]byte(nil), v.Redacted...)})
		return
	}
	if v.Text != "" {
		if !f.grow(c, len(v.Text), "thinking") {
			return
		}
		if last, ok := lastOf[*llm.Thinking](c.parts); ok && last.Signature == "" && len(last.Redacted) == 0 {
			last.Text += v.Text
		} else {
			c.parts = append(c.parts, &llm.Thinking{Text: v.Text})
		}
	}
	if v.Signature == "" {
		return
	}
	for i := len(c.parts) - 1; i >= 0; i-- {
		if t, ok := c.parts[i].(*llm.Thinking); ok && t.Signature == "" && len(t.Redacted) == 0 {
			t.Signature = v.Signature
			return
		}
	}
	c.parts = append(c.parts, &llm.Thinking{Signature: v.Signature})
}

func (f *Folder) mergeTool(c *choice, v *llm.ToolUse) {
	var idx int
	switch {
	case v.Index != nil:
		idx = *v.Index
		if idx >= c.nextTool {
			c.nextTool = idx + 1
		}
	case v.ID != "" && c.current >= 0 && c.tools[c.current] != nil && c.tools[c.current].ID == v.ID:
		idx = c.current
	case v.ID != "" || v.Name != "":
		idx = c.nextTool
		c.nextTool++
	case c.current >= 0:
		idx = c.current
	default:
		idx = c.nextTool
		c.nextTool++
	}
	c.current = idx

	t, ok := c.tools[idx]
	if !ok {
		i := idx
		t = &llm.ToolUse{Index: &i}
		c.tools[idx] = t
		c.parts = append(c.parts, t)
	}
	if v.ID != "" {
		t.ID = v.ID
	}
	if v.Name != "" {
		t.Name = v.Name
	}
	if len(v.Args) > 0 && f.grow(c, len(v.Args), "tool_arguments") {
		t.Args = append(t.Args, v.Args...)
	}
}

func (f *Folder) mergeImage(c *choice, v *llm.Image) {
	key := v.ID
	var img *llm.Image
	switch {
	case key != "":
		img = c.images[key]
	case c.openImg != nil:
		img = c.openImg
	}
	if img == nil {
		img = &llm.Image{ID: v.ID, MIME: v.MIME, URL: v.URL, Detail: v.Detail, Prompt: v.Prompt, Index: v.Index}
		c.parts = append(c.parts, img)
		if key != "" {
			c.images[key] = img
		}
		c.openImg = img
	}
	if n := len(v.Data) - len(img.Data); len(v.Data) > 0 && (n <= 0 || f.grow(c, n, "image")) {
		img.Data = append(img.Data[:0:0], v.Data...)
	}
	if v.MIME != "" {
		img.MIME = v.MIME
	}
	if v.Prompt != "" {
		img.Prompt = v.Prompt
	}
	if !v.Partial {
		img.Partial = false
		if c.openImg == img {
			c.openImg = nil
		}
		return
	}
	img.Partial = true
}

func (f *Folder) Response(meta llm.StreamMeta, usage llm.Usage) *llm.Response {
	resp := &llm.Response{
		ID:        meta.ID,
		Provider:  meta.Provider,
		Model:     meta.Model,
		Requested: meta.Requested,
		Created:   meta.Created,
		Session:   meta.Session,
		Usage:     usage,
		Stop:      f.stop,
		Block:     f.block,
		Extra:     f.extra,
	}
	if resp.Extra == nil {
		resp.Extra = meta.Extra
	}
	if len(f.order) == 0 && f.stop == llm.StopNone && f.block == nil && f.err == nil {
		f.note(llm.Diagnostic{
			Code: llm.WireNoFinish, Class: llm.ClassTransport, Stage: llm.StageResponse,
			Severity: llm.SeverityLoss, Provider: meta.Provider,
			Detail: "stream delivered no content and no finish reason",
		})
	}
	sort.Ints(f.order)
	for _, i := range f.order {
		c := f.choices[i]
		stop := c.stop
		if stop == llm.StopNone {
			stop = llm.StopUnknown
			f.note(llm.Diagnostic{
				Code: llm.WireNoFinish, Class: llm.ClassTransport, Stage: llm.StageResponse,
				Severity: llm.SeverityLoss, Provider: meta.Provider,
				Detail: "stream ended without a finish reason",
			})
		}
		role := c.role
		if role == "" {
			role = llm.RoleAssistant
		}
		resp.Choices = append(resp.Choices, llm.Choice{
			Index:        i,
			Message:      llm.Message{Role: role, Content: c.parts},
			Stop:         stop,
			StopSequence: c.stopSeq,
		})
	}
	resp.Diagnostics = append(resp.Diagnostics, f.diags...)
	return resp
}

func (f *Folder) Err() *llm.Error { return f.err }

func Fold(s *llm.Stream, opts FoldOptions) (*llm.Response, error) {
	f := NewFolder(opts)
	for fr := range s.Frames() {
		f.Add(fr)
	}
	resp := f.Response(s.Meta(), s.Usage())
	resp.Diagnostics = append(resp.Diagnostics, s.Diagnostics()...)
	resp.Attempts = append(resp.Attempts, s.Attempts()...)
	err := s.Err()
	if err == nil && f.Err() != nil {
		err = f.Err()
	}
	if cerr := s.Close(); err == nil && cerr != nil {
		err = cerr
	}
	resp.Usage = s.Usage()
	return resp, err
}

func lastOf[T llm.Part](parts []llm.Part) (T, bool) {
	var zero T
	if len(parts) == 0 {
		return zero, false
	}
	v, ok := parts[len(parts)-1].(T)
	return v, ok
}
