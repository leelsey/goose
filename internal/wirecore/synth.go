// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import "github.com/leelsey/goose/llm"

func Synthesise(resp *llm.Response) *llm.Stream {
	meta := llm.StreamMeta{}
	var frames []*llm.Frame
	if resp != nil {
		meta = llm.StreamMeta{
			ID:        resp.ID,
			Provider:  resp.Provider,
			Model:     resp.Model,
			Requested: resp.Requested,
			Created:   resp.Created,
			Session:   resp.Session,
			Extra:     resp.Extra,
		}
		frames = SynthesiseFrames(resp)
	}
	s := llm.StaticStream(meta, frames)
	for _, a := range resp.GetAttempts() {
		s.Record(a)
	}
	return s
}

func SynthesiseFrames(resp *llm.Response) []*llm.Frame {
	if resp == nil {
		return nil
	}
	var out []*llm.Frame
	for _, c := range resp.Choices {
		idx := c.Index
		role := c.Message.Role
		if role == "" {
			role = llm.RoleAssistant
		}
		out = append(out, &llm.Frame{Choice: &idx, Role: role, ID: resp.ID, Model: resp.Model, Extra: resp.Extra})

		next := 0
		for _, p := range c.Message.Content {
			part := p
			if t, ok := p.(*llm.ToolUse); ok {
				i := next
				if t.Index != nil {
					i = *t.Index
				}
				next = i + 1
				copyTool := *t
				copyTool.Index = &i
				part = &copyTool
			}
			if img, ok := p.(*llm.Image); ok && img.Partial {
				done := *img
				done.Partial = false
				part = &done
			}
			out = append(out, &llm.Frame{Choice: &idx, Delta: []llm.Part{part}})
		}

		fin := &llm.Frame{Choice: &idx, Stop: c.Stop, StopSequence: c.StopSequence}
		if fin.Stop == llm.StopNone {
			fin.Stop = llm.StopEnd
		}
		out = append(out, fin)
	}

	// A block is why the backend stopped, and a candidate having arrived does
	// not make that less true.
	if len(resp.Choices) == 0 && (resp.Block != nil || resp.Stop != llm.StopNone) {
		out = append(out, &llm.Frame{Block: resp.Block, Stop: resp.Stop})
	} else if len(resp.Choices) > 0 && resp.Block != nil {
		out = append(out, &llm.Frame{Block: resp.Block})
	}

	u := resp.Usage
	out = append(out, &llm.Frame{Usage: &u, TurnEnd: true, Diagnostics: resp.Diagnostics})
	return out
}
