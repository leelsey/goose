// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package wirecore

import "github.com/leelsey/goose/llm"

// Synthesise serves a whole answer as a stream. The answer is in hand before
// any of it is delivered, so what it costs, what it noted, and what to cannot
// carry are all weighed against a policy before the first frame rather than
// one frame at a time: strict either hands over the whole synthesised stream
// or refuses it outright, never part of it, and a refusal that ends the
// stream before its last frame does not lose what the answer cost.
func Synthesise(resp *llm.Response, to llm.Renders) *llm.Stream {
	meta := llm.StreamMeta{}
	var frames []*llm.Frame
	var diags llm.Diagnostics
	var usage llm.Usage
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
		diags = append(append(llm.Diagnostics{}, resp.Diagnostics...), to.Unrendered(resp)...)
		usage = turnUsage(resp.Usage)
	}
	s := llm.StaticStream(meta, frames)
	// The answer's usage travels on its last frame, which a refusal part way
	// through never delivers. Settled now, from the same answer, the call is
	// still booked at what it cost rather than at what a backend "reported no
	// usage" for, whichever frame the stream ends on.
	s.Settle(usage)
	for _, d := range diags {
		s.Note(d)
	}
	for _, a := range resp.GetAttempts() {
		s.Record(a)
	}
	return s
}

// turnUsage is what absorbing the answer's one usage frame and closing the
// turn it ends would leave banked, for a stream that may never deliver that
// frame to absorb it itself. A usage carrying nothing is left as it is: a
// turn nobody counted is not one to claim.
func turnUsage(u llm.Usage) llm.Usage {
	turn := llm.Usage{}.Merge(u)
	if turn.Empty() {
		return llm.Usage{}
	}
	if turn.Turns == 0 {
		turn.Turns = 1
	}
	return llm.Usage{}.Add(turn)
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
	// not make that less true. An answer with no candidate came back whole all
	// the same, so it says why it ended as a candidate's last frame does, or
	// the stream reads it as cut off.
	switch {
	case len(resp.Choices) == 0:
		stop := resp.Stop
		switch {
		case stop != llm.StopNone:
		case resp.Block != nil:
			stop = llm.StopContentFilter
		default:
			stop = llm.StopEnd
		}
		out = append(out, &llm.Frame{Block: resp.Block, Stop: stop})
	case resp.Block != nil:
		out = append(out, &llm.Frame{Block: resp.Block})
	}

	u := resp.Usage
	out = append(out, &llm.Frame{Usage: &u, TurnEnd: true})
	return out
}
