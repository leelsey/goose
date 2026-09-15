// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package llm

import (
	"encoding/json"
	"time"
)

type StopReason string

const (
	StopNone            StopReason = ""
	StopEnd             StopReason = "end"
	StopLength          StopReason = "length"
	StopSequence        StopReason = "stop_sequence"
	StopToolUse         StopReason = "tool_use"
	StopContentFilter   StopReason = "content_filter"
	StopRefusal         StopReason = "refusal"
	StopPause           StopReason = "pause"
	StopContextOverflow StopReason = "context_overflow"
	StopCancelled       StopReason = "cancelled"
	StopError           StopReason = "error"
	StopUnknown         StopReason = "unknown"
)

type Block struct {
	Reason string
	Detail string
}

type Choice struct {
	Index        int
	Message      Message
	Stop         StopReason
	StopSequence string
}

type Response struct {
	ID        string
	Provider  string
	Model     string
	Requested string
	Created   time.Time
	Session   string

	Choices []Choice
	Stop    StopReason
	Block   *Block

	Usage       Usage
	Attempts    []Attempted
	Diagnostics Diagnostics

	Extra map[string]json.RawMessage
}

func (r *Response) Note(d Diagnostic) {
	if r == nil {
		return
	}
	r.Diagnostics = append(r.Diagnostics, d)
}

func (r *Response) GetAttempts() []Attempted {
	if r == nil {
		return nil
	}
	return r.Attempts
}

func (r *Response) First() *Choice {
	if r == nil || len(r.Choices) == 0 {
		return nil
	}
	return &r.Choices[0]
}

type Outcome string

const (
	OutcomeOK        Outcome = "ok"
	OutcomeError     Outcome = "error"
	OutcomeCancelled Outcome = "cancelled"
	OutcomeRejected  Outcome = "rejected"
	OutcomeDiscarded Outcome = "discarded"
)

type Attempted struct {
	Route     string    `json:"route,omitempty"`
	Provider  string    `json:"provider"`
	Model     string    `json:"model,omitempty"`
	Requested string    `json:"requested,omitempty"`
	Index     int       `json:"index"`
	Sent      bool      `json:"sent"`
	Outcome   Outcome   `json:"outcome"`
	Started   time.Time `json:"started"`
	Elapsed   int64     `json:"elapsed_ms"`
	Usage     Usage     `json:"usage"`
	Err       string    `json:"err,omitempty"`
	Detail    string    `json:"detail,omitempty"`
	Status    int       `json:"status,omitempty"`

	Diagnostics Diagnostics `json:"diagnostics,omitempty"`
}

type Frame struct {
	Choice *int
	Role   Role
	Delta  []Part

	Stop         StopReason
	StopSequence string
	Block        *Block

	Usage   *Usage
	Total   bool
	TurnEnd bool

	Err *Error

	ID      string
	Model   string
	Session string

	Diagnostics Diagnostics
	Extra       map[string]json.RawMessage
}

func (f *Frame) ChoiceIndex() int {
	if f == nil || f.Choice == nil {
		return 0
	}
	return *f.Choice
}

func (f *Frame) Empty() bool {
	if f == nil {
		return true
	}
	return len(f.Delta) == 0 && f.Stop == StopNone && f.Usage == nil &&
		f.Err == nil && f.Block == nil && !f.TurnEnd && len(f.Diagnostics) == 0
}
