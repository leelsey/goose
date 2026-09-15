// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package agentcli

import (
	"context"
	"os/exec"

	"github.com/leelsey/goose/llm"
)

func (p *Provider) Probe(context.Context) error {
	if _, err := exec.LookPath(p.opts.Binary); err != nil {
		return llm.Errorf(llm.KindNotFound, "%s: %v", p.ID(), err)
	}
	return nil
}

func (p *Provider) Models(context.Context) ([]llm.Model, error) {
	if p.opts.Model == "" {
		return nil, nil
	}
	return []llm.Model{{ID: p.opts.Model, Owner: p.ID()}}, nil
}
