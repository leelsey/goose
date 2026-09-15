// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package bedrock

import (
	"context"

	"github.com/leelsey/goose/llm"
)

func (p *Provider) Probe(ctx context.Context) error {
	creds, err := p.opts.Credentials(ctx)
	if err != nil {
		return err
	}
	if creds.AccessKeyID == "" || creds.SecretAccessKey == "" {
		return llm.Errorf(llm.KindAuth, "%s: no usable credentials", p.ID())
	}
	return nil
}
