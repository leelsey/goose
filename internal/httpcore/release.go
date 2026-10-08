// SPDX-FileCopyrightText: 2026 Leelsey
// SPDX-License-Identifier: MIT

package httpcore

import "context"

type releaseKey struct{}

// WithRelease hands a request the means to give back whatever admitted it, for
// a handler whose answer goes on long after its work is done.
func WithRelease(ctx context.Context, release func()) context.Context {
	return context.WithValue(ctx, releaseKey{}, release)
}

// Release gives back whatever admitted the request, if anything did. The one
// that admitted it gives it back again when the request ends, so what it hands
// over has to be safe to call twice.
func Release(ctx context.Context) {
	if release, ok := ctx.Value(releaseKey{}).(func()); ok {
		release()
	}
}
