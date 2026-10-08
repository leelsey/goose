# Goose: Proxy & Provider — architecture

Goose sits between a program that speaks one LLM wire format and an upstream that speaks another. It
runs two ways from one codebase: as an HTTP proxy binary, and as a Go library another project imports
and calls in-process with no HTTP anywhere.

## The four goals

1. **Embeddable.** Public packages, no package-level mutable state, no environment reads, no writes to
   a global logger, no `os.Exit` below `cmd/`. The `http.Client`, logger, clock and meter are injected.
2. **Routed.** Many providers registered at once; each request routed by model pattern, alias, client
   format, header or caller logic, with ordered fallback chains.
3. **Measured.** Usage is a value returned on every call, unary and streamed, attributed to the
   provider and model that actually served it, correct across retries and chains.
4. **Lossless.** Nothing the model produced is dropped without the caller being told. This is the
   headline requirement and it drives most of what follows.

## Losslessness is a type problem, not a discipline problem

A codec layered on ad hoc structs drops content in at least eight ways, each a real bug: refusals
destroyed in both directions of the stream/unary adapters; parallel tool calls merged into one when
fragments carry no index; a stream that ends without a finish reason discarding its accumulated usage;
malformed upstream frames skipped with no error and no counter; a 200 response carrying an error body
decoded as an empty success; multi-choice responses collapsed to one; non-text MCP tool results
destroyed; system prompts silently discarded for three of four CLI providers.

None of these are fixed by care. They are fixed by making the lossy shape unrepresentable:

- **Content is an ordered list of typed parts.** `Message.Content []Part` is a closed sum with pointer
  receivers. Text, thinking with its own signature, redacted thinking, tool use, tool result, image and
  refusal keep their source order inside a turn, and several thinking blocks keep several signatures.
  Flattening this to `Thinking string` + `Signature string` + a separate `ToolCalls` slice loses
  interleaving and corrupts multi-block thinking on replay.
- **Anything unmodelled round-trips verbatim.** `Raw` is a part; `Request.Extra` holds unmodelled
  request fields. A field goose does not understand survives, or is reported as dropped.
- **Absent is not zero.** Every index that a vendor may omit — choice, content block, tool-call index —
  is `*int`. A bare `int` conflates "the upstream said 0" with "the upstream said nothing", which is
  exactly how parallel tool calls merge into one.
- **A drop is a value.** Every seam returns `[]Diagnostic` with a stable code. The site that emits one
  chooses its severity, because only it knows whether output was destroyed or a field merely re-spelled;
  a loss marked fatal there fails the call under any policy. `LossPolicy` then raises or lowers by code,
  and `Strict` promotes every content loss to fatal, whatever severity its site chose — which is how a
  caller asks goose to refuse to be lossy at all. The one note it leaves alone is the round-trip tier:
  a `drop.unknown_block` at info says a block was carried through whole, and a site that could not
  keep one reports it at loss. Whether the destination can write such a block out is a separate
  question, answered by `Renders.Raw`: `Renders.Unrendered` reports a carried block the destination
  has no shape for as a loss, as it does reasoning, an image or a tool call the destination declares
  no field for, and `Strict` refuses it. A stream answers to the same policy, for the same
  destination, through `Stream.Enforce`, which weighs those parts before any of the frame carrying
  them is delivered, and an encoder holds to it what it drops in writing the stream out through
  `Stream.Check`. Policy is keyed by code, so modelling a new loss later cannot change what an
  existing caller already accepts.
- **Failure carries its evidence.** `Complete` may return a non-nil `*Response` beside a non-nil error,
  and a failed `*Stream` still answers `Usage()`, `Err()` and whatever frames it delivered. The upstream
  has already been paid; a cancelled call is recorded, not forgotten.

## Decisions

### A. The codec seam is bytes in, bytes out

`wire.Codec` decodes an `Inbound{Method, Path, Query, Header, Body []byte}` and encodes an `Outbound`.
No `*http.Request`, no `http.ResponseWriter`, no server type crosses it, so a format is unit-testable
from a byte slice and an embedder can decode a request it never received over a socket. The query
string is part of the input because Gemini selects its transport with `?alt=sse` and carries a
credential in `?key=`. Every codec method also returns `[]Diagnostic`, including `Decode`: a request
whose translation lost something must say so before the call is made, not after it is billed. What
`Decode` loses reaches the call through `goose.WithDiagnostics` and is held to the policy before any
link is tried. What `EncodeResponse` loses is held to `Response.Policy` before the answer is written.
What `EncodeStream` drops it holds to the stream's policy through `Stream.Check` as it goes, and a
codec written outside goose must call `Stream.Check` too if it is to be held to the policy.

### B. Streams pull, and commit exactly once

`*llm.Stream` is a concrete struct, not an interface, so frame folding and usage accumulation have one
implementation that nothing can bypass. It offers `Frames() iter.Seq[*Frame]`, `Err()`, `Usage()`,
`Meta()`, `Diagnostics()` and `Close()` — the `sql.Rows` shape. Iteration is consumer-paced with no
goroutine and no channel; a `break` is honoured and the accounting survives it.

`Meta()` is readable before the first frame because a wire format must render an id, a model and a
created time on its opening event, and because a session id has to reach a response header that is
written before any frame.

The commit point is structural: the router holds what arrives before the first frame of content — a
backend's opening, its counts — and that frame, before it returns the stream, so the commit point is
the first content delivered. A failure before it, in-band or not, is one before anything was written.
Failing over is possible while nothing has been written and impossible afterwards, because after
that a `*Stream` exists and there is no API that re-routes one.

### C. Providers declare capabilities; nothing switches on a name

`Provider` is `ID`, `Capabilities`, `Complete`, `Stream`. Everything else is an optional interface:
`Closer` (a CLI provider owns child process groups and 0600 temp files and must be able to reap them),
`Binder` (receives the injected environment), `Passer` (opaque passthrough), `Gated` (per-provider
concurrency), `Precheck` (what a request will lose, or that it will not be sent, found before the
gate), `Lister` (model enumeration).

`Capabilities` replaces hardcoded name-string switches: whether streamed usage exists, which
output-ceiling field the backend accepts, how many cache breakpoints it will carry, whether it tolerates
extension fields, which vendor family's signatures it verifies, whether it honours a session id,
whether it can take a system prompt at all. A capability is consulted; a name never is. A signature
records the family that minted it — Claude wherever it runs, OpenAI with Azure OpenAI, Gemini
wherever it runs — so one bound for a backend of another family is stripped before the call rather
than refused by it, and a client's wire, which has no field for who minted a signature, is handed a
foreign one marked with its family.

### D. Routes are data, with one function escape hatch

A `Rule` has a name, a `Matcher` (model glob, client format, header, path — composable with `All`,
`Any`, `Not`), an optional model rewrite, an ordered chain of targets, a stickiness policy, a retry
policy, a tool group and per-route limits. A caller-supplied `Selector` func is consulted first when
set. A model glob with no separator in it is matched against the whole name and then against each tail
beginning after a `/`, `.` or `:`, because names arrive route- and vendor-qualified
(`publishers/anthropic/models/claude-sonnet-4`, `us.anthropic.claude-sonnet-4-v1:0`) and a rule
written as `claude-*` has to reach them. The tail match is anchored, and a pattern that spells a
separator is taken as asking about the whole name, so `gpt-4.1` still matches only itself.

Stickiness matters because a session id is provider-bound: a `previous_response_id` minted by one
provider is meaningless to another, so a request carrying one pins to the provider that issued it.

A chain is walked in the operator's order, with one exception: a request that offers the model tools,
and does not forbid it to call one, goes first to the links whose capabilities carry every one of
them, and a link that would have to strip some is reached only once those have failed, been skipped
or been refused. An answer without the tools answers a different question. Each attempt keeps its
link's place in the route, so the ledger's rows follow the order tried and still say where each link
stood. The breaker a chain consults is shared by every caller, so it counts only what a backend did:
a failure the caller brought with it — a malformed or oversized request, a model the caller named
that the backend does not have, a refused key the backend was handed on the caller's behalf — gives
the probe slot back without counting. An attempt's deadline is fixed before it waits for a slot at a
gated backend, so the wait and the send share one budget, and a send that runs out of what the wait
left it is not held against the backend. A stream is judged when it ends rather than when it is
handed over, since a backend can fail after its first content as surely as before it, and a probe
that streams holds the probe slot until then. When every link fails, the caller is given the
failure that reached a backend over a refusal made before anything was sent.

### E. Usage is one carrier, three combinators

`llm.Usage` carries counts, cache read/write with the five-minute and one-hour split where the backend
measured one, reasoning, cost, service tier, speed, turn count, duration, the `Reported` flag and any
anomalies — and every field serialises. Marking half of them `json:"-"` is a trap: any hop that
re-encodes usage then loses the price and turns "measured" into "unmeasured".

The convention is fixed and every adapter normalises into it: **`Prompt` includes both cache figures;
`Completion` includes `Reasoning`.** Vendors disagree — Anthropic and Bedrock report cache beside the
prompt count, Gemini reports thinking beside the completion count — so `Convention` describes the
upstream shape and normalisation happens once, in the adapter, before anything aggregates.

Three combinators, because there are three distinct situations and conflating them is a common source
of bugs:

- `Merge` — frames inside one turn are complementary. Newest non-zero per field, `Reported` OR-ed.
  Summing multiplies the bill by the frame count; taking the last frame erases the cache breakdown.
- `Add` — turns are sequential billed calls. Counts, cost and duration sum.
- `Cover` — a per-call breakdown against a summary that disagrees: component-wise maximum, never
  all-or-nothing, because neither source can be wrong low about work it reports.

`Frame.TurnEnd` marks the boundary, so one CLI invocation running several billed turns accounts
correctly without the stream having to guess.

Attribution is per attempt. A chain that tried two providers produces two `Attempt` records with their
own provider, model, outcome and usage, plus the aggregate. A discarded retry that reached the backend
is recorded; a failure that never left the process is not.

### F. The request is immutable, and containment is structural

`llm.Request` is owned by the caller and never written through. The router derives a fresh per-attempt
copy carrying the resolved model, the route's rendered tools and that attempt's credential. Even a
third-party provider that mutates what it is handed cannot poison the next attempt, because the next
attempt is a different copy. This closes off a common structural defect: an adapter that writes
`req.Model = provider.model` directly onto the shared request makes a fallback chain silently wrong.

An adapter notes what building its request loses through `Attempt.Note` and calls `Attempt.Check`
before it sends, so a loss the policy refuses costs nothing and the chain walks on.

### G. One core package, thin layers around it

```
goose            facade: Client, options, Register, Chat, Stream, Handler
goose/llm        neutral core: Request, Message, Part, Response, Frame, Stream, Usage,
                 Diagnostic, Error, Provider, Attempt, Capabilities. No I/O, no globals.
goose/wire       Codec seam + Inbound/Outbound
goose/wire/*     openaichat, openairesp, anthropicmsg, geminiapi, ollamaapi
goose/provider/* anthropic, openai, google, ollama, azure, cloudflare, openrouter,
                 bedrock, vertex, agentcli
goose/route      Rule, Matcher, Table, Selector
goose/usage      sinks and aggregation (memory, JSONL, func)
goose/mcp        tool servers: stdio, SSE, streamable HTTP, both protocol eras;
                 the protocol served back over HTTP and over a pipe, with
                 goose's own tools on it
goose/httpx      the proxy: mux, SSE/NDJSON writer, auth, rate limiting, the
                 concurrency gate, CORS and the browser origin and host checks,
                 health, models, metrics, passthrough
goose/internal/httpcore  the retry loop, the client whose timeout does not kill a stream
goose/internal/* scanners, frame folding, subprocess lifetime, cloud auth, the AWS
                 event stream, globs
cmd/goose        the binary: flags, config files, environment, signals, stderr
```

Diagnostics and usage live in `llm` rather than in packages of their own. They are referenced by every
layer, and giving them their own package while they must also carry `llm` payloads is how the design
panel's observability proposal acquired an import cycle. The graph above is acyclic.

### H. Three tiers of loss, all values

- **Reject** — the request cannot be expressed at all: a typed `*llm.Error` before any spend.
- **Report** — the translation proceeds and something is lost: a `Diagnostic` on the returned value and
  on the ledger row, at the severity the emitting site chose. `LossPolicy{Strict: true}` turns every
  content loss into a refusal instead, on a stream as on an unstreamed answer, and one known before
  the request is sent, whether once it is planned, while the proxy decodes it, or while the adapter
  builds it, is refused before it is sent, and the chain walks on.
- **Round-trip** — goose does not understand it but does not need to: `Raw` parts and `Extra` fields
  carry it through untouched. Nothing was lost in carrying it, so `Strict` does not refuse it where the
  destination can write it out. A destination that cannot, as its `Renders.Raw` says, loses it, and
  `Strict` refuses that.

Nothing is logged from library code. The injected `*slog.Logger` is optional and additive; it is never
the only place a fact appears.

## Implementation order

Each step is independently testable and lands with its own contract tests.

1. `llm` — types, usage combinators, diagnostics, stream, errors, capabilities. **Done.**
2. `internal/wirecore` — frame folding, the stream/unary adapters in both directions, bounded SSE and
   NDJSON scanners that surface a malformed frame as an error. **Done.**
3. `wire/openaichat` end to end against a stub provider, then the other four codecs. **Done.**
4. `provider/anthropic` and `provider/openai`, then google and ollama. **Done.**
5. `route` and the facade, including chains, stickiness and the commit point. **Done.**
6. `usage` sinks and the ledger. **Done.**
7. Cloud adapters, then the CLI providers. **Done.**
8. `mcp`. **Done.**
9. `httpx` and `cmd/goose`. **Done.**
