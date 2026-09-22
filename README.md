# Goose: Proxy & Provider

Goose sits between a program that speaks one LLM wire format and whatever upstream actually holds the
model. It runs two ways from one codebase.

- **Embedded.** A Go program imports it and calls `Chat` or `Stream` directly. No HTTP anywhere.
- **Proxy.** The same client is mounted behind the wire formats a tool already speaks, so an agent is
  pointed at a base URL and needs no change.

```
OpenAI client  ─┐                     ┌─→ Anthropic
Anthropic tool ─┼─→ goose ─ route ─┬──┼─→ OpenAI
Gemini SDK     ─┤    convert       │  ├─→ Google
Ollama app     ─┘                  │  └─→ Ollama, or anything speaking their shapes
                                   └─ fallback chain, per-attempt usage
```

## Formats and providers

| Client format      | Endpoint                                                        | Streaming | Model list           |
|--------------------|-----------------------------------------------------------------|-----------|----------------------|
| OpenAI             | `POST /v1/chat/completions`                                     | SSE       | `GET /v1/models`     |
| OpenAI (Responses) | `POST /v1/responses`                                            | SSE       | `GET /v1/models`     |
| Anthropic          | `POST /v1/messages`                                             | SSE       | `GET /v1/models`     |
| Google             | `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent` | SSE | `GET /v1beta/models` |
| Ollama             | `POST /api/chat`, `POST /api/generate`                          | NDJSON    | `GET /api/tags`      |

A client populates its picker from the list endpoint its own wire defines, so each is served in that
wire's shape. `/v1/models` is claimed by two wires at once and cannot be two shapes, so every entry
there carries both spellings of the same facts and each client reads the keys it knows.

| Upstream kind | Serves | Credential |
|---|---|---|
| `anthropic` | Anthropic Messages | `x-api-key` |
| `openai` | OpenAI chat completions and Responses | bearer |
| `google` | Gemini generateContent | `x-goog-api-key` |
| `ollama` | a local or remote Ollama | optional bearer |
| `azure` | Azure AI Foundry, rolling or dated surface | `api-key`, or an Entra bearer |
| `cloudflare` | Workers AI | bearer |
| `openrouter` | OpenRouter, with optional vendor pinning | bearer |
| `bedrock` | AWS Bedrock through the Converse API | SigV4, from a key, a profile or the runtime's own |
| `vertex` | GCP Vertex: Gemini, Claude, Llama, Mistral | OAuth2 bearer |
| `compatible` | anything else speaking the chat-completions shape | bearer or a named header |
| `claude`, `codex`, `antigravity`, `copilot` | a local agent CLI, on its own subscription login | the binary's own |

The Entra bearer an `azure` provider can use is minted per request, because these tokens expire
mid-session, so it comes from an embedder through `azure.Options.Token` rather than from the file; a
configured `azure` provider authenticates with its `api-key`.

`compatible` covers vLLM, LM Studio, Together, Groq, Fireworks and the rest. It assumes nothing about
extensions it has not been told about: declare what a backend honours with `dialect`, and goose reports
a streamed call as unmeasurable rather than booking it as free.

AWS signing and the Google token exchange are implemented here against their published specifications
rather than pulled in as vendor SDKs, so the module stays standard library only.

A CLI backend runs the binary as a child process in its own process group, with a scrubbed environment
that does not include this proxy's own key. It reports a real price per call and how many turns it
billed, which no HTTP backend does. The prompt always travels on stdin, never argv, which any local
user can read. The system prompt travels on the CLI's own cached channel where one exists, in a
directory only this process can open; only one of the four binaries has such a channel, so for the
rest goose folds the prompt into the conversation and says so, because folded it is billed as fresh
input on every call. A binary whose own tools goose cannot turn off says so on every call, and a
sandbox setting a binary does not honour is refused at construction rather than accepted and ignored.

Any format can be served from any provider. An Anthropic-format client streaming from an Ollama
upstream produces correct `message_start` / `content_block_*` / `message_delta` sequencing, with the
token counts folded into the Anthropic spelling.

## Using it as a library

```go
claude, err := anthropic.New(anthropic.Options{APIKey: key})
gpt, err := openai.New(openai.Options{APIKey: other})

c, err := goose.New(
    goose.WithProvider(claude),
    goose.WithProvider(gpt),
    goose.WithRoutes(route.Rule{
        Name:  "claude",
        Match: route.Model("claude-*"),
        Chain: []route.Target{{Provider: "anthropic"}, {Provider: "openai", Model: "gpt-5"}},
    }),
    goose.WithUsage(meter),
)

resp, err := c.Chat(ctx, &llm.Request{Model: "claude-sonnet-5", Messages: msgs})
fmt.Println(resp.Usage.Prompt, resp.Usage.Completion, resp.Attempts)
```

Streaming pulls, so backpressure is the consumer's:

```go
s, err := c.Stream(ctx, req)
defer s.Close()
for f := range s.Frames() {
    for _, p := range f.Delta { ... }
}
fmt.Println(s.Usage(), s.Err())
```

The client is safe to share: build one, keep it for the life of the process and
call it from as many goroutines as you like. It is the breaker's memory of what
has been failing, so one rebuilt per request forgets that.

Nothing here reads your environment or opens a file for you. The two backends
that authenticate with something other than a key take a source you supply, and
the usual ones are in the package rather than behind `internal/`, so the binary
and an embedder resolve credentials the same way:

```go
bedrock.New(bedrock.Options{Region: "eu-west-1", Credentials: bedrock.FirstWorkingCredentials(
    bedrock.StaticCredentials(bedrock.Credentials{AccessKeyID: id, SecretAccessKey: secret}),
    bedrock.InstanceCredentials(nil, ""),
)})

token, project, err := vertex.ServiceAccountToken(keyJSON, nil)
vertex.New(vertex.Options{Project: project, Region: "europe-west1", Token: token})
```

## Running the proxy

```sh
goose -p ollama -m llama3.1:8b
goose -c goose.toml
```

```toml
formats = ["openai", "anthropic", "ollama"]

[server]
host = "127.0.0.1"
port = 11234

[[provider]]
id = "claude"
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[[provider]]
id = "bedrock"
kind = "bedrock"
options = { region = "us-east-1", profile = "work" }

[[provider]]
id = "local"
kind = "ollama"
base_url = "http://127.0.0.1:11434"

[[route]]
name = "claude"
models = ["claude-*"]
chain = ["claude", "bedrock", "local"]

[[route]]
name = "rest"
chain = ["local"]

[usage]
log = "usage.jsonl"
summary = true
```

[`goose.example.toml`](goose.example.toml) is the same file with every key annotated.

A chain walks on when a link fails for a reason the next one might not share: a refused or expired
credential, a model that backend does not serve, a rate limit, an outage. It stops where walking on
would change nothing — a request no backend can read, a capability none of them has, a caller that
has gone, or its own attempt deadline having passed, since the caller has already waited that whole
budget and the next link would be given its own. Bear in
mind that a key which has been revoked on the first link is a reason to reach the second, so a chain
is also a way to keep spending after a backend has started refusing.

A provider takes `headers = { ... }` for a gateway in front of it that wants one; the credential is
set after them, so a header named there cannot replace it.

`server.metrics` serves `GET /metrics` for a scraper; `server.max_concurrent` and
`server.max_per_minute` bound what callers can take. The allowance is keyed on what a caller has
proved rather than on what it claims, since a caller writes its own headers: with `server.api_key`
set, everyone holding it shares one allowance, and everyone who does not share the allowance of the
connection they arrived on; with no key set, each upstream credential carries its own. `server.tls_cert` and `server.tls_key` serve HTTPS: off loopback a plaintext listener hands
every credential to anything on the path, and the binary says so at startup.

`goose -v` logs one line per request with what it cost and what the translation lost.
`goose -check` validates the file and prints what would be served, spawning nothing. An unknown key
is an error, not a shrug: in TOML a root key written below a table header belongs to that table, so
a setting can look applied and not be. That holds inside `options` too, which is a free map and
would otherwise take a misspelt key without a word; the error names what the kind does read.

## Tool servers

Model Context Protocol servers connect over stdio, streamable HTTP or the older SSE transport. Their
tools are advertised to the model on the routes that ask for them, and the proxy serves the protocol
back at `/mcp` so the client executes them. Running them inside the proxy would run the caller's tools
in the proxy's process.

### Tool servers goose connects to

```toml
[[mcp]]
name = "files"
command = "mcp-server-filesystem"
args = ["/srv/data"]

[[route]]
name = "everything"
chain = ["claude"]
tools = ["mcp"]
```

The tool list is read per request, so a server that gains or loses tools is not misrepresented. Names
that collide between servers are qualified on both sides rather than one winning silently, the list
cursor is followed to the end, and a result that arrives as an image or audio reaches the caller as
one. An unauthenticated tool endpoint is refused on anything but a loopback address, and the refusal
happens before any server is started. A request carrying a browser `Origin` is refused outright: this
endpoint runs the operator's tools, a proxy usually listens on loopback, and without the check any
page the operator visits can point a name of its own at `127.0.0.1` and drive them.

### goose as a tool server

`server.mcp_tools` puts goose's own tools on the same endpoint, so an agent that speaks this protocol
and no vendor wire still reaches every provider, route and fallback chain the file configures.

```toml
[server]
mcp_tools = true
```

| Tool | Takes | Gives |
|---|---|---|
| `ask` | `prompt`, and optionally `model`, `system`, `max_tokens`, `temperature` | the answer, with the model, the provider and the bill beside it |
| `models` | nothing | every model the proxy can route to |

It is off by default, because an endpoint that runs prompts is a larger thing to leave open than one
that lists tools; the loopback and API-key rules above cover it either way. A backend that refuses
comes back as a failed tool call carrying the reason, not as a protocol error the agent cannot read.

Most agents spawn a command and talk to its pipes rather than opening a socket. `goose mcp` serves the
same protocol on stdin and stdout for them — the same configuration, the same routes, the same tool
servers, with goose's own tools always on:

```json
{ "mcpServers": { "goose": { "command": "goose", "args": ["mcp", "-c", "/path/to/goose.toml"] } } }
```

Everything goose has to say in that mode goes to stderr, because a line of its own among the frames is
a parse error at the other end. A cancellation withdraws the call it names — a pipe has no stream to
close per request — and the withdrawn call is answered with nothing at all.

Both halves speak both eras of the protocol. `2026-07-28` replaced the `initialize` handshake with
per-request metadata, a `server/discover` call and no session at all; the revisions before it,
`2024-11-05` through `2025-11-25`, open a session and remember what was agreed. goose serves a client
of either era on the same endpoint and the same pipe, and dials a server of either: it asks
`server/discover` first, and a server that refuses it — or says nothing, which some do — gets the
handshake instead, once, with a deadline of its own rather than the whole call timeout.

On the endpoint the current revision's rules are followed as written: the `MCP-Protocol-Version`,
`Mcp-Method` and `Mcp-Name` headers must match the body they mirror or the request is refused with
`400` and `-32020`, a version goose does not speak comes back as `400` and `-32022` naming the ones it
does, a method it does not have is `404` and `-32601` rather than a bare `404`, a browser `Origin` is
`403`, and `GET` or `DELETE` on the endpoint is `405` — the older era's standalone stream and session
were removed with the handshake. A tool result that asks the client for something before it can finish
is reported as such rather than handed back as an answer: goose sits between two ends that are not
there and can sample no model and ask no user on their behalf.

A client that wants to know when the set of tools moves opens a stream with `subscriptions/listen`
and is told; goose re-reads its servers on a timer and is the only one that knows, so a client that
listed once at startup would otherwise never learn of a tool that appeared after. A client of the
older era is told on the channel it already has, and only that era is — a client of the newer one is
sent nothing it did not ask for. `listChanged` is claimed only where the notification has a way to
arrive.

A call that takes minutes says so while it takes them. A request carrying a `progressToken` is
answered on a stream: what the tool reports about its progress first, the answer last. Reports from a
tool server upstream are relayed rather than read as noise, under the token the caller chose, and
`ask` reports the answer forming as the model writes it — a minute of silence and a minute of visible
work are the same wait and not the same experience.

A tool that asks for some of its arguments to be copied into headers — the `x-mcp-header` annotation,
which lets an intermediary route or refuse a call without reading the body — has them copied on the
way out and checked against the body on the way in; a definition that asks for something the rule does
not allow is left out of the list rather than called incorrectly, and the startup banner names what it
left out. A tool server's own definition crosses the proxy whole: its title, its icons, its
annotations and whatever a later revision adds reach the client that displays them, under the name
goose advertises it by.

### Tool servers a backend dials itself

Two vendors let the request name a tool server that the *backend* connects to: it lists the tools,
calls them and answers, inside the completion. OpenAI spells this as an `mcp` tool; Anthropic as
`mcp_servers` beside the messages with an `mcp_toolset` among the tools. goose models it once and
translates, so a client that arranged its tools on one wire keeps them against a backend that serves
the other, and the beta flag that opens the field travels with the request that needs it.

A restriction that cannot be carried is refused rather than reported: an approval step the backend
cannot stop for, a read-only filter it cannot apply, a denylist on a wire that names only what is
allowed. Serving those would run exactly what the caller asked to have withheld, and a fallback chain
may still have a link that can honour them. What is merely descriptive — a server description, a
header, a loading hint — is reported instead, and `strict` refuses on that too. A backend with no such
field at all says so once, with the count of servers the model did not get.

## What it guarantees

The point of it is that nothing goes missing between the two ends.

- **Content keeps its shape.** A turn is an ordered list of typed parts, so interleaved thinking, tool
  calls and text survive, and each thinking block keeps its own signature. A generated image is one of
  them, streamed preview by preview on the wire that has an item for it. Anything goose does not model
  round-trips verbatim instead of being dropped at parse — a request field, a content block, and the
  fields a backend puts on its own answer, which reach a client speaking that same wire.
- **A drop is a value, not a log line.** Every seam returns diagnostics with stable codes. They reach
  an embedded caller on the response and a proxy client in `X-Goose-Diagnostics`.
  `WithLossPolicy(llm.LossPolicy{Strict: true})` turns any content loss into a refusal.
- **Usage is first class.** It is returned on every call, unary and streamed, attributed to the
  provider and model that actually served it, summed correctly across retries and fallback chains, and
  pushed to any sink you register. `X-Goose-Usage` carries it to a proxy client that did not ask for it
  in band. Prompt counts include the cache figures, completion counts include reasoning, and each
  adapter normalises into that convention before anything aggregates. A column the client's own wire
  has no field for — a cache split on the Ollama wire, a price on any of them — is reported rather
  than left as a short bill the client cannot tell from a complete one.
- **Failure carries its evidence.** A call that failed part-way still returns what was generated and
  what it cost: the upstream has already been paid. A 200 carrying an error body is an error, not an
  empty success, and a stream that ends without a finish reason is reported rather than presented as
  a complete answer — on the unary path and on the streamed one, where the framing is still closed
  but an error event goes out first, because `[DONE]` is what a client reads as "that was all of it".
- **A refusal is written in the caller's own wire.** Not only an upstream's: the proxy's own — an
  unaccepted key, a rate limit, a saturated listener — reaches the client in the shape its SDK parses,
  because an error it cannot read is an error it cannot act on.
- **An upstream's words are read, not relayed.** A backend's error body is parsed into the message,
  type and code the client's own format expects, whatever shape the vendor wrapped it in; the raw
  body reaches the operator's ledger and log and stops there. `redact_upstream_errors` keeps it out of
the ledger as well, for a refusal that quotes the request back, and replaces the
  vendor's sentence with a generic one for the kind of failure, for a proxy shared with people who do
  not hold the account.
- **A request nothing can serve is refused before it is sent.** Capabilities are consulted, not just
  declared: a second candidate a backend will never return, an image it cannot read, a history whose
  tool results it has nowhere to put, or a request larger than it accepts fails at the routing step
  and hands on to the next link of the chain. What a backend merely will not honour is removed from
  that attempt and reported instead: a system prompt with no channel of its own is folded into the
  conversation, a thinking signature another vendor cannot verify is stripped, cache breakpoints past
  the backend's budget are dropped — on Bedrock, where only some models serve prompt caching at all,
  a breakpoint for one of the others is dropped whole and reported — a sampling control or a tool's strictness the backend has no field
  for is named, a tool result's failure flag that has no field to travel in is named, and an unknown
  request field that would be refused is not sent. A control two vendors spell differently — one tool
  call at a time, which one puts beside the tools and the other inside the tool choice — is one
  setting here and each backend's own spelling there. Every field in
  `llm.Capabilities` is consulted by the router or by the adapter that owns it; none is decoration. A
  vendor verb that is not generation — Gemini puts the verb in the path — is refused rather than
  answered with a completion nobody asked for, or forwarded to `server.passthrough` where one is
  configured.
- **A failed call is reported as one.** An upstream that generated something and then failed is not
  served as a 200 with a short answer. What it cost still reaches the client on the usage header,
  because no error body in any served format has a field for it.
- **Every delivered call is on the record.** A retry an adapter made and threw away gets its own
  ledger row: it reached the backend and was charged for, even though only one answer came back.
- **Readiness means the upstreams.** `GET /ready` probes each provider that offers a cheap way to ask
  and answers 503 when one cannot serve, rather than reporting that the listener is up. The verdict is
  public, because an orchestrator holds no key; the per-provider reason is not, because an upstream's
  own error text names accounts and quotas. The answer is held briefly, so an endpoint that takes no
  credential is not a way to spend one. The model list is served on the endpoint each wire defines for
  itself, and a provider with no catalogue — or one whose catalogue could not be read — is named on
  the diagnostics header rather than quietly shortening the list.
- **A caller's own key goes only where it is good.** It reaches a backend only where the operator
  said it may, on the routed path and the passthrough one alike. A credential presented to the proxy
  reaches the backend it was minted for and no other: a chain that fails over to a second vendor authenticates
  with the operator's own, and the substitution is reported rather than assumed. A client may hold
  the proxy's key and its own at the same time by sending them on different headers — every
  credential on the request is checked against the proxy's, and the one that is not it is the one
  handed on.
- **Failover is bounded by the commit point.** The router pulls the first frame before handing the
  stream over, so another provider can be tried while nothing has been written, and never afterwards.

## Building

```sh
make build      # bin/goose
make test       # go test ./...
make race       # go test -race ./...
make fuzz       # each fuzz target for a minute
make vuln       # govulncheck ./...
make release    # dist/ for six platforms, with checksums
```

Everything above runs against a stand-in, which pins what goose does and says
nothing about whether the vendor agrees. The conformance suite asks the real
backends, so it is behind a build tag and each one is skipped unless its
credential is in the environment:

```sh
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... OLLAMA_HOST=... \
  make conformance
```

It spends a few tokens per backend: a one-token answer, a short stream, one
tool call, one refusal and a readiness probe. Each asserts what goose promises
rather than what the vendor happened to return — in particular that a real
stream says why it ended, which is the assumption the truncation check rests on.

Go 1.27 or newer. The only external dependency is a TOML parser, used by the binary; the library is
standard library only.

## Versioning

Semantic versioning from v0.1.0. Before v1 the seams named in [What it
guarantees](#what-it-guarantees) are settled and the rest may still move; a
breaking change to `llm`, `route`, `wire` or `httpx` gets a minor bump and a
line in [CHANGELOG.md](CHANGELOG.md).

## Licence

MIT. See [LICENCE](LICENSE) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
