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
| `bedrock` | AWS Bedrock through the Converse API | SigV4, from a key, a profile, a web identity or the runtime's own |
| `vertex` | GCP Vertex: Gemini, Claude, Llama, Mistral | OAuth2 bearer |
| `compatible` | anything else speaking the chat-completions shape | bearer or a named header |
| `claude`, `codex`, `antigravity`, `copilot` | a local agent CLI, on its own subscription login | the binary's own |

The Entra bearer an `azure` provider can use is minted per request, because these tokens expire
mid-session, so it comes from an embedder through `azure.Options.Token` rather than from the file; a
configured `azure` provider authenticates with its `api-key`.

`compatible` covers vLLM, LM Studio, Together, Groq, Fireworks and the rest. It assumes nothing about
extensions it has not been told about: declare what a backend honours with `dialect`, and goose reports
a streamed call as unmeasurable rather than booking it as free. An `openai` or `compatible` provider
pointed at OpenAI's own endpoint, `api.openai.com` or a regional one such as `eu.api.openai.com`,
speaks OpenAI's dialect without being told.

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
A call past a binary's `max_concurrent`, four unless set, waits for a slot rather than being shed to
the next link, which for a CLI on its own subscription is usually one billed per call or none at
all. The wait counts against the attempt deadline, and a call that runs out of time in the queue
ends there. Only as many wait as can run: each holds a `server.max_concurrent` slot while it waits,
so a call past them is shed, to the next link or a `503`.

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
has been failing, so one rebuilt per request forgets that. Every caller shares
the breaker, so it counts only what a backend did: a failure the request would
meet on a healthy one — malformed, too long or too large, naming a model the
backend does not have, refused before it was sent, asking for a capability
that link lacks, carrying a loss the caller's policy refuses — does not count.
On a backend with no key of its own that hands on the caller's, a refused or
rate-limited key does not count either, whether the caller sent one or none.
Nor does a send that ran out of time after waiting for a slot at the backend's
gate, whose time went on the calls ahead of it. A model the operator named on
the route is the backend's to serve, and its absence counts. A stream is
judged when it ends, not when it is handed over, because a backend can fail
after its first content as surely as before it. After its cooldown a backend
gets exactly one probe, and a probe that streams holds the slot until it ends.

Nothing here reads your environment or opens a file you did not name. The two backends
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

An EKS pod identity is handed only a token file, which the agent rotates under the running process,
so `bedrock.ContainerCredentialsFromFile` reads it again at every refresh. A service account's role
under IRSA is assumed through STS with `bedrock.WebIdentityCredentials`, which reads its token file
again at every refresh too. Either container source given neither URI fails when asked rather than
crashing, so the AWS SDK's variables can be handed to it as they are.

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
model = "anthropic.claude-sonnet-4-5-20250929-v1:0"
options = { region = "us-east-1", profile = "work" }

[[provider]]
id = "local"
kind = "ollama"
base_url = "http://127.0.0.1:11434"

[[route]]
name = "claude"
models = ["claude-*"]
chain = ["claude", "bedrock", "local:llama3.1:8b"]

[[route]]
name = "rest"
chain = ["local"]

[usage]
log = "usage.jsonl"
summary = true
```

[`goose.example.toml`](goose.example.toml) is the same file with every key annotated. A chain link
written `provider:model` asks that link for the model named; one written bare asks for the route's
`model`, else the provider's own, else the model the caller named, which a fallback on another
vendor seldom serves. A relative path in the file — the `log`, a TLS certificate or key, a
credentials file, an `[[mcp]]` `command` or `dir`, an agent CLI's `base_url`, `options.dir` or
`options.temp_dir` — is resolved beside the file that names it, not in the working directory; a
command is found inside its `dir` where one is set, and a bare command name on `PATH`. `-usage-log`
is relative to where it is typed.

Without `-c` goose reads `~/.config/goose/config.toml`. A `goose.toml` or `.goose.toml` in the
working directory is read only when `-c` names it, since that file chooses the commands goose runs
and the hosts its keys go to; one skipped there is named on stderr with the `-c` that would read
it, and a start that finds no file at all says where one goes.

A Bedrock `profile`, named there or by `AWS_PROFILE`, must hold a key pair in the shared credentials
file; SSO, assume-role and `credential_process` profiles are not read. A named profile that cannot be
used fails the call rather than falling back to the runtime's role, which would sign as somebody
else, so the runtime's role is reached only when no profile is named. One named in
`options.profile` also comes ahead of `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, while
`AWS_PROFILE` yields to those and to an IRSA web identity (`AWS_ROLE_ARN` and
`AWS_WEB_IDENTITY_TOKEN_FILE`), as it does in the AWS SDKs. A web identity is used or refused in the
same way, so a pod whose role STS refuses never signs as its node; STS is asked at the region's own
endpoint, or at `AWS_ENDPOINT_URL_STS` over HTTPS or to a local address. A Vertex key named by
`options.credentials_file` or `GOOGLE_APPLICATION_CREDENTIALS` is used or refused too, and the
metadata server is asked only when none is named.

A chain walks on when a link fails for a reason the next one might not share: a refused or expired
credential, a model that backend does not serve, a rate limit, an outage. It stops where walking on
would change nothing — a request no backend can read, a capability none of them has, a caller that
has gone, or its own attempt deadline having passed, which covers any wait for a slot at the
backend's concurrency gate as well as the send, since the caller has already waited that whole
budget and the next link would be given its own. Links are tried in the route's order, except that
a request offering the model tools goes first to the links that carry every one of them. Bear in
mind that a key which has been revoked on the first link is a reason to reach the second, so a chain
is also a way to keep spending after a backend has started refusing.

A provider takes `headers = { ... }` for a gateway in front of it that wants one; the credential is
set after them, so a header named there cannot replace it, and they go on passthrough calls too,
over any header of the same name the caller sent. A vendor's usual variable, `ANTHROPIC_API_KEY` and
the rest, is sent only to that vendor's own host over TLS or to a base URL typed with `-u`: a
provider whose `base_url` points anywhere else, a gateway included, gets a key only from `api_key`
or `api_key_env`, and a variable held back is named on stderr. OpenAI's regional hosts, such as
`eu.api.openai.com`, count as its own, as do Azure's government (`*.openai.azure.us`) and China
(`*.openai.azure.cn`) clouds, and an Azure `options.resource` is judged by the host it builds.

`server.metrics` serves `GET /metrics` for a scraper; `server.max_concurrent` and
`server.max_per_minute` bound what callers can take. The allowance is keyed on what a caller has
proved rather than on what it claims, since a caller writes its own headers: with `server.api_key`
set, everyone holding it shares one allowance, and everyone who does not share the allowance of the
connection they arrived on; with no key set, nothing is proven, and each address carries its own:
the one `forwarded_header` names where it is set, read from the last entry the nearest hop added,
or the connection's otherwise. A `subscriptions/listen` stream on `/mcp` gives its slot back once
it is acknowledged, since from then on it only waits; at most 256 are open at once, and one past
that is refused with `503`. `server.tls_cert` and `server.tls_key` serve HTTPS: off loopback a
plaintext listener hands every credential to anything on the path, and the binary says so at
startup. `server.max_frame_mb` raises the longest line an upstream stream may send, eight megabytes
by default, which is where a streamed image arrives whole.

The vendor endpoints and the passthrough refuse what a web page could make a browser send. A request
carrying an `Origin` that `cors_origin` does not name is refused with `403` in the caller's own wire,
and so is one that `Sec-Fetch-Site` marks as coming from another site, so a browser client, an
extension or a desktop app built on a browser engine needs `cors = true` and its origin in
`cors_origin` even where CORS itself is not needed. A loopback listener with no `server.api_key` and
no `forwarded_header` also refuses a request addressed to anything but `localhost`, a `*.localhost`
name or a loopback address, since a page can point a name of its own at `127.0.0.1` and then read
the answers as its own; a client that reaches it by the machine's hostname or `host.docker.internal`
needs `server.api_key`. A refusal names the origin or host name it turned away and the setting that
admits it, and the first refusal of each is logged as a warning even without `-v`, up to sixty-four,
after which the log says it has stopped. `cors_origin = "null"`, the origin a sandboxed frame on any
site sends, admits every site as `"*"` does, so it needs `server.api_key` too. With CORS off,
`OPTIONS` goes through the key check, the limiter and the gate like any other request, and only
`GET /health` and `GET /ready` are answered without the key. Passthrough forwards a caller's path
still escaped, so it cannot climb out of `base_url` on the operator's own credential or open a
second query string once rejoined: a literal `.` or `..` segment never reaches it, cleaned away by
the listener's own routing first, and one written percent-encoded is refused with `400`. Only
`GET`, `HEAD`, `POST`, `PUT`, `PATCH`, `DELETE` and `OPTIONS` are forwarded; any other method,
`TRACE` and `CONNECT` among them, is refused with `405`. A body over `max_body_mb` is refused with
`413`, before anything is sent where its length is announced. With `cors = true`, the upstream's own
copies of the headers goose sets — `Access-Control-Allow-Origin`, `-Allow-Headers`, `-Allow-Methods`
and `-Expose-Headers` — are dropped rather than sent beside goose's, since a browser treats two
`Access-Control-Allow-Origin` values as no match at all; any other CORS header the upstream sends
passes through.

`goose -v` logs one line per request with what it cost and what the translation lost. Under
`goose mcp`, `-v` logs each call to goose's own tools, and what `redact_upstream_errors` withheld
from `ask`, on stderr. `goose -check` validates the file and prints what would be served, spawning
nothing and writing nothing, though it does refuse a `[usage] log` whose directory does not exist.
An unknown key is an error, not a shrug: in TOML a root key written below a table header belongs to
that table, so a setting can look applied and not be. That holds inside `options` too, which is a
free map and would otherwise take a misspelt key without a word; the error names what the kind does
read. A setting the kind does not read is refused the same way — a key on Bedrock or Vertex, a key
on an agent CLI with no `api_key_env_name` option to hand it on, `max_retries` on an agent CLI — and
so are `server.api_key_env` naming a variable that is not set and an argument left over after the
flags. A provider's `api_key_env` naming one is said on stderr instead, and that backend's calls
carry no key until it is set. `goose mcp -check` checks what `goose mcp` runs — not the listener's
TLS files — and lists its tools, not the proxy's endpoints.

Stopping the proxy with requests or streams open waits up to ten seconds for them to finish, and says
so on stderr. A second interrupt cuts them off at once, and goose then waits only for them to book
what they spent.

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

A server that goes away is dialled again rather than written off: a stdio server that exits or stops
reading is started afresh, an SSE stream that ends is opened again, whatever was said on it last, as
is one whose endpoint the server answers with `404`, and a streamable server that no longer holds
the session is greeted again. While a server cannot be reached its tools are not advertised, and
they return with the next refresh that reaches it. One server that stops answering costs its own
refresh, not everyone else's, and one answer too large to hold costs only the call it answers.

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
that lists tools; the loopback and API-key rules above cover it either way. `ask` may leave out
`model` where the route or the provider names one, or where the backend picks its own, as an agent
CLI does; otherwise it says to pass one rather than sending a call with none. A backend that refuses
comes back as a failed tool call carrying the reason, not as a protocol error the agent cannot read;
under `redact_upstream_errors` the reason is the provider, the kind of failure and the status,
without the backend's own words. An empty prompt or unreadable arguments come back as a failed tool
call too. Each note `models` gives beside its list, such as a catalogue that could not be read,
names the backend it is about.

Most agents spawn a command and talk to its pipes rather than opening a socket. `goose mcp` serves the
same protocol on stdin and stdout for them — the same routes, the same tool servers, with goose's own
tools always on. Like the proxy, it reads only a file named with `-c`, or
`~/.config/goose/config.toml`, never one in the directory its client starts it in, since that is
whatever project is open; and it binds no port, so it neither reads nor needs `server.api_key_env`:

```json
{ "mcpServers": { "goose": { "command": "goose", "args": ["mcp", "-c", "/path/to/goose.toml"] } } }
```

The variables that hold the providers' keys must reach `goose mcp` too. Many clients start a stdio
server with only a small default environment, so pass them through the entry's `env` where the
client supports one; one that `api_key_env` names and that is missing is said on stderr, and leaves
only that backend without its key.

Everything goose has to say in that mode goes to stderr, because a line of its own among the frames is
a parse error at the other end. A cancellation withdraws the call it names — a pipe has no stream to
close per request — and the withdrawn call is answered with nothing at all.

Both halves speak both eras of the protocol. `2026-07-28` replaced the `initialize` handshake with
per-request metadata, a `server/discover` call and no session at all; the revisions before it,
`2024-11-05` through `2025-11-25`, open a session and remember what was agreed. goose serves a client
of either era on the same endpoint and the same pipe, and dials a server of either: it asks
`server/discover` first, and a server that refuses it — or says nothing, which some do — gets the
handshake instead, once, with a deadline of its own rather than the whole call timeout. Opening a
stream of the older SSE transport is held to the caller's deadline and the server's `timeout`, as a
call is.

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
way out and checked against the body on the way in, an integer as the body wrote it however large it
is; a definition that asks for something the rule does
not allow is never called incorrectly. From a streamable HTTP server, which goose mirrors for, it is
left out of the list and the startup banner names what it left out. From a stdio or SSE server, which
mirrors nothing, it is served on the pipe and to clients of the older era, and left out only of the
list served to a current-revision client over HTTP, which would mirror it. A tool server's own
definition crosses the proxy whole: its title, its icons, its
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
  calls and text survive, and each thinking block keeps its own signature, as does a Gemini call,
  text part or generated image that carries one. A signature handed to a client on another vendor's
  wire is marked with the family that minted it, as base64 on the Gemini wire whose field is bytes,
  and is read back the same way; a wire with no field for a signature reports that it did not
  travel. A generated image is one of the parts, streamed preview by preview on the wire that has an
  item for it. Anything goose does not model round-trips verbatim instead of being dropped at parse —
  a request field, a content block, and the fields a backend puts on its own answer, which reach a
  client speaking that same wire. A request field on the Anthropic wire reaches an Anthropic backend
  with the caller's `anthropic-beta` flags, which open most of them. Where those flags are not sent,
  the field is not sent either, because the backend refuses a field sent without its flag, and that
  is reported as a loss: on Vertex, or on a provider with `client_betas = false`.
- **A drop is a value, not a log line.** Every seam returns diagnostics with stable codes. They reach
  an embedded caller on the response and a proxy client in `X-Goose-Diagnostics`.
  `WithLossPolicy(llm.LossPolicy{Strict: true})` turns any content loss into a refusal, on a stream
  as on an unstreamed answer: a stream ends at the frame carrying the loss, though what it delivered
  before stands. A block goose carried through whole is delivered to a destination that can write it
  out, which is an embedded caller that declares no `Renders` or a proxy client on the block's own
  wire, and `Strict` refuses it for any other. So it is for reasoning, an image or a tool call the
  destination's wire has no field for, and all of these are weighed before any of the frame that
  carries them is written: a stream whose first content holds one is refused with the failure's own
  status rather than a 200 carrying only an error. A loss the proxy meets decoding a request is held
  to the policy before anything is sent. One it meets encoding the answer is refused in place of
  that answer or, on a stream, where the loss is met; that call was still sent and billed, and the
  refusal says so.
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
  body, or the error a backend reported inside a 200 or on a stream, reaches the operator's ledger and
  log and stops there. `redact_upstream_errors` keeps it out of the ledger as well, for a refusal
  that quotes the request back, and replaces the vendor's sentence with a generic one for the kind of
  failure, for a proxy shared with people who do not hold the account. That covers the vendor
  endpoints, goose's own `ask` and `models` tools, a passthrough call that fails before the upstream
  answers, so the upstream's address does not reach the caller, and `GET /ready`'s per-provider
  reason. goose's own refusals, such as a loss the policy would not take or a capability a backend
  lacks, are not an upstream's words, so they keep their code and message.
- **A request nothing can serve is refused before it is sent.** Capabilities are consulted, not just
  declared: a second candidate a backend will never return, an image it cannot read, a history whose
  tool results it has nowhere to put, or a request larger than it accepts fails at the routing step
  and hands on to the next link of the chain. So do a tool choice that insists on a call, on a
  backend that keeps none of the tools or cannot be told to make one, and a loss the caller's policy
  refuses, which is known before the request is sent: once it is planned, while the proxy decodes
  it, or while the adapter builds it. A request that offers tools — definitions, tool servers the
  backend dials, or tools carried for one wire — goes first to the links that carry them all, in the
  route's order. A link that would strip some is tried only once those have failed, been skipped or
  refused, and there the tools are stripped and reported. What a backend merely will not honour is
  removed from that attempt and reported instead: a system prompt with no channel of its own is
  folded into the conversation; a signature another vendor minted is stripped, whether it signs
  thinking, a call, text or an image — Claude on the Anthropic API, Bedrock, Vertex or the Claude CLI
  is one family, OpenAI and Azure OpenAI another, and Gemini on its own API or Vertex a third, and
  each backend is sent only its own family's; a Gemini call in the current turn that has lost its
  signature is sent with the placeholder Google documents for that case, and the loss is named;
  cache breakpoints past the backend's budget are dropped — on Bedrock, where only some models serve
  prompt caching at all, a breakpoint for one of the others is dropped whole and reported; a sampling
  control or a tool's strictness the backend has no field for is named; a temperature, `top_k` or
  narrow `top_p` Claude refuses beside thinking is left out or widened and named, thinking beside a
  tool call the model is made to make is left off and reported, and a thinking budget outside what
  Claude takes is moved to the nearest one it does; a tool result's failure flag that has no field to
  travel in is named; and an unknown request field that would be refused is not sent. A control two
  vendors spell differently — one tool call at a time, which one puts beside the tools and the other
  inside the tool choice — is one setting here and each backend's own spelling there. Every field in
  `llm.Capabilities` is consulted by the router or by the adapter that owns it; none is decoration. A
  vendor verb that is not generation — Gemini puts the verb in the path — is refused rather than
  answered with a completion nobody asked for, or forwarded to `server.passthrough` where one is
  configured.
- **A failed call is reported as one.** An upstream that generated something and then failed is not
  served as a 200 with a short answer. What it cost still reaches the client on the usage header,
  because no error body in any served format has a field for it.
- **Every delivered call is on the record.** A retry an adapter made and threw away gets its own
  ledger row: it reached the backend and was charged for, even though only one answer came back.
  Stopping goose waits up to five seconds for the calls it cut off to book what they spent before it
  closes the ledger and prints the summary, and says so on stderr when one does not.
- **Readiness means the upstreams.** `GET /ready` probes each provider that offers a cheap way to ask
  and answers 503 when one cannot serve, rather than reporting that the listener is up. The verdict is
  public, because an orchestrator holds no key; the per-provider reason is not, because an upstream's
  own error text names accounts and quotas — under `redact_upstream_errors` a trusted caller is told
  the kind of failure rather than the vendor's own account or quota sentence. The answer is held
  briefly, so an endpoint that takes no
  credential is not a way to spend one. The model list is served on the endpoint each wire defines for
  itself, and a provider with no catalogue — or one whose catalogue could not be read — is named on
  the diagnostics header rather than quietly shortening the list.
- **A caller's own key goes only where it is good.** It reaches a backend only where the operator
  said it may, on the routed path and the passthrough one alike. A credential presented to the proxy
  reaches the backend it was minted for and no other: a chain that fails over to a second vendor authenticates
  with the operator's own, and the substitution is reported rather than assumed. A client may hold
  the proxy's key and its own at the same time by sending them on different headers — every
  credential on the request is checked against the proxy's, and the one that is not it is the one
  handed on. A key sent as `?key=` is a credential like one in a header: it is never forwarded in
  the URL, and it reaches a backend only under the same rule, and a query the proxy cannot read is
  refused with `400`. Passthrough follows the routed path's rule too: a key on `x-api-key` goes only
  to an Anthropic backend and one on `x-goog-api-key` or `?key=` only to a Google one, and a key held
  back is named in `X-Goose-Diagnostics`. Passthrough carries the headers, betas and OpenAI
  organisation and project configured for the backend, and a caller's own organisation and project
  only with the caller's own key. On the routed path an Anthropic backend is also sent the caller's
  own `anthropic-beta` flags, and the request fields they open, under the operator's key, unless the
  provider sets `client_betas = false`; some of those flags change what a call costs.
- **Failover is bounded by the commit point.** The router holds what a backend sends ahead of its
  first content — its opening, its counts — and pulls that content before handing the stream over,
  so a backend that fails before it leaves the call to another while nothing has been written, and
  never afterwards.

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
