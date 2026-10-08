# Changelog

## Unreleased

A second audit, this time of the code rather than the README's claims about it.
Everything below was found by reading the source and confirmed against it before
it was touched; a handful of reported problems turned out not to be problems and
were left alone.

### Fixed
- A route chain stopped at the first link that failed for a reason the next one
  might not have shared: a revoked key, a model that backend does not serve, a
  region that refused. It now walks on for those and stops only where walking on
  would change nothing — a malformed request, a capability none of them has, a
  caller that has gone, or the attempt deadline having passed.
- The shipped binary set no `llm.Limits` at all, so an upstream call had no
  deadline and its body no ceiling: one stalled backend held a request, its
  connection and its concurrency slot until the client gave up. Both now have
  defaults, settable with `server.attempt_timeout` and `server.max_response_mb`.
- Rate limiting and request logging sat inside the API-key check, so the flood
  and the credential stuffing they exist to shed and to record passed through
  untouched. The log is now outermost and the limiter outside the key check,
  while the concurrency gate stays inside it.
- The Claude CLI restates a streamed message one block at a time, printing a
  line as each block closes and before the message is finished, and every one
  of those lines was delivered again beside the deltas that had already carried
  it, so every answer arrived twice and a tool call's arguments were written
  twice over. Each line was also counted as a turn of its own carrying the
  whole prompt again, so the prompt and cache figures were counted once per
  line and the turn count inflated. A line restating a message the deltas
  carried now gives only what the deltas could not, and a message is one turn
  however many lines it takes.
- Streamed tool calls whose fragments carried no index were split into separate,
  malformed calls on four client wires. The merging rule now lives in one place.
- Bedrock advertised prompt caching and never sent a cache point, so every call
  was billed at full price with nothing said. It now sends them, for the models
  that serve them.
- `llm.Stream` was read by its consumer and closed by its supervisor with
  nothing between them: closing one mid-frame raced the accounting and could
  corrupt the row that was billed. A consumer that abandons the stream now ends
  it rather than leaving the body open and the call unbilled.
- A multi-turn stream cut off after its first turn was reported as complete.
- An MCP tool whose own name read like another server's qualified one was called
  in its place, and a server could take a name that way on purpose.
- `AWS_EC2_METADATA_DISABLED` was ignored whenever a container credential was
  configured, so a task whose role failed fell through to the node's. A
  configured container endpoint is now the only runtime source, as it is in
  the AWS SDKs: when it refuses, or `AWS_CONTAINER_CREDENTIALS_FULL_URI` is
  itself refused, the request fails rather than borrowing the node's role.
- A credential refresh held its lock across the network exchange, so every
  waiting request ignored its own deadline and the refreshes ran one after
  another. They now share one exchange, and every request waits on its own
  context, the one that started the exchange included. A credential still
  being fetched when the deadline passes is reported as that timeout, not as a
  key the backend refused.
- A child process that ignored SIGINT held shutdown open for ever, and a
  process group was signalled by a pid that may since have been reused.
- A backend that answered 200 and then failed on its first event crashed the
  connection serving the Google wire, so the client was left with no reply at
  all. The same shape reached it through a caller hanging up early.
- The proxy's own attempt deadline was reported to a client still on the line as
  though that client had hung up — 499 on a streamed call where an unstreamed
  one said 504 — and the ledger could not tell the two apart either. Once a
  backend had answered its headers and then gone quiet, neither shape was named
  at all: the wall read as the backend failing, 502, and counted a fault against
  a backend that had none. A context error is now the proxy's wall or the
  caller's departure wherever it lands, on every provider.
- A caller that went away — hung up or out of its own time — counted as a
  failure against the backend it left, so a client that abandons requests could
  open the breaker on a backend that had answered nothing wrong. The probe slot
  such a call was holding goes back either way, or the one probe a backend gets
  after its cooldown would take it out of the chain for good.
- An MCP server offering both a name and its own qualified spelling of another
  had one of the two silently dropped, and nothing could call it again. A server
  listing one name many times now costs one entry rather than a name per copy,
  which it built under the lock every refresh takes.
- Vertex read its catalogue from the project-scoped path, which serves no
  listing: every Vertex backend read as unready however healthy it was. Model
  Garden answers that instead.
- Vertex in the `global` region built a hostname nobody answers, a provider
  given neither model nor publisher picked one silently, and a model written the
  way Vertex writes it — `google/gemini-...` — had its slash escaped into the
  model id, naming a model that does not exist.
- Health checks and model lists went out unsigned, so every backend that mints a
  token instead of holding a key — Azure with Entra, all three Vertex surfaces —
  read as unreachable however healthy it was.
- Passthrough sent a bearer token and the caller's query to backends that take
  neither: Azure's own `api-key` header and pinned `api-version` were dropped,
  and a minted token was never asked for at all.
- Azure's Responses API was addressed under `/deployments/`, where it does not
  live. It is addressed by naming the deployment in the body instead, which the
  dated surface now does — it serves one deployment either way.
- A Gemini tuned model was addressed under `/models/`, which is not the
  collection it is served from.
- Cloudflare read its catalogue from an endpoint Workers AI does not serve, so
  its models were missing from `/v1/models` and readiness never turned green. It
  now reads the account's own, whose refusals arrive as 200 with the reason in
  the body, and follows the host's paging rather than assuming the page size it
  asked for was honoured.
- Smaller: the breaker's probe was kept for good by a provider that answered
  with neither stream nor error; an attempt already sent went unbilled if the
  caller left during a backoff; a truncated AWS event stream read as a clean
  end; a route could not say `strict = false`; a candidate index the upstream
  chose was an unbounded key while a turn ran; a client-chosen model name was an
  unbounded metrics label, an unbounded route-matching cost, and an unbounded
  ledger entry.
- `goose.example.toml` was missing twenty of the keys the code reads, and an
  unknown key is fatal, so those were keys nobody could use. A test now holds it
  to what the config structs declare, and another holds it to passing `-check`
  with every setting uncommented, which the `prefix` shown under the Ollama
  provider did not: it sits under a Google provider now.

- `server.redact_upstream_errors` reached the unstreamed reply and nothing else.
  A backend that failed after the stream had started wrote its own sentence —
  the operator's organisation, its billing contact — straight into the body,
  and the code it chose survived redaction on both shapes. `WithRedactedErrors`
  keeps it out of the ledger, and the binary's one key now sets both, which it
  had not been setting at all.
- A caller that asked a tool server for none of its tools got all of them: an
  empty allow list read as no list at all, through the wire, through the
  request's own copy, and out to the backend. An approval mode goose did not
  recognise read as "no approval needed", a policy naming both `always` and
  `never` kept the permissive half, and `never` exempting nothing exempted
  everything. All four now fail closed.
- An event past the frame ceiling was handed over as a whole one and the error
  raised afterwards, so an upstream could choose where a frame ended and
  therefore what it said. Reading also carried on past the ceiling.
- A tool server could take a name reading as another server's qualified one
  while nothing collided, and the model was offered it as that server's. A
  server could also report a name, a version or a tool name that moved the
  operator's cursor, answer with neither a result nor an error and have a tool
  that never ran read as one that ran and said nothing, offer one name twice so
  that the definition on display was not the one used on the wire, and pick a
  protocol revision that no header could carry. How long a catalogue is is no
  longer the server's to choose either, and listing it no longer costs the
  square of its length under the lock every refresh wants.
- A system turn that was not at the front was reported folded and left where it
  was, which fails a caller who refuses that loss only after the backend has
  answered and billed. The fold also swallowed the cache breakpoint it
  absorbed, and the input ceiling weighed neither the caller's extras nor its
  tool servers, so a megabyte of either walked past a four-kilobyte wall.
- Losing every cache breakpoint was quieter than losing one of them, and
  dropping every tool definition was quieter than dropping every tool server.
- A call nobody counted booked as a call that cost nothing whenever the backend
  happened to mention how long it took. The same row was missing from the
  discarded attempts, which are sent and may well be billed, and a streamed
  call's copy of the bill left out what the ledger recorded.
- An upstream's error message was thrown away whole when any sibling field had
  a shape goose did not expect — `{"error":{"message":"…"},"status":429}` left
  the client with nothing but the status line.
- A redacted thinking block was base64 on one backend and raw bytes on another,
  so passing one through the other corrupted it beyond recovery.
- The Responses wire left out fields its own schema requires — `annotations`,
  `summary`, `call_id`, `refusal` — which a strict client refuses to parse.
- Smaller: a reasoning count larger than the completion it sits inside went
  unflagged; a total assembled from a priced turn and an unpriced one read as
  fully priced; two prices in two currencies compared by number; two copies of
  one usage could overwrite each other's notes; a refusal became ordinary text
  and an image lost its media type, both without a word; a tool call whose
  arguments did not survive travelled as a call with none; `Frame.Empty` did not
  count an id, a model or a session, so three providers each patched around it
  differently; a line of exactly the ceiling was refused for its newline; a
  server's stray stdout line killed the call it interrupted; and an MCP resource
  arrived with its uri and media type stripped off.
- An MCP server chooses its own name, its version, its tool names and the words
  it fails with, and goose prints all of them. None of them was held to a line,
  so one of them could move the cursor and rewrite what was printed above it, or
  simply be four megabytes long. Every line somebody else chose is now held to
  one: what a failed call said, the tool name a caller asked for, the method it
  named, the revision it asked to speak, the header names a tool's own schema
  carries, and what a server says while its tool runs. What is kept is what Go
  itself prints unescaped, which leaves out the C1 controls, the bidi overrides
  and the other invisibles as well as the ASCII ones.
- A tool name goose will not carry is reported as the name it was rather than as
  what is left once the characters that made it unusable are taken out — which
  was another tool's name, and one that was being served.
- A catalogue is read and held on ceilings that a server cannot spend against
  itself: how many tools, how many names goose will not carry, how many entries
  are read and how much they weigh are bounded separately. A run of unusable
  names at the front of a list no longer leaves no room for the tools behind it,
  a schema this proxy cannot mirror no longer takes the place of a tool it would
  serve, a server repeating one page is no longer read a hundred times over
  every thirty seconds, and two thousand tools sent as a gigabyte and a half are
  no longer all held. A catalogue that lists each of its tools more than once is
  still read whole and is no longer reported as one that was cut.
- A tool result that holds itself ended the process with a stack overflow rather
  than an error, and one that holds itself twice doubled at every level. How
  deep one is read and how much comes out of it are both bounded now, reading
  stops at the ceiling rather than building blocks it will throw away, and a
  result with more in it than will fit says so rather than losing the rest
  quietly.
- A locally registered tool answering with neither a result nor an error was
  read as an answer, and the first thing that reached into it took the process
  with it.
- The accumulation ceiling reported itself once for every delta that arrived
  after it rather than once, so the setting that exists to stop an answer
  filling memory filled it faster than the text it was refusing: thirty-two
  megabytes past the ceiling held eight hundred thousand copies of one
  diagnostic, and every copy travelled into the answer and the ledger row.
- An attempt that was sent and then discarded on a retry had the upstream's own
  message written into the ledger whole, whatever `WithRedactedErrors` had been
  told, because that row is booked from under the provider. A backend's refusal
  routinely quotes the request back, so the passage an operator asked to keep
  out was kept out of the row that failed and written down on every row that was
  retried past.
- Every attempt that was sent now leaves a row saying the same thing about what
  it cost: nobody counted it, which is not the same as it having cost nothing.
  One ledger could hold a discarded retry saying the backend reported no usage
  beside the attempt that actually failed saying it was free.
- A backend answering with neither a response nor a failure was booked as a call
  that worked and told the circuit breaker it was healthy, while the caller was
  handed an error. The streamed side of the same case left no row at all, for a
  call the vendor may well have billed.
- A usage sink panicking took the process rather than the row whenever it was
  the only sink configured.
- A note could appear twice on an answer, and for an agent CLI or a
  synthesised stream on its ledger row too. A frame's own notes reached an
  answer folded from a stream once from the folder and once from the stream
  that had kept them as well, and only the agent CLIs fold a stream. What a
  provider noted while building its request, or an agent CLI while flattening
  it or folding its stream, was kept on the attempt with the caller's policy
  applied and on the answer without it, so a loss the caller allowed still
  read as one. Each note now appears once, with the policy applied.
- Every OpenAI answer read as one that had lost content, because each choice
  carries a null `logprobs` unless the caller asked for them and a null field was
  counted as a field left behind. And on a stream the same note was raised on
  every chunk, so the ledger row for an answer held a copy of it per token. A null
  field is no longer counted, and the fields a backend puts beside its answer are
  said once a stream.
- Ollama's `/api/generate` carried a backend's own fields on an unstreamed
  answer and dropped every one of them on a streamed one, with nothing said.
- Two consecutive frames carrying nothing but fields a wire does not model were
  one set of them by the time the next chunk could carry either, on the OpenAI
  and Ollama streams. They are all carried now, up to what one chunk will hold,
  and that ceiling is reported rather than the rest going quietly. Fields still
  waiting when a stream ends are not reported: a backend goose asks for usage
  ends its stream with a chunk carrying no content, a caller who did not ask for
  usage gets nowhere to put that chunk's fields, and whatever the backend was
  decorating its chunks with cannot be told there from a vendor payload somebody
  wants.
- A loss the caller's policy refuses was known once the request had been
  planned, but the request was still sent and billed before being refused, and
  the next link, which might have carried it whole, was never tried. It is now
  refused before anything is sent, booked as a rejected attempt that cost
  nothing, and the chain walks on.
- `strict` and a route's `deny` did nothing for a streamed request: every frame
  was delivered and the stream ended cleanly, with the fatal diagnostic sitting
  only in the trailer. A stream is now held to the same policy as an unstreamed
  answer. A loss it refuses is not delivered and the stream ends in a `loss`
  failure; what was delivered before the loss cannot be taken back. Where the
  router knows a loss before handing the stream over — its first content
  carries a block of another vendor's that the caller's wire cannot write out,
  or reasoning, an image or a tool call that wire has no field for — the proxy
  answers with the failure's own status rather than a 200 carrying only an
  error event. Met later in the stream, such a part is refused before any of
  the frame carrying it is written. Either way the stream itself says what was
  lost, so the ledger row and `X-Goose-Diagnostics` name it as well as the
  error message. A refusal answered with its own status, streamed or not,
  carries the codes that caused it on that header too; it used to carry none.
  A loss
  an encoder only discovers while translating a frame into the caller's own
  wire — a signature with no field there, a refusal turned to plain text, a
  stop reason with no spelling there — is caught before that frame's bytes go
  out, but the stream has already opened by then, so it is still a 200 whose
  body is only the error event, the same as any other loss met after the first
  frame. A non-streaming backend's answer, synthesised into a stream, that a
  strict policy refused before its last frame was booked with no usage at all
  rather than what the call actually cost, and was weighed a frame at a time,
  so a caller could be handed a prefix of the answer before the rest was
  refused. It is now weighed whole against the destination before the first
  frame, images and tool calls included, so a synthesised stream is delivered
  whole or refused whole, and its usage is booked either way.
- The breaker counted failures the caller brought with it against a backend
  every caller shares, so any one caller could take a healthy backend out of
  the chain for everyone: a few over-long or oversized prompts, a model the
  caller named that the backend does not have, a request refused before it was
  sent, a capability that backend lacks, or a revoked, missing or withheld key
  on `allow_client_key`. Those now give the probe slot back without counting.
  A 413 now reads as a request too large for that backend rather than as an
  upstream failure, and the chain still walks on from it, since the next
  backend has a ceiling of its own. A model the operator named on the route
  still counts when its backend lacks it. On an `allow_client_key` backend with
  no key of its own, a refused or rate-limited key is the caller's whether one
  was sent or none; an operator key that stops working still opens the breaker.
- A half-open breaker could let a second probe through: a call admitted before
  the breaker opened gave back a slot it never held when it finished during the
  probe. Only the probe gives the slot back now.
- A backend that answered 200 and then reported a failure before any content —
  Anthropic's `overloaded_error` after `message_start`, and the same shape from
  the OpenAI Responses and Bedrock openings and the Claude CLI's closing line —
  handed the caller an error in place of the next link's answer, and told the
  breaker it was healthy. The router now holds what arrives before the first
  content and walks on from a failure there, with what the opening counted kept
  on the row. The attempt deadline now runs until that first content.
- A request carrying tools reached a link that takes none, which stripped them
  and answered, so the link that could call them was never tried. A tool choice
  that insists on a call was dropped the same way. A call that offers tools now
  goes first to the links that carry every one of them, in the route's order:
  definitions, tool servers a backend dials itself, and tools carried as
  written for one wire alike. A link that would have to strip some is tried
  only after those have failed, been skipped by the breaker or been refused,
  and there it strips them and says so, so an outage of the link with the tools
  no longer takes the route down with it. A tool choice that insists on a call
  is refused on a link that keeps none of the tools, or that cannot be told to
  make a call, such as Ollama, and the chain walks on, `strict` or not. A
  request that forbids a call, or whose format has no field for one, keeps the
  route's order. When a link left to last cannot be tried either, the caller
  is given the failure of the links before it rather than a refusal that reads
  as final. That held only for a refusal the router itself made; a link that
  refused its own request before sending — an agent CLI adapter saying what
  its own translation would lose, say — still replaced the real failure of
  the tool-carrying link tried before it, and now does not either. A provider
  that can say what its own request will lose, or that it will not send it at
  all — an agent CLI given a model that reads as a flag, or left nothing to
  ask — is asked before it queues for a concurrency slot, so a call that was
  always going to be refused is refused at once rather than timing out behind
  busy ones, a bad request as the `400` it is rather than a `504`, and its row
  says what the request lost.
- A capability one backend lacked ended the chain, although every link had
  already been checked against what it declares and the next one might well
  have served. It now walks on.
- A backend that refused a key, an account or a region with the status it gives
  a malformed request stopped the chain there, as though no other backend could
  have served it. Gemini answers a key that is not valid or has expired, and a
  region or tier it will not serve, with 400; Anthropic does the same for an
  account out of credit, past its spend cap or disabled; Ollama for a model
  without tools or thinking; and Bedrock, Vertex and Workers AI for a model
  they do not serve, such as an unknown model ID, one Bedrock serves only
  through an inference profile, or a Vertex publisher model the project may
  not use. Each is now read for what it is — a refused key, a refused account,
  a model lacking what was asked of it, a model that backend does not have —
  and the chain walks on. Only a Gemini `FAILED_PRECONDITION` that names the
  user's location or billing is read as the account's; any other, such as one
  naming a file still being processed, is a malformed request.
- OpenAI's "organization must be verified" refusal is given per model or
  feature — streaming `o3`, say, or `gpt-image-1` — on a key that otherwise
  works, not as a refusal of the whole account, but it opened the shared
  breaker on every caller asking for any model behind that key. It is read as
  the model-specific refusal it is, and the chain walks on. OpenRouter's 400
  for a model id it does not have (`"<id> is not a valid model ID"`) gets the
  same reading the other backends above already do: a model that backend does
  not serve, not a malformed request, so the chain walks on past it too.
- A route or the default tools could name a tool group nobody had registered,
  and every call on it reached its backend with no tools and nothing said.
  `goose.New` now refuses it.
- `strict` refused an answer that had lost nothing: a content block goose does
  not model, carried through whole, read as a loss. It is delivered now to a
  destination that can write it out, which is a library caller that declares no
  `Renders` or a proxy client on the block's own wire, and `strict` refuses it
  for any other, whose encoder would drop it: a server tool's call and result
  from Anthropic to an OpenAI chat client, say, or a Bedrock block or a Codex
  CLI item on any wire. An unstreamed answer is weighed before any of it is
  written, and a stream at the frame that carries the block. Every loss a site
  reports is still refused whatever severity the site chose, and an unmodelled
  Bedrock stream event, which is dropped rather than carried, is now reported
  as the loss it is.
- `WithRedactedErrors` kept a refusal's body out of the ledger and nothing else.
  A failure reported inside a 200 wrote the vendor's sentence into the row
  whole, and so did a stream's error event, a Responses `response.failed`,
  Cloudflare's catalogue envelope, an agent CLI's error line and a Bedrock
  exception. Every one of them is now redacted the same way, and the note on an
  answer that arrived beside an error no longer quotes it either. Of a failed
  Responses answer only the failure is kept, not the instructions and the
  partial output that travel with it.
- `redact_upstream_errors` did not reach a passthrough call, so one that failed
  before the upstream answered returned the upstream's address, query and all,
  to the caller. It gets the same generic sentence as the routed path now.
- A streamed image failed the whole stream with `wire: line exceeds limit` once
  its base64 passed a megabyte, which a 1024×1024 PNG commonly does, although
  the same call unstreamed succeeded. A line may now be as long as an event,
  eight megabytes by default, and both follow `llm.Limits.MaxFrameBytes`, which
  the binary could not set until `server.max_frame_mb`.
- Passthrough handed the backend the caller's query whole. A key given to the
  proxy as `?key=` — its own, or a caller's that `allow_client_key` does not
  admit — reached a third-party vendor's URL and its logs, and a Google
  passthrough authenticated with it. The credential parameters `key`,
  `api-key`, `api_key` and `access_token` are now taken out, a key presented
  that way is handed on by the client-key rule like one in a header, and a
  query that cannot be read is refused with `400` in the caller's own wire
  rather than sent on.
- Passthrough could be reached without the proxy's key and outside the rate
  limiter and the concurrency gate: every `OPTIONS` request skipped all three,
  any method on `/health` and `/ready` skipped them too, and any method on
  `/metrics` skipped the limiter and the gate. An anonymous caller could send
  unlimited traffic upstream on the operator's key and fill the ledger. Only
  `GET /health` and `GET /ready` now skip the key check, only they and
  `GET /metrics` skip the limiter and the gate, and everything else the
  passthrough takes goes through all three.
- Passthrough to an OpenAI-shaped backend said the version twice:
  `/v1/embeddings` against `https://api.openai.com/v1` went to
  `/v1/v1/embeddings`, and OpenRouter, Azure and Workers AI did the same, so
  every unrouted OpenAI call was a 404.
- Passthrough sent none of the headers configured for a backend, none of
  Anthropic's configured betas, and neither the configured OpenAI organisation
  nor project. A gateway that needed its header refused the call, and OpenAI
  billed the key's default project. A caller could also pick any organisation
  or project the operator's key belongs to just by naming it. The configured
  headers now win over a caller's of the same name, and a caller's own
  organisation and project are sent only with the caller's own key.
- A client holding the proxy's key and its own on different headers was served
  only when its own was on `x-api-key`. With the proxy's key there and its own
  on the bearer, the routed Anthropic wire and passthrough handed on nothing.
  The credential that is not the proxy's is now handed on whichever header
  carries it.
- A keyless proxy on loopback, the default, served any web page the operator
  visited. `/mcp` refused browser origins, but the vendor endpoints and the
  passthrough spent the operator's credentials for any page that posted to
  them, and with DNS rebinding a page could read the answers too. Those
  endpoints now refuse a browser `Origin` that `cors_origin` does not name and
  a request `Sec-Fetch-Site` marks as coming from another site, and a keyless
  loopback listener with no `forwarded_header` refuses a `Host` that is not
  `localhost`, a `*.localhost` name or a loopback address. Each refusal names
  the origin or host name it turned away and the setting that admits it, and
  the first refusal of each is logged as a warning whether or not `-v` is on,
  up to sixty-four, after which the log says it has stopped.
- With the `google` format enabled, the default, the proxy's own refusals on
  `GET /v1/models`, `GET /models` and `GET /v1/models/{id}` were written in
  Gemini's shape and counted as `format="google"`, because Gemini's `POST`
  routes begin with those paths. The wire a refusal speaks and the format it is
  counted under are now taken from the route the request reached.
- `X-Goose-Session` was sent twice on every response that carried a session, so
  a client that joins repeated headers read `sess-1, sess-1`.
- A throttled request was always told `Retry-After: 1`. Below sixty a minute,
  an SDK that honours the header used up its retries before the allowance came
  back. The hint is now the time until the next request will be let through,
  rounded up.
- The CORS allow-list refused headers the official browser SDKs send —
  `x-stainless-*`, `anthropic-dangerous-direct-browser-access`,
  `x-goog-api-client` — and `X-Goose-Provider`, so the browser never sent the
  request.
- An OpenAI-compatible backend that gave an error's code as a number, as vLLM
  and llama.cpp do, had its message thrown away as a malformed frame.
- `openai.New` given OpenAI's own endpoint as its base URL, with no dialect,
  spoke the compatible one: it sent `max_tokens` to gpt-5 and the o-series,
  which refuse it, and asked for no usage on a stream. OpenAI's endpoint,
  regional ones such as `eu.api.openai.com` included, now takes the OpenAI
  dialect whether its base URL was written out or left to the default, and so
  does a `compatible` provider pointed at it with no `dialect` of its own.
- An Anthropic client's own request fields that goose does not model
  (`service_tier`, `context_management`, `output_config`, `container` and the
  rest) were dropped on the way to an Anthropic backend, and the only note said
  the backend takes no unknown fields, so `strict` served the call. They now
  reach it, and so do the caller's `anthropic-beta` flags beside the
  operator's: most of those fields are opened by one, and the backend refuses a
  field sent without it. A backend that is not sent the caller's flags is not
  sent the fields either, and reports the drop as a loss that `strict` refuses.
  That means Vertex, or a provider with `client_betas = false`. Reading those
  flags scanned what had been kept so far for each one found, so a client
  sending many `anthropic-beta` values cost the square of how many; it now
  costs one pass.
- A thinking budget outside what Claude takes, under 1024 tokens or not under
  the output ceiling, was sent to Anthropic and Bedrock as it was, and the
  refusal stopped the chain. A budget goose chooses, for a Responses
  `reasoning.effort` or a Gemini `thinkingConfig` without one, is now at least
  1024 and under the ceiling; on Bedrock it is 2048, or half the ceiling where
  that is less, and never under 1024. A budget the caller named, such as a
  Gemini `thinkingBudget` of 512, is moved to the nearest one Claude takes, and
  that is noted. Where the ceiling leaves no room at all, thinking is left off
  and that is reported, instead of sending a request certain to be refused.
  Where the caller gave no ceiling and named a budget, Anthropic and Bedrock
  are sent one of the budget plus 4,096 for the answer, up to 32,000 in all, so
  the answer always has room after the thinking, and a budget past what that
  allows is moved down to fit. Bedrock, whose own default is the model's
  largest and which a budget could pass, notes the ceiling as
  `cap.ceiling_supplied`. Under the `interleaved-thinking` beta a budget may
  run past the ceiling, and it is not cut to fit it.
- Anthropic's `thinking: {type: "adaptive"}` was read as thinking off, with a
  loss noted, so current Claude models answered without thinking. It is now
  sent to Anthropic and Bedrock as adaptive. A backend without that mode reads
  it as a thinking budget the model sets for itself.
- A server tool's streamed input from Anthropic (`mcp_tool_use`,
  `server_tool_use`) became a nameless tool call for the client to run, and the
  block itself kept an empty input. The block now carries its input, as the
  unstreamed answer does, and a stream that ends inside such a block still
  delivers it, with a note that it did not arrive whole. Through the Claude CLI
  the same input was written into the previous call's arguments, and no longer
  is.
- A function call that a Responses gateway delivered whole on the item that
  closes it, with no argument deltas, arrived with no arguments. What the
  deltas did not carry is now taken from that item, and a call the stream never
  announced arrives whole.
- OpenAI backends dropped every part and item that the OpenAI wires carry
  verbatim, even from a client speaking that same wire: a chat `file` part, a
  Responses `input_file` or `input_audio`, a replayed `mcp_approval_response`
  or `item_reference`. A document never reached the model, and the hosted MCP
  approval flow could not finish. These now reach an OpenAI backend whole; a
  compatible dialect still reports them as dropped.
- The Responses wire never handed a client a reasoning item's
  `encrypted_content`, streamed or not, and named every reasoning item itself.
  A client that keeps nothing on the backend therefore could not carry its
  reasoning into the next turn, and on a Claude backend the thinking it
  replayed arrived unsigned and was stripped, so the tool turn after it was
  refused. Each reasoning item now carries its own signature, a signature with
  no summary opens an item of its own, and an item keeps the id the backend
  gave it. A signature from another vendor is stripped before the call; see
  the entry on signatures below.
- The Responses stream dropped every item the wire does not model — a hosted
  tool server's tool list, call or approval request, a hosted search — although
  the unary path carried them, so a streaming agent never saw the approval it
  was waiting on. They are streamed now, and on both paths they are written as
  the backend wrote them rather than through the fields goose models, which had
  lost the server label, the tool list and the rest.
- `reasoning.effort: "none"` switched extended thinking on, although that is
  exactly what it asks to avoid, and a `reasoning.summary` on its own did the
  same. Now only an effort other than `none` does. The effort still reaches an
  OpenAI backend as it was sent.
- A Responses `custom` tool (freeform text or a grammar), or any other tool
  kind the wire does not model, became a function that takes nothing, so the
  model called it with `{}` and its grammar was lost. It now travels as written
  to an OpenAI Responses backend, after the function tools and with its numbers
  untouched, and any other backend reports it as not sent. A tool choice that
  names such a tool, or that is made over tools which are all of that kind,
  travels with them rather than being dropped for having nothing to govern.
- An `input_image` that names an uploaded file instead of a URL was decoded as
  an image with neither, and reached the backend as `image_url: ""` with
  nothing said. It is carried as the file reference it is now, and a backend
  that cannot resolve it reports the loss.
- A Gemini thought signature on a function call or a text part was dropped in
  both directions, so Gemini 3 refused the next turn of any tool loop, and a
  part carrying nothing but a signature reached other clients as an unknown
  block. On the Gemini wire each part now keeps its signature from the backend
  to the client and back, and so does an image Gemini generated, which it will
  not edit on the next turn without one. The OpenAI chat wire carries a call's
  signature where Google's own OpenAI-compatible endpoint does, in
  `tool_calls[].extra_content.google.thought_signature`, streamed or not. The
  Anthropic, Responses and Ollama wires have no field for a signature on a
  call or a text part, and report `drop.signature`. A call in the current turn
  that comes back unsigned is sent to Gemini with the placeholder Google
  documents for that case, with a note that the model's reasoning before it
  was not carried, and the loop goes on. A signature-only part is an empty text
  part. The other wires leave it out and report that its signature did not
  travel. A signed part goose does not model, such as code the model ran, is
  carried whole, signature and all.
- A Gemini `thinkingConfig` lost its `thinkingLevel` without a word and always
  asked for the thoughts back, even Gemini to Gemini. The level now reaches a
  Gemini backend and is reported as not sent to any other, and thoughts come
  back only when the client asked for them with `includeThoughts`.
- The OpenAI chat stream numbered a tool call by the upstream's content block,
  so a Claude answer that spoke before calling a tool sent its only call as
  index 1, which OpenAI clients file in the wrong place or fail on. Calls are
  numbered from 0 in the order they open in each choice now.
- The Anthropic wire labelled every cache write a one-hour write when the
  backend had not said how its writes split, so they were priced at the dearer
  rate. The split is written only as the backend measured it now, a one-hour
  share included.
- The earlier fix for a refusal flattened into text and a tool call whose
  arguments did not survive reached only the unstreamed Anthropic answer. The
  Anthropic stream, the Gemini wire and both Ollama endpoints now report
  `drop.refusal`, once an answer however many deltas the refusal arrived in,
  and the Anthropic and Responses streams, the Gemini wire and the Ollama chat
  wire report `drop.tool_call` where arguments were replaced with `{}`.
- Every streamed Bedrock call failed on its first frame. The event-stream
  reader checked both of AWS's checksums as CRC-32C, where AWS uses the IEEE
  CRC-32; the tests passed only because they framed their own messages the
  same wrong way. It reads what AWS sends now, and a test checks it against
  AWS's own reference frame.
- An EKS pod identity could not authenticate. Its agent hands the pod only a
  token file, `AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE`, which nothing read, so
  the agent refused and the chain went on to sign with the node's instance
  role. The file is now read at every refresh, so a rotated token is followed.
- `AWS_EC2_METADATA_DISABLED` was honoured only as lower-case `true`, where the
  AWS SDKs take any case, so `True` or `TRUE`, as tooling often writes them,
  left the node's role reachable.
- A profile named in `options.profile` or `AWS_PROFILE` that could not be read
  — a missing one, or an SSO or assume-role profile with no key pair — was
  passed over for the container or instance role, and requests signed as
  somebody else. A profile named in `options.profile` was also passed over for
  `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. A named profile, `default`
  included, is now used or refused and never passed over for the runtime role.
  One named in `options.profile` also comes ahead of keys in the environment,
  as the AWS SDKs put a profile the program names. `AWS_PROFILE`, the
  environment's own, still yields to those keys and to a web identity, as it
  does in the SDKs. The example configuration no longer names one.
- `GCE_METADATA_HOST` was used as a URL, but Google defines it as a host and an
  optional port, so any value spelt Google's way made every Vertex token fetch
  fail. It is read as Google's own libraries read it now, by the binary and by
  `vertex.MetadataToken`, and a full URL still works.
- A Bedrock tool result with nothing in it, such as a tool that printed
  nothing, was sent as `content: null`, which Converse refuses, failing the
  whole turn. It now says `(no content)`.
- A tool loop's history sent to Bedrock with no tools was refused, for instance
  when closing the loop with a summary, because Converse wants a `toolConfig`
  beside any tool blocks. The tools the history names are now declared again
  with no schema, and a `drop.tool_definition` note says the model may call one
  the request did not offer.
- A stdio tool server that stopped reading its input held the write that was
  feeding it, and with it every later call to that server whatever its
  deadline, while the refresh timer waited on it and so never reached the
  servers listed after it. A write now gives up when its caller does, the
  server is treated as gone and started afresh for the next call, and each
  server is refreshed on its own.
- A call that gave up while it waited behind another for a tool server was
  sent anyway once its turn came, so a tool with side effects ran for a caller
  already told it failed. It is no longer sent.
- Connecting to a tool server over the older SSE transport ignored the caller's
  deadline and the server's `timeout`, so a server that took the connection and
  never named an endpoint held startup for ever. Opening the stream is held to
  both now.
- A tool server that went away stayed away for the life of the process. A
  stdio server that exited, an SSE stream that ended, and a streamable server
  that restarted or let its session lapse each failed every later call, and
  their tools stayed advertised. Such a server is now dialled and greeted again
  for the next call. While it cannot be reached its tools are withdrawn, and
  they return with the next refresh that reaches it. An SSE stream is seen to
  end whatever was said on it last: a frame no call was waiting for, such as a
  log line or the answer to a call whose caller had already given up, is
  passed over rather than held for the next call. A `404` from its endpoint is
  taken to mean the stream has gone, so the call is sent again on a fresh one.
- One answer larger than the transport's ceiling took the tool server out of
  the proxy for good, and left a stdio child running with nobody reading it.
  That answer now fails only its own call, and the server is started afresh
  for the next one.
- A tool from a stdio or SSE server with an `x-mcp-header` annotation goose
  cannot read was listed to current-revision clients and then refused on every
  call from them, even on `goose mcp`, where nothing is mirrored. It can be
  called wherever nothing is mirrored now, and it is left out only of the list
  served to a client that mirrors.
- A `subscriptions/listen` stream held a `server.max_concurrent` slot for as
  long as its client kept it open, so a few open agent sessions shut out every
  LLM request. The slot is given back once the stream is acknowledged. Open
  streams are bounded on their own now, at 256, and one past that is refused
  with 503.
- JSON-RPC ids and progress tokens were read as floats, so a number past 2^53
  came back rounded and the client could not match its answer, and a string id
  that looked like a number came back as a number in a subscription's
  acknowledgement and notifications. Both come back exactly as they were sent
  now.
- On `goose mcp`, a `notifications/cancelled` arriving right behind its request
  could be handled before the call was registered, and was dropped, so the
  withdrawn call ran and was answered. It is registered before the next frame
  is read now.
- Stopping goose closed the usage ledger and printed the summary while the
  calls it had just cut off were still unwinding, so a stream still running ten
  seconds after SIGTERM, or an `ask` in flight when an MCP client closed the
  pipe or sent a signal, was billed and never booked. `mcp.Manager.ServeStdio`
  now waits for its calls, for a bounded time, and the binary waits up to five
  seconds more for them to book what they spent, and says so on stderr when
  one does not.
- goose read `.goose.toml` or `goose.toml` from whatever directory it was
  started in, ahead of `~/.config/goose/config.toml`. For the proxy that was
  whatever checkout it ran in, and for `goose mcp` whatever project its client
  had open. Either way the file ran its `[[mcp]]` commands and sent the
  operator's keys wherever it pointed. Neither mode reads such a file now
  unless `-c` names it. A skipped file is named on stderr with the `-c` that
  would read it, and a start that finds no file says where one goes, rather
  than only that no providers are configured.
- A vendor's usual variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and the
  rest) was sent to whatever `base_url` a provider named, so a file could point
  a kind at a host of its own and collect the key. It goes only to the vendor's
  own host over TLS now, or to a base URL typed with `-u`. OpenAI's own hosts
  include its regional ones, such as `eu.api.openai.com`, and Azure's include
  its government and China clouds. An Azure `options.resource` is judged by
  the host it is spliced into, so one written to point elsewhere does not take
  the key with it. Anywhere else a provider gets a key only from `api_key` or
  `api_key_env`, and a variable held back is named on stderr.
- `server.api_key_env` naming a variable that was not set was read as no key
  at all, which left a loopback proxy open to anyone who could reach it. It is
  an error at startup and under `-check` now, naming the variable. `goose mcp`
  binds no port and does not ask for it. A provider's `api_key_env` naming an
  unset variable is named on stderr, and that backend's calls carry no key
  until it is set: an MCP client starts `goose mcp` with little of the
  operator's environment, and one backend without its key is no reason to
  serve none.
- `api_key` and `api_key_env` on Bedrock and Vertex, a key on an agent CLI with
  no `api_key_env_name` option to pass it on, and `max_retries` on an agent CLI
  were accepted and never used, while the banner said "key set". They are
  refused now like every other setting a kind does not read.
- `goose -check` created the usage ledger to validate it, so a config whose
  directory the checking user could not write to failed `-check` even though
  nothing would be written there until a call was served. It no longer opens
  or creates the file to check the path, and still refuses, without writing
  anything, a log whose directory does not exist or that is itself a
  directory.
- A provider's `base_url` with no scheme, such as Ollama's own `host:port`
  form, passed `-check` and then answered every call with a 400 the caller
  was blamed for. It now fails `-check` and startup, naming the provider and
  the setting.
- A provider that failed to build reported the library's bare error, an API
  kind and the library's own field, as in `invalid_request: bedrock: region is
  required`, naming neither the `[[provider]]` entry nor the setting at fault.
  The message now names the provider and the settings that supply what was
  missing — `options.region or AWS_REGION`, `options.project or
  GOOGLE_CLOUD_PROJECT`, `options.resource or base_url`, `options.account_id
  or base_url` — and for Vertex the credentials setting.
- A relative `server.tls_cert`, `server.tls_key`, a provider's
  `options.credentials_file`, an `[[mcp]]` `command` or `dir`, or an agent
  CLI's `base_url`, `options.dir` or `options.temp_dir` was read against
  whatever directory goose was started in. A home config with such a path
  broke as soon as `goose mcp` ran from a client's own working directory, and
  `goose mcp` opened in a project ran that project's `./srv.sh` rather than
  the one beside the operator's file. Each now resolves beside the config file
  that names it, as `usage.log` already did, and a program inside its `dir`
  where one is set; a bare command name is still found on `PATH`.
- An argument after the flags was dropped along with every flag behind it, so
  `goose -c goose.toml mcp` served the HTTP proxy and wrote its banner onto the
  MCP client's pipe, and `goose -v check` started serving instead of checking.
  A stray argument is an error now, and `mcp` anywhere but first says where it
  belongs.
- A model name went into an agent CLI's argument list after `--model` without a
  check, and it can come from a request body, so whether one beginning with a
  dash was read as a value or as another flag was left to each binary's own
  parser. Such a name is refused before the binary is run now.
- The agent CLIs declared that they took a thinking setting, but none of them
  is given one, so a caller's request for extended thinking vanished without a
  word. They no longer declare it, and the drop is reported like any other.
- A tool choice made over tool servers alone — `mcp_servers` on the Anthropic
  wire, a hosted `mcp` tool on Responses — was dropped before sending, with a
  note saying no tool definition had reached the backend. A caller's `none`
  left the model free to run a server's tool, and `required` forced nothing.
  The choice now travels with the servers, and the Anthropic provider sends it
  beside a tool server alone.
- A call that waited at a backend's concurrency gate until its time ran out was
  reported as shed at once and booked as rejected, so a queue that timed out
  read as a gate refusing load. It is now said to have ended while waiting and
  booked as cancelled, with the wait on its row, and the chain stops there,
  since its budget is spent. The wait also ignored the attempt deadline, and a
  call let through late in it was then given the whole deadline again for the
  send. The wait and the send now share the one deadline,
  `server.attempt_timeout` or `llm.Limits.Timeout`, and the row's time covers
  both. A send that had to wait for its slot, and then ran out of what the wait
  left it, is not counted against the backend, whose time went on the calls
  ahead of it at the gate; one given its slot at once still counts.
- A loss an adapter meets while building its request — a participant's name,
  a thinking block without its signature, an image the backend cannot take, a
  field of the Anthropic wire it does not model — was held to the policy only
  once the answer was back. Under `strict` the request was sent and billed
  before it was refused, and the next link was never tried. Such a loss is now
  refused before anything is sent and booked as a rejected attempt that cost
  nothing, and the chain walks on.
- The proxy's own losses in decoding a request, such as a thinking setting it
  could not read or a role it did not know, were never held to the policy:
  under `strict` the request was sent without them and the answer served. Such
  a loss is now refused in the caller's own wire before any backend is tried.
  These notes also reach every attempt's ledger row, where an unstreamed call,
  or a stream that brought no content, used to leave them out.
- The proxy's losses in encoding an answer for the caller's wire were only
  listed in `X-Goose-Diagnostics`, and the answer was served without them under
  `strict` or a `deny` naming them: an image on the OpenAI chat or Ollama wire,
  reasoning where the wire has no field for it, a tool result on an assistant
  turn, a stop reason the wire cannot spell. An unstreamed answer is now
  refused whole in its place, and the call stays on the ledger as served and
  billed. A stream is refused where the loss is met, in the wire's own error
  event, and nothing after it is written.
- A loss an encoder marked fatal was served anyway, with its code only in the
  header: an answer made only of tool calls on `/api/generate`, or more
  candidates than the wire carries. It now fails the call under any policy, as
  a fatal loss everywhere else already did.
- Under `redact_upstream_errors`, goose's own refusal of a loss, or of a
  capability a backend lacks, was reworded as the upstream failing whenever it
  named the backend it refused, so a caller could not tell a request it could
  fix from a fault it could not. goose's own refusals now keep their code and
  message; a backend's words are still replaced.
- A loss refused in an answer said the request had not been sent, although it
  had been sent and billed, so a caller that counts only sent calls as billed
  would take such a call as free. The refusal now says the request was sent,
  and which backend and model served it — on a streamed answer as well as an
  unstreamed one, which had been left naming neither.
- A signature went to whichever backend served the next turn, whoever had
  minted it. OpenAI's `encrypted_content` reached Claude as a thinking
  signature, Claude's or Gemini's reached OpenAI as `encrypted_content`, and
  Claude's reached Gemini as a `thoughtSignature`. Each backend refused the
  stranger's signature with a 400, and the chain stopped there. Every signature
  now records the vendor family that minted it. Claude is one family, whether
  it runs on the Anthropic API, Bedrock, Vertex or the Claude CLI; OpenAI and
  Azure OpenAI are a second; and Gemini, on its own API or on Vertex, a third.
  A signature from another family is stripped before the call and reported as
  `drop.signature`, and a strict policy refuses that link instead, so a chain
  hands the turn to the family that can verify it. A client's wire has no field
  for who minted a signature, so one from another family goes out to the
  client marked with its family and is read back the same way. On the Gemini
  wire, whose field is bytes, the mark is base64 like the signature it carries,
  so Gemini's typed SDKs can still read the answer.
- A Gemini-wire stream held a Claude-family thinking block's text across an
  intervening tool call without flushing it first, so the block that was
  eventually joined to a signature by matching ids picked up a later,
  unrelated thinking block's text as well as its signature. The Gemini
  encoder now flushes a pending, unsigned thinking block before a tool call,
  as it already did for ordinary content.
- Under `strict`, the Anthropic Messages and OpenAI Responses stream encoders
  wrote a buffered tool call to the wire from their deferred flush before
  checking whether that call's own signature loss had just been refused, so
  the refused call reached the client — and, on the Responses wire, was
  echoed again in the closing snapshot — immediately ahead of the error
  event. Both encoders now diagnose a buffered tool call before writing it,
  and write it only once the stream is confirmed not refused.
- OpenAI Responses refused thinking it could neither look up nor decrypt:
  Claude's or Gemini's once their signature was stripped, or a summary a client
  kept without its id. goose sent it as a reasoning item with no id and no
  encrypted content, and the refusal ended a chain that should have gone on.
  It is left out now and reported at info as `drop.thinking`.
- A tool result from the Anthropic wire, the Responses wire or most chat
  clients reached Gemini with no function name. Those wires name only the call,
  and Gemini refuses a result without a name, so no tool loop from another wire
  got past its first call. The name is now taken from the call the result
  answers, and a result that answers no call in the conversation is refused
  before it is sent.
- The Responses wire threw away the id of every reasoning item a client
  replayed, so an item OpenAI had stored could not be found again. OpenAI now
  gets back the id it issued. goose's own names for items it made up now start
  `rs_goose_`, and only those are dropped. A streaming client never received an
  item OpenAI stored with neither a summary nor encrypted content; it arrives
  now, as it did without streaming. The Gemini wire no longer writes a thought
  with nothing in it.
- Claude refuses extended thinking beside a temperature other than 1, a
  `top_k`, a `top_p` under 0.95 or a tool choice that forces a call, and other
  wires allow each of them: a Gemini client sends a temperature with its
  `thinkingConfig`, and a Responses client sends `tool_choice: "required"` with
  a reasoning effort. goose sent them together to Anthropic and Bedrock, and
  the refusal stopped the chain. Beside a forced tool call, thinking is now
  left off and the loss is noted — including a choice forced over a tool
  server alone, with no function tool beside it, which was missed the first
  time and still reached Anthropic with thinking on. Beside thinking, the
  temperature and `top_k` are left out and `top_p` is raised to 0.95, each
  with a note.
- An Anthropic-wire image held in the Files API (`source: {type: file,
  file_id}`) was read as an image with neither bytes nor an address. It is
  carried whole now and reaches an Anthropic backend with the caller's
  `files-api` beta. Everywhere else it is reported as a loss.
- Bedrock hands back redacted thinking but did not say it takes it back. The
  block was stripped before the next turn, and Claude refused a tool loop that
  had lost the thinking it opened with. It is replayed to Bedrock now. A
  thinking block goose itself had to strip, or that never carried a
  signature, was still replayed as `reasoningText` with an empty signature,
  which Claude also refuses; such a block is left out of the replay instead.
- Claude refuses thinking beside a tool loop whose last assistant turn does
  not open with its thinking, and Anthropic and Bedrock were sent thinking on
  regardless: a turn whose thinking had been dropped unsigned, stripped before
  it arrived or never written resumed with a setting Claude will not honour,
  and the 400 stopped the chain. That held whether the tool result came in a
  user message, as a tool message as the OpenAI wires send it, or with more
  from the user after it. The check now reads the turns as each backend
  receives them, and thinking is left off for that turn, with the loss
  reported, the same as when a forced tool choice rules it out.
- A Responses client that interleaved function tools, `mcp` tools and tools
  goose carries whole had them regrouped with nothing said: function tools
  first, then tool servers, then the rest. The order still cannot be kept, but
  the reorder is noted now.
- A Responses `tool_choice` naming a hosted `mcp` server was refused as
  provider-specific, even on the OpenAI route it was written for. It is sent
  now.
- An agent CLI stopped because the call ran out of time, whether the caller's
  deadline or the provider's own `timeout`, or because the caller went, was
  reported as the binary failing: `upstream`, a 502, and an error on the
  ledger. It is now the timeout or the cancellation it was, and a timeout the
  provider set says so. That held only once the binary had run: a context
  already done when a slot came free was still given one, and a context that
  ended just as the binary was about to start was reported as the binary
  itself failing to spawn. Neither takes a slot or reads as a transport
  failure now.
- An EKS pod given its service account's role through IRSA signed as the node.
  Nothing read `AWS_ROLE_ARN` or `AWS_WEB_IDENTITY_TOKEN_FILE`, so the chain
  went on to the instance role wherever the pod could reach instance metadata,
  and failed with an error about metadata where it could not. The role is now
  assumed through STS with the token in the file, which is read again at every
  refresh. As in the AWS SDKs, it comes after keys in the environment and ahead
  of the shared file and the runtime role. A web identity that fails is an
  error rather than a reason to borrow the node's role. STS is asked at the
  provider's regional endpoint, or at the one `AWS_ENDPOINT_URL_STS` names,
  over HTTPS or to a local address.
- `bedrock.ContainerCredentials` given neither URI returned a source that
  panicked when called, and `FirstWorkingCredentials` did not pass over it, so
  code handing it the AWS SDK's own variables crashed on its first call on any
  host outside a container. Such a source now fails with an error, and the next
  one answers. `ContainerCredentialsFromFile` does the same.
- A Vertex key named by `options.credentials_file` or
  `GOOGLE_APPLICATION_CREDENTIALS` whose exchange failed, a revoked key say, was
  passed over for the metadata server's default service account, which then
  signed for the key's project as somebody else. A named key is now used or
  refused, as in Google's own libraries, and the metadata server is asked only
  when no key is named.
- An integer argument mirrored into an `Mcp-Param-` header was read as a float,
  so one past 2^53 was mirrored as its neighbour: goose refused a client that
  mirrored it exactly, with `400` and `-32020`, and sent an upstream a header
  that disagreed with the body. It is mirrored as the body wrote it now, so a
  client that rounds such a number in its header is the one refused.
- Passthrough handed a caller's own key to the backend whichever vendor had
  minted it. A Gemini client's `countTokens`, which the `google` format
  forwards, reached an Anthropic passthrough with its Google key as
  `x-api-key`, where the routed path withheld the same key. Passthrough now
  follows the routed path's rule: a key on `x-api-key` goes only to an
  Anthropic backend and one on `x-goog-api-key` or `?key=` only to a Google
  one, and a key held back is reported as `policy.client_key_withheld` in
  `X-Goose-Diagnostics`.
- A passthrough call that failed before it was sent, such as one whose token
  could not be minted, was booked as sent and counted in
  `goose_upstream_unmeasured_total` as a call whose usage the backend had not
  reported. It is now booked as unsent, with no usage anomaly. A caller that
  hung up, or whose call ran past its deadline, on the passthrough door was
  booked and counted the same way, as an upstream error charged against the
  backend, where the routed path already knew a departed caller from a
  backend that had answered nothing wrong; passthrough now books the same
  distinction.
- Passthrough forwarded the caller's decoded path into a backend's `base_url`,
  so an encoded `..` (`%2F` or `%2E%2E`) could climb back out of it on the
  operator's own credential, and an encoded `?` (`%3F`) in the path could open
  a second query string once the joined URL was reparsed. The path is now
  forwarded still escaped, and a percent-encoded `.` or `..` element is
  refused with `400` rather than sent on; a literal one never reached this
  far, cleaned away by the listener's own routing first.
- With `cors = true`, a passthrough answer from an upstream that sets its own
  CORS headers — Ollama's default handling, or Anthropic's own browser access
  — carried two `Access-Control-Allow-Origin` values, which the Fetch
  standard never treats as a match, so every such response failed in the
  browser. Passthrough now drops the upstream's copies of the four headers
  goose sets itself — `Access-Control-Allow-Origin`, `-Allow-Headers`,
  `-Allow-Methods` and `-Expose-Headers` — so a browser sees one of each. Any
  other CORS header the upstream sends still passes through.
- `cors_origin = "null"` let any site drive a keyless proxy and read its
  answers, since a browser sends that origin for a sandboxed frame and any page
  can create one. It now needs `server.api_key`, as the wildcard does, and the
  proxy will not start without it.
- `goose mcp -check` built and listed the HTTP listener, which `goose mcp`
  never serves: it printed the proxy's endpoints, and could refuse a file over
  a rule only the listener has, such as a wildcard CORS origin needing a key.
  It checks what `goose mcp` runs now, and lists its tools.
- Stopping the proxy waited up to ten seconds for any open stream, an MCP
  client's idle listen stream included, and said nothing while it did, and a
  second Ctrl-C was ignored. It says what it is waiting for now. A second
  signal cuts the open requests off at once, and goose then waits only for
  them to book what they spent, up to five seconds.
- Under `redact_upstream_errors` the `ask` tool repeated a backend's refusal
  word for word, so a proxy shared with people who do not hold the account
  handed them what the listener's other endpoints hold back. It gives the
  provider, the kind of failure and the status now, and what it holds back goes
  to the debug log, as the listener's does. A later change to `ask`
  reworded goose's own refusals — a loss the policy would not take, a
  capability a backend lacks — as the upstream failing too, once they carried
  the backend's name; `ask` keeps such a refusal's own code and message again,
  and a backend's own words are still replaced. The `models` tool and
  `GET /ready` handed a caller a backend's own words under this setting all
  along — an account's balance on a model listing, or an upstream's address
  when a readiness probe's dial never landed — while the vendor endpoints and
  `ask` already withheld theirs; both now report the kind of failure instead.
- `ask` with no `model`, on a route that pins none, sent an empty model; with
  the README's own configuration, the tool's documented form got back only
  Ollama's `model is required`. The provider's `model` is used where one is
  set. Otherwise, for a backend that addresses models by name, `ask` now says
  to pass `model` and sends nothing.
- A relative `[usage] log` was opened in the working directory. `goose mcp`
  reads the operator's own file wherever its client starts it, so it left a
  ledger in each project it was opened in, and would not start where that
  directory was read-only. The path is resolved beside the file that names it
  now. `-usage-log` is still relative to where it is typed.
- A link refused before anything was sent hid a failure that had been sent. A
  later link's breaker skip or capability refusal replaced an earlier link's
  failure, handing back one link's answer with another's error, and a
  last-resort link that went, failed and may have billed was reported as the
  refusal of the link before it. The failure that reached a backend is now the
  one the caller gets, with its own answer and what it counted.
- A streamed answer that failed after its first content, such as Anthropic's
  `overloaded_error` once the text had begun, told the breaker the backend was
  healthy as soon as the stream was handed over, and every stream handed over
  reset the failure count. The breaker now gives its verdict when the stream
  ends. A probe that is a stream holds its slot until then, and gives it back
  if it is abandoned without being closed.
- An answer from a backend that does not stream, served as a stream, named no
  backend unless its provider had named itself, so neither the
  `X-Goose-Provider` header nor a refusal of it said which link had served it.
  One with no candidate — only usage, or only a block — read as a stream cut
  off, was counted against the backend, and was sent and billed again to the
  next link. Both now read as the unstreamed answer does.
- Bedrock answers a model the account has not been granted, or cannot
  subscribe to through the Marketplace, with a 403, which was counted against
  the whole provider: a caller naming one such model a few times opened the
  breaker for every model the account is served. That refusal now reads as a
  model this backend does not serve, and the chain walks on.
- Ollama takes a tool result only as a tool message, and a result inside any
  other turn, as the Anthropic and Gemini wires carry one, was dropped with
  nothing said, even under `strict`, so every such tool loop reached the model
  without its result. Each result is now sent as a tool message, named by the
  call it answers where it does not name its tool, and a turn that held
  nothing else is not sent empty after it. The OpenAI chat providers sent the
  results but followed them with that empty user turn; they no longer do.
- A thinking block streamed to an Anthropic client stayed open while a tool
  call was held back, so the next block's text and signature were written into
  it: one block, two signatures, and a replay Anthropic refused. A signed
  block, or one the backend names differently, now closes before the next
  opens.
- On the Anthropic and Responses streams a part whose loss `strict` refuses
  was written first and refused after, so the client got exactly what was
  refused ahead of the error. The loss is now checked before any of the part
  is written.
- Redacted thinking was dropped without a word on the OpenAI chat wire, on
  Ollama's generate endpoint and on its chat stream, and so was a tool result
  on either Ollama stream. Each is now noted, and `strict` refuses it.
- Ollama's generate endpoint noted a lost tool call only once the stream had
  ended, so `strict` let the text after it out first. The loss is now noted in
  the frame the call arrives in, and an answer that was nothing but calls is
  still refused outright.
- The Gemini wire held unsigned thinking with no ceiling and rebuilt it with
  every delta, so tens of thousands of small deltas cost seconds and gigabytes
  before a word was sent. What waits for a signature is now built up once and
  bounded, at thirty-two megabytes a stream, and a block that outgrows it goes
  out unsigned, with the loss said.
- An Anthropic request read each nested tool result again from the top, so one
  request a few hundred kilobytes long could hold a core for seconds. A tool
  result inside another is now refused, as the API refuses it.
- The Responses stream rebuilt its held text, refusal and summary with every
  delta, at a cost that grew with the square of their number. It now builds
  each once.
- The Responses wire reported a stop it has no spelling for as completed and
  said nothing, where every other wire notes it and `strict` refuses it. It
  now does the same. An unstreamed answer that was cancelled is reported as
  `cancelled`; a stream has no event that ends a response that way, so there
  it is noted like any other stop the wire cannot spell.
- A context-window cut-off reached an Anthropic client as `max_tokens`,
  telling it to raise a ceiling where it needed a smaller context. It is
  `model_context_window_exceeded` now, Claude's own name for it.
- An OpenAI chat `tool` or `function` message's `name` was read as a
  participant's name as well as the result's, and every backend but the
  OpenAI chat ones has no field for one, so `strict` refused ordinary tool
  loops. The name now belongs to the result alone.
- An assistant image reached an Anthropic client unstreamed as an `image`
  block, which the Messages API has no answer block for and the stream already
  dropped. The unstreamed answer now drops it too and says so.
- With no `server.api_key`, the rate limiter gave each credential a caller
  presented an allowance of its own, though nothing had proven it and a
  backend authenticating with the operator's key never saw it, so a caller
  writing a new made-up bearer each request was never throttled. A keyless
  proxy now meters every caller by the address it came from, the one
  `forwarded_header` names where that is set, so callers behind one address
  share an allowance whatever keys they carry.
- `forwarded_header` was read for the first entry of its first line, which is
  what the caller wrote, so behind a front proxy that appends — on the same
  line, or on a line of its own as HAProxy's `option forwardfor` does — the
  caller chose its own bucket. The lines are now read as one list, and the
  last entry, the one the nearest hop added, is the one that counts.
- A passthrough body over `max_body_mb` was forwarded until the listener cut
  it off: the backend was left with a truncated request, the caller got `502`
  (with redaction on, "the upstream failed to serve this request", which SDKs
  retry), and the ledger booked an error against the backend. A body that
  announces its length is now refused with `413` before anything is sent, and
  one that does not is answered `413` when it is cut off and booked as goose's
  refusal.
- Passthrough forwarded every method, `TRACE` and `CONNECT` included, with the
  operator's credentials attached, so an upstream, or anything between, that
  answers `TRACE` would have echoed those credentials to the caller. It now
  forwards only `GET`, `HEAD`, `POST`, `PUT`, `PATCH`, `DELETE` and `OPTIONS`,
  and refuses any other method with `405` and an `Allow` header before
  anything is sent.
- A route that set only `allow` or `deny` under `server.strict = true` was
  served leniently: naming any policy counted as choosing one, and the
  strictness it never mentioned fell to off. Such a route keeps the server's
  strictness now, and only `strict = false` opts out.
- `-p` replaced the file's providers but kept its `server.passthrough`, so
  `goose -check -p ollama` with a home config naming a passthrough failed over
  a provider it had just dropped. `-p` now drops the passthrough as it drops
  the routes.
- `goose mcp` refused a file whose `server.tls_cert` or `tls_key` was not on
  the machine, although only the listener reads them. It no longer asks for
  them.
- An `[[mcp]] url` with no scheme or host passed `-check` and failed at
  startup. It is refused under `-check` now, naming the server but not the
  URL, which may carry its token.
- A `base_url` with a query or a fragment took every call's path inside it, so
  `http://host:port?x=1` sent each call to `/?x=1/api/chat`. It is refused
  now, and `-u http://` says the host is missing rather than the scheme.
- A Bedrock or Vertex region, from `options.region`, `AWS_REGION`,
  `AWS_DEFAULT_REGION` or `GOOGLE_CLOUD_LOCATION`, was spliced into the host
  as written, so `x@127.0.0.1:18409/#` sent the call and its credential to
  another host. A region with anything but lower-case letters, digits and
  hyphens is refused now, by the binary and by `bedrock.New` and `vertex.New`
  for an embedder's own `Region`. An Azure `options.resource`, and
  `azure.Options.Resource` in the library, is spliced into the host the same
  way, and is now held to letters, digits and hyphens.
- `ask` answered an empty prompt, or arguments it could not read, with a
  JSON-RPC internal error, where every other refusal of the tool was a failed
  tool call the model could read. They are failed tool calls now.
- The `models` tool's notes said a model list could not be read without saying
  whose. Each now names its backend.
- `goose mcp -v` logged nothing at all. It now logs each call to goose's own
  tools on stderr; calls to the tool servers goose proxies are still not
  logged.
- Negative durations — `attempt_timeout = "-5m"`, `breaker_cooldown`, a
  provider's or tool server's `timeout` — passed `-check` and were swapped for
  the default without a word, while a negative limit beside them was refused.
  They are refused now.
- A tool server that would not start was named twice (`mcp files: mcp files:
  …`). It is named once.

### Added

- `server.attempt_timeout` and `server.max_response_mb` bound one attempt and
  the body it may read back. `make fuzz` runs every target, reports the ones
  that failed, and runs in CI.
- `server.max_frame_mb` raises the ceiling on one line of an upstream stream,
  where a streamed image arrives whole.
- `llm.Stream.Enforce` holds a stream to a loss policy on its way to a
  destination described by an `llm.Renders`, and `llm.Stream.Check` holds to
  that policy what an encoder loses in writing the stream out. Every built-in
  codec calls `Check`, and a codec written outside goose should too.
  `llm.Capabilities.ClientKey` says a backend sends the caller's own credential
  in place of the operator's.
- `llm.Thinking.ID` is the backend's own name for a reasoning block,
  `llm.Text.Signature` and `llm.ToolUse.Signature` carry Gemini's thought
  signatures, `llm.ThinkingConfig` gained `Adaptive` and `Hidden`, and
  `llm.Usage` and `llm.RawUsage` gained `CacheWrite1h`.
- `httpx.Options.Host` is the listener's host, with or without its port. It
  turns on the loopback `Host` check for a server with no `APIKey` and no
  `ForwardedHeader`.
- `bedrock.ContainerCredentialsFromFile` reads the container endpoint's token
  from a file at every refresh, which is what an EKS pod identity is given.
- `bedrock.WebIdentityCredentials` assumes a role through STS with a token read
  from a file at every refresh, which is what an EKS service account's role
  (IRSA) is given.
- `route.Target.FromCall` says the target's model is the one the call asked
  for, passed on because neither the target nor its rule named one.
- `goose.WithDiagnostics` hands a call what was noted of the request before it
  reached the client, such as a proxy's decoding of its caller's wire. The
  notes are held to the call's policy before any link is tried, and they travel
  on every attempt's row and on the answer.
- `llm.Attempt.Check` refuses, unsent, an attempt whose notes include a loss
  its policy will not take. An adapter calls it once its request is built and
  before sending.
- `llm.Renders.Raw` names the vendors whose carried blocks a destination writes
  out whole, and `llm.Renders.Unrendered` lists the parts of an answer that a
  declared destination has no field for.
- `llm.Response.Policy` is the loss policy the answer was held to, so a proxy
  can hold its own translation of the answer to the same policy.
- `llm.Family` names the vendors whose signatures verify one another's.
  `llm.Thinking`, `llm.Text`, `llm.ToolUse` and `llm.Image` gained `Signer`, the
  family that minted their signature, and `llm.Image` gained `Signature`.
  `llm.Capabilities.Signer` is the family whose signatures a backend verifies;
  a backend that leaves it empty is sent every signature, as before.
- `llm.Request.Betas` carries the vendor feature flags a caller asked for. The
  Anthropic wire fills it from `anthropic-beta`, and the Anthropic adapter sends
  it beside its configured `Betas`. `anthropic.Options.NoClientBetas` withholds
  it, and the fields it opens, for a surface that refuses flags it does not
  know. In a file, a provider's `client_betas = false` does the same.
- `agentcli.Options.MaxQueued` bounds how many calls may wait at an agent CLI's
  gate at once.
- `llm.Precheck` lets a provider say what its own request will lose, or that
  it will not send it at all, without spending anything to learn it, so a
  request that was always going to be refused is refused before it queues for
  a `Gated` provider's slot rather than after. Its method is
  `Precheck(a *Attempt) (Diagnostics, error)`.
- `goose.OutcomeFor` is the ledger outcome a failure is booked under, so a
  caller booking a row for a call that did not go through the chain, such as
  passthrough, reads it the same way the chain does.
- `openai.Options.ResponsesBody` is `Body` for the Responses shape alone, for
  a field that belongs there and nowhere else, which is how `azure.New` names
  the deployment in a Responses call on the dated surface.
- `mcp.Client.Truncated` says a tool server's catalogue was longer than goose
  will hold, so that what is listed is only part of it.

### Changed

- A tool call that ran and failed is answered `-32603` rather than `-32602`,
  which now means only that no server offers a tool by that name.
- `mcp.ServerStatus.Refused` names each refusal quoted, and a name too long to
  print whole is given by its length rather than cut, since two names cut alike
  would read as one.
- `route.Rule` and `route.Decision` gained `PolicySet`, which tells a loss policy
  that was chosen from one that was left alone. A `Rule` built without it keeps
  taking the table's default, as before.
- `agentcli` gates itself at four concurrent invocations by default, since each
  one forks a whole agent CLI. A call past the limit waits for a slot until its
  context ends, which through a goose client is its attempt deadline passing
  or its caller going: `Acquire` used to turn it away at once with
  `KindSaturated`, which for a CLI on its own subscription login meant failing
  it outright or handing it to a backend billed per call. A wait ended by
  either deadline is `KindTimeout`, and one ended by the caller going
  `KindCancelled`. Only as many calls wait as the gate admits, since each holds
  whatever admitted it while it waits, a proxy's own concurrency slot among
  them; a call past those is still shed as saturated, to the next link or a
  503. `MaxQueued` sets how many may wait, as many as may run unless set, and a
  negative one queues nothing. A zero `MaxConcurrent` takes the default and a
  negative one removes the gate; in a file, leaving `max_concurrent` out takes
  the default, zero removes the gate, a negative number is refused, and as many
  may wait as may run.
- `cloudflare.New` returns `*cloudflare.Provider` rather than `*openai.Provider`,
  so it can read the catalogue from where Cloudflare keeps it. The type embeds
  the old one, so everything it served still works through it.
- `openrouter.New` returns `*openrouter.Provider` rather than
  `*openai.Provider`, so it can read OpenRouter's 400 for an unknown model id
  as `KindNotFound`. The type embeds the old one, so everything it served
  still works through it.
- `llm.Stream` gained `Redact` and `Unredacted`, which let a proxy replace what
  a backend said on the way to a caller while the operator's record keeps it,
  and `Settle`, which fixes what a stream reports it cost to the row that was
  booked. Its bookkeeping is behind a mutex, so closing a stream that is being
  read no longer races the row it is about to bill.
- `goose.WithRedactedErrors` is new, and `Client.Models` and `Client.Ready` ask
  every backend at once and turn a provider that panics into an error rather
  than taking the process with it.
- `llm.Limits` resolve field by field rather than whole, so a call that pins one
  of them no longer drops the rest.
- `usage.Memory` keeps its last ten thousand rows rather than all of them, and
  `usage.Multi` writes to its sinks at once and survives one that panics.
- `llm.DropRefusal` is a diagnostic code again, `llm.AnomalyOutputUnknown` is
  new, and `mcp.ServerStatus` gained `Truncated`.
- `agentcli.DefaultMaxConcurrent` and `route.Rule.PolicySet` are new. A route's
  `strict` is a three-state now, so `strict = false` is an opt-out rather than
  a setting nobody can tell from silence.
- `llm.MCPServer`'s tool and approval lists tell an empty list from an absent
  one: naming nothing is a setting, and it is kept through a clone and out to
  every wire.
- `llm.Error` gained `Reason` and `Detail`, which say what a ledger is told about
  a failure and whether the upstream's own words are part of it, and
  `llm.Attempt` gained `Redact`, which carries that choice down to the rows
  booked from under a provider.
- `llm.Stream.Prime` holds the frames a backend sends before its first content,
  up to sixteen, and returns a failure that arrives before that content instead
  of delivering it after the commit. The held frames are delivered once, in
  order.
- Under `Strict`, a `drop.unknown_block` at info — a block carried through
  whole — is no longer promoted to fatal. A destination whose `Renders.Raw`
  does not name the block's vendor cannot write it out and loses it, and
  `Strict` refuses that. Every other content diagnostic below fatal is still
  promoted.
- An `llm.Error`'s `Code` is the vendor's name for a failure where one was
  given beside a number: `RESOURCE_EXHAUSTED` from Gemini rather than `429`.
  `Body` is set on every failure an upstream reported in its own words,
  whatever the status, so `Reason` and `Detail` treat an in-band failure as
  they do a refusal.
- `llm.Limits.MaxFrameBytes` bounds a stream's event as well as its line. The
  event ceiling used to follow `MaxBufferBytes`, which still bounds what is
  accumulated from a stream. The line's default rises from one megabyte to
  eight.
- Passthrough no longer forwards a caller's `OpenAI-Organization` or
  `OpenAI-Project`, as it already did not forward `X-Goog-User-Project`, unless
  the key it forwards is the caller's own.
- A browser client, a browser extension or a desktop app built on a browser
  engine that sends an `Origin` must be named with `cors = true` and
  `cors_origin`, even where CORS itself is not needed, and that includes a page
  served from the same origin through a front proxy. A keyless loopback proxy
  reached under any other name — the machine's hostname,
  `host.docker.internal` — answers 403; set `server.api_key`, or use
  `localhost` or `127.0.0.1`. With CORS off, `OPTIONS` requests go through the
  key check, the limiter and the gate like any other.
- Upgrading can refuse a file that used to start. A provider behind a gateway,
  or behind any `base_url` off its vendor's host, needs `api_key_env` naming
  the variable it should receive. A `goose.toml` or `.goose.toml` in the
  working directory needs `-c`, for the proxy as for `goose mcp`. A variable
  named in `server.api_key_env` must be set, and `cors_origin = "null"` needs
  `server.api_key`. Keys on Bedrock and Vertex, and `max_retries` on agent
  CLIs, must be removed, and an agent CLI given a key needs
  `options.api_key_env_name`. A named AWS profile that cannot be read fails
  rather than falling back to the runtime role. A profile named in
  `options.profile` is used ahead of keys in the environment. Where
  `AWS_WEB_IDENTITY_TOKEN_FILE` is set, the web identity is used or refused
  ahead of `AWS_PROFILE`, the shared file and the runtime role. A Vertex key
  named by `options.credentials_file` or `GOOGLE_APPLICATION_CREDENTIALS` that
  cannot be exchanged fails rather than falling back to the metadata server.
  Negative durations, a `base_url` carrying a query or fragment, an `[[mcp]]
  url` without a scheme or host, a Bedrock or Vertex region outside
  `[a-z0-9-]`, an Azure resource outside letters, digits and hyphens, and,
  under `-check`, a `[usage] log` whose directory is missing are refused;
  `azure.New`, `bedrock.New` and `vertex.New` refuse such a `Resource` or
  `Region` too. A relative `[[mcp]] command` or
  `dir`, and an agent CLI's relative `base_url`, `options.dir` and
  `options.temp_dir`, are now read beside the config file rather than the
  working directory.
- An empty text part is left out of an Anthropic-wire message rather than sent
  as an empty text block, which that wire refuses.
- `llm.KindForStatus(413)` is `KindInvalidRequest` rather than `KindUpstream`.
  The chain still walks on from it, since each backend has a ceiling of its
  own, and it no longer counts against the breaker.
- A call that offers tools may try a route's links out of their order, the
  ones that carry the tools first. `Attempted.Index` and `Attempt.Index` still
  give each link's place in the route, so the ledger's rows may not follow it.
- `llm.Attempt.Note` holds a note to the attempt's policy whether or not the
  attempt has a `Recorder`, so `Check` sees every note. A `loss` failure
  refused before anything was sent is booked as a rejected attempt, as a
  saturated one is.
- `drop.signature` from the planning step now counts what it removed, and it
  covers a signature on any part, not only on thinking. A backend that
  verifies no signatures is sent none, so a Gemini signature bound for such a
  backend is now reported as lost where before it vanished silently, and a
  strict policy refuses that link.
- A reasoning item goose names itself on the Responses wire is `rs_goose_<n>`,
  not `rs_<n>`.
- A routed call to a `kind = "anthropic"` provider now carries the caller's
  `anthropic-beta` flags, and the Anthropic-wire request fields they open,
  under the operator's key. That used to need `server.passthrough`. Some of
  those flags change what a call costs, such as the long-context one. Set
  `client_betas = false` on the provider to withhold them; the fields are then
  reported as a loss, which `strict` refuses.
- `bedrock.Provider` declares `RedactedThinking`, and `*cloudflare.Provider`
  has its own `Complete` and `Stream`, which wrap the embedded provider's so
  that a model Workers AI does not have is `KindNotFound`.
- A relative `[usage] log` is resolved beside the file that names it rather
  than in the working directory, so a proxy started as
  `goose -c /etc/goose/goose.toml` with `log = "usage.jsonl"` writes to
  `/etc/goose/usage.jsonl`.
- A proxy with no `server.api_key` meters callers by address, the one
  `forwarded_header` names where that is set, rather than by whatever
  credential each presents, so callers behind one address share one
  allowance.
- Passthrough answers any method but `GET`, `HEAD`, `POST`, `PUT`, `PATCH`,
  `DELETE` and `OPTIONS` with `405`, and a body over `max_body_mb` with `413`
  rather than `502`.

## v0.1.0

First tagged release. Everything below was found by auditing the claims in the
README against the code that was meant to keep them.

### Fixed
- A streamed answer cut off mid-generation reached the client as a complete one:
  `[DONE]` on the OpenAI wire, a full `message_delta`/`message_stop` on the
  Anthropic wire, and a 200 either way. `llm.Stream` now reports a stream that
  ran out of frames without ever saying why it stopped, so every codec and every
  embedded caller sees it; the framing is still closed, with an error event
  before it. A consumer that stops reading of its own accord is not a truncation
  and closing such a stream is not reported as one.
- An upstream's raw error body travelled to the client inside `error.message`,
  as an escaped JSON blob carrying whatever the vendor said about the operator's
  account. The body is now parsed into a message, type and code in the client's
  own format — OpenAI, Anthropic, Google, Ollama, AWS, Cloudflare and FastAPI
  shapes — and the raw text is kept on `llm.Error.Body` for the ledger and log.
  `server.redact_upstream_errors` replaces the vendor's sentence entirely.
- The same leak through three other seams: a Google token exchange reported only
  its OAuth code, an agent binary's stderr moved to the operator's record, and a
  tool server's refusal is parsed rather than relayed.
- `llm.Limits.Timeout`, `MaxRequestBytes` and `MaxResponseBytes` were settable
  through `WithLimits`, `WithCallLimits` and a route, and read by nothing. A
  unary attempt is now bounded end to end; a streamed one up to its first frame,
  because the body takes as long as the answer does. Bodies are bounded in both
  directions, and a request over the limit is refused before it is sent.
- Every provider built an `http.Client` with the standard library's default of
  two idle connections per host, so a proxy under load reopened TCP and TLS for
  most calls. Retry backoff had no jitter, which sent every call failed by one
  rate limit back at the same instant.
- `httpx.Server.ServeHTTP` rebuilt its middleware chain on every request.
- Passthrough copied hop-by-hop headers onto the reply and buffered a streamed
  body instead of flushing it.
- An oversized request body was detected by matching words in an error string.
- `Request.SystemAndMessages` appended into the caller's own slice.
- A stream abandoned without being closed was never booked, so a delivered call
  left no ledger row at all. The row the closing consumer writes wins the race
  with the fallback, because it is the one that knows the figures.
- Both MCP transports read under the lock that serialises calls, and neither
  read could be cancelled. A tool server that stopped answering held that lock:
  the timeout the client declares never fired, the caller's own cancellation
  did not reach the call, and every later call to that server waited behind it
  for the life of the process.
- `mcp.Manager.Refresh` existed and nothing called it, so a tool list was read
  once at boot and cached for the life of the process — advertising tools a
  server had dropped and hiding ones it had gained, which is what
  `WithToolSource` was written to avoid. It now runs on a timer, off the
  request path.
- Four copies of the hop-by-hop header list disagreed: the Google and Ollama
  passthroughs named five headers to drop and forwarded the rest, handing the
  upstream a `Transfer-Encoding` that confuses its framing, an `Upgrade` nobody
  arranged and a `Proxy-Authorization` minted for this proxy. None honoured the
  further headers a `Connection` nominates.
- A turn that rendered to nothing vanished from the conversation in five
  adapters without a word, changing whose turn came next. `Message.Name` was
  decoded by the codec that owns it and emitted by nobody. Every adapter sent
  `description: ""` for a tool that had none. Two adapters sent a null message
  list where the other three refuse a conversation that rendered to nothing.
- A tool choice with no tools, a response format naming no type, an unknown
  thinking mode read as disabled, a role this wire does not have read as the
  caller's own turn, and an output ceiling of zero replaced by a
  four-thousand-token default: each changed the answer and none said so.
- `num_predict` carried a pointer to tell "the client said nothing" from "the
  client said zero", then threw it away with `> 0`.
- A tool with no name and a tool result naming neither a call nor a tool
  travelled to the backend to be refused there, at the backend's price.
- Both model listings pasted a pagination cursor into the query as it arrived
  from the backend; one carrying an ampersand writes a parameter of its own.
- The AWS event-stream parser compared a `uint32` length as an `int`, which is
  32 bits on some builds: a length near the top of the range read as negative,
  passed both size checks and reached `make` as a negative size.
- The Google token source held its lock across the exchange with no bound, so a
  token endpoint that stopped answering held it for the whole deadline of
  whoever got there first and every Vertex request waited that out.
- A negative server limit read as no limit, and a TLS certificate that could
  not be opened failed at the first connection rather than at the start.
- The Claude CLI announces each turn and then summarises the invocation. Both
  were banked, so a call of 200 input and 20 output was reported as 300 and 30 —
  fifty per cent high, on the ledger and on the client's usage header, for the
  one backend that reports a real price. `Usage.Cover` was written for exactly
  this and wired to nothing.
- A dropped content block was filed under capability on four wires and content
  on three, and `Strict` promotes only content: the same translation refused on
  one wire and was delivered on another. A thinking signature and a redacted
  block were filed under capability too, so `Strict` delivered both.
- `Request.Metadata` carried the end user a request names — OpenAI's `user`,
  Anthropic's `metadata.user_id` — and no adapter sent it, so an upstream saw
  every tenant's traffic as one. Vendors bucket rate limits and abuse reports
  by it.
- `wire.Renders` declared seven things a format can carry and one was read. A
  route that puts tools in front of a format with no field for a tool call
  bought a turn that goes nowhere.
- Both model listings decoded a readable model name that `/v1/models` dropped.
- The abandon grace was a package variable a timer goroutine read while a test
  wrote it.
- Eight request fields were listed in a codec's known set — which is what keeps
  a field out of `Extra` — and decoded by nothing, so each was dropped at parse.
  Among them Gemini's `safetySettings`, so a request was filtered under the
  backend's defaults rather than the caller's thresholds, and the Responses
  `store` flag, so a caller asking not to have its prompt retained was not
  asking anything.
- `Request.Extra` was forwarded to any backend declaring it tolerated unknown
  fields, so an Anthropic client's field reached an OpenAI backend, which
  refuses the whole request for it: every cross-vendor call carrying one failed
  at the backend. It carries the wire it arrived on now, and travels only
  there — which also lets Google and Ollama take their own vendor's fields, so
  Gemini's safety thresholds and cache handle reach the backend that has a
  field for them.
- Truncation was tracked for a stream rather than each candidate, so one answer
  finishing read as the end of all of them.
- Three refusals got a different status depending on the client's format: a
  cancelled call was 499 on one wire and 502 on five, a request no backend can
  serve was 422 on two and 400 on four, and Ollama reported a permission denial
  as an authentication failure. A code that changes when an operator changes
  the route breaks the client's retries.
- A duration an operator writes and goose cannot read was replaced by a default
  in silence.
- Behind a load balancer every caller shares its address, so a per-caller rate
  limit metered all of them as one.
- A streamed request for several completions delivered one. Both parsers that
  can return more than one read `choices[0]` and `candidates[0]` of every
  chunk and threw the rest away, while the unary path and every writer handled
  them all: a client asked for four answers, was billed for four and was handed
  the first. Gemini's terminal chunk collapsed them too, so the finish reason
  and the buffered tool calls of every candidate but the first were lost.
- A streamed Responses call never gave the client the id the backend issued.
  The opening event carries it and nothing read that event, so goose minted one
  of its own, wrote it on every event, and a client sending it back as
  `previous_response_id` named a conversation the backend has never heard of.
  The session header was empty for the same reason.
- `/v1/responses` sent no `tool_choice` at all. The codec read it, the router
  kept it, and the request builder had no line for it: a client that said the
  model must call a tool was not asking anything. The same builder dropped
  `Request.Extra`, so the fields that codec deliberately leaves unmodelled —
  `store` among them — reached the backend on the chat endpoint and nowhere
  else.
- OpenRouter's vendor pinning is configurable, documented in the example file
  and named in the README, and reached no request. `Options.Preferences` was
  called by nothing, so `providers` and `allow_fallbacks` were settings an
  operator wrote and did not get.
- An image the model generated was dropped by the one wire that has an item
  for it. `/v1/responses` is where an image-only model is deliberately routed,
  and neither the adapter nor the codec knew `image_generation_call`: the whole
  answer to a paid call went missing. Previews are streamed as they arrive and
  the finished image closes the item; `llm.Image.Partial` now means a whole
  rendering of the answer so far, because that is what the backend sends and
  concatenating two of them makes a broken third.
- A tool result that was not text vanished into `""`. Four request builders
  flattened one with `TextOf` and reported nothing, so an MCP tool that
  answered with an image reached the model as a tool that answered with
  nothing. Every sampling control a backend has no field for is reported the
  same way now, and Gemini and Ollama were given the two penalties they do
  accept and were not being sent.
- A CLI backend declared that it takes no tool results while folding them into
  the prompt, so every conversation that had ever called a tool was refused at
  the routing gate before that backend was asked.
- An image carrying a URL rather than bytes was sent to Anthropic as an empty
  base64 payload — a refusal bought at the backend's price, for something that
  wire spells perfectly well — and to Bedrock as empty bytes with no word.
- `POST /mcp` answered with a redirect instead of the tools. Mounted under its
  own prefix the path left for the mux is empty, which a mux redirects rather
  than serves, so a client that does not follow one never reached the endpoint
  the banner advertises. A tool result that arrived as audio came back labelled
  as an image, and a server whose name contains the qualifier separator had the
  wrong tool name called.
- `server.cors` with no origin served `Access-Control-Allow-Origin: *`, which
  the guard beside it exists to refuse: it compared the literal `*` and never
  saw the one an unset origin becomes.
- A scrubbed environment that matched nothing was handed to the child as nil,
  which the standard library reads as "inherit this process's" — the proxy's
  own key and every vendor key, to the one backend the scrub was written for.
- A probe the breaker let through after a cooldown kept its slot when the
  attempt was refused before it was sent, by the capability gate or a
  concurrency limit. Nothing ever gave the slot back, so that link was skipped
  for the life of the process.
- A vendor verb that is not generation was served as generation. Gemini puts
  the verb in the path, so `:countTokens` and `:embedContent` were read as
  `generateContent`: a call the caller never asked for, answered in a shape it
  cannot read. They are refused now, or forwarded where the operator said
  opaque requests go.
- An error body inside a 200 was read as an empty answer on the Anthropic and
  Bedrock paths, which is the one thing the other three adapters already
  guarded against.
- Six request fields were named in a codec's known set and decoded by nothing,
  so each was dropped at parse: the deprecated `function_call`, and Ollama's
  `keep_alive`, `suffix`, `template`, `context` and `raw`.
- `Response.Extra` and `Frame.Extra` were declared and nothing wrote or read
  them, so a backend's own answer fields — a fingerprint, a service tier, a
  flag a vendor added last week — were lost on a pass-through between two
  clients of the same wire.
- `UsageShape.MultiTurn` and `UsageShape.Cost` were read by nothing, against
  the README's claim that no capability is decoration, and a CLI backend
  declared an output ceiling it has no field for. A delivered call from the one
  backend that prices its own and did not price this one now says so, because
  nearly every backend reports no price at all and the absence otherwise reads
  the same either way.
- A conversation handle that could not travel was dropped in silence, so a
  chain that failed over answered without the history the handle stood for and
  said nothing about it.
- A route's own format and header names went unchecked, and so did
  `server.passthrough` and the diagnostic codes a route's loss policy names: a
  typo was a route that never matched, an endpoint that answered 404 at the
  first request, or a policy that downgraded nothing and promoted nothing,
  rather than an error at the start. A format named twice refused to start at
  all.
- A retry's ledger row counted the backoff as part of what the backend took.
- A model that answers with nothing but an image was served to a format with no
  place to put one: a 200 carrying an empty message, and the full bill. Which
  formats carry an image is on `wire.Renders` now, and the adapter that knows
  which models produce only one pairs the two and names the endpoint that does
  serve it.
- A tool server's structured answer and the schema a client validates it
  against were both read and neither written back.
- The streamable tool transport sent no `MCP-Protocol-Version`, so a server was
  entitled to assume an older protocol than the one being spoken; it ended no
  session when it was finished, leaving the server holding one until a timeout
  of its own; and it carried a configured header to a remote server over
  plaintext, which is a secret given to anything on the path. An oversize body
  to `/mcp` was truncated and reported as malformed JSON rather than as what it
  was.
- Gating the proxy and bringing your own key were mutually exclusive. A client
  that holds both has to send both, on different headers, and only the first
  credential found was checked — so the caller's own upstream key was refused
  at the door before the path that forwards it could see it. Every credential
  on the request is checked now, in constant time, and the one that is not the
  proxy's is the one handed on.
- A stdio tool server's process was bound to the context that dialled it, so a
  server connected during one request died with that request and every later
  call found a transport whose process was gone. The manager owns the child and
  closes it; the stream transport beside it had always detached.
- A tool-call fragment that repeated the id of the call already open started a
  new call each time, so one call became many, each holding a shard of the
  arguments and none of them parseable. Several compatible servers send exactly
  that.
- A streamed call that failed reported what it cost to the ledger and not to
  the client: the usage header, which is the only place any served error body
  has for it, was written on the unary path alone.
- The tool server's endpoint event was checked against the URL that was dialled
  rather than the one the stream came from, so a redirected stream was refused
  for naming its own host and could name the host it had been redirected away
  from — which is what the check exists to stop.
- The AWS event stream spells a failure two ways and goose read one of them, so
  a mid-stream failure spelt the other way fell through every case and the
  answer ended clean and short. An event this adapter does not name is reported
  rather than dropped.
- A provider configured with a model did not serve it. `goose -p ollama -m
  llama3.1:8b` — the README's own example — forwarded whatever model string the
  client sent, so the flag pinned nothing and a backend that has one model was
  asked for another. The route resolves the configured model where neither the
  chain entry nor the rule names one, and both of those still win.
- The copilot binary fronts the Anthropic API and some builds relay its framing
  unchanged, which goose read for none of its names: the whole answer was
  dropped and the client handed an empty stream closed by a synthesised stop.
- A CLI turn that ended in a tool call reported that it had stopped, so a
  client did not run the tool and the agent loop ended there. The reason a turn
  ended is now what the last thing it produced says, where the binary does not
  say for itself.
- An agentic codex run that ended in tool work contributed nothing to the
  answer: only a message, a reasoning item and an error were read, and every
  other completed item fell through. The tool work reaches the client as a tool
  call, and anything else is carried whole.
- The Claude CLI streams a whole assistant turn on one line unless partial
  messages are asked for and honoured, and goose read only the wrapped events:
  a fully billed answer reached the client as an empty message with a
  synthesised stop. Both shapes are read now, and an invocation that delivered
  nothing at all takes its answer from the line that closes it rather than
  serving silence.
- A Gemini answer with no candidate and no reason for there being none was
  served as an empty success, so the caller read it as the model having nothing
  to say and the next link of the chain was never tried.
- A passthrough forwarded every credential header the client presented except
  the one that backend uses itself, so a key minted for one vendor reached
  another — the leak the routed path has always refused — and a browser's
  cookies for the proxy's origin went with them.
- A Responses upstream that delivers the message without streaming its tokens —
  a gateway that buffers, a reply short enough to send whole — returned an
  empty answer with a finish reason saying it was complete. What was not
  streamed is delivered on the item that closes it, and what was is not sent
  twice.
- A thinking budget aimed at the chat-completions endpoint, which takes an
  effort rather than a budget, reached it as neither and said nothing.
- A passthrough call left no ledger row, so an opaque request that reached the
  backend and spent the operator's credit was on no record but the request
  counter. What it cost is not knowable there, and the row says so rather than
  saying nothing. `Client.Record` is how a proxy books a call goose did not
  translate.
- The SSE scanner never cleared the event type after dispatching one, so a
  frame carrying only data was processed under the name of the frame before
  it — on a wire that dispatches by name, the wrong handler over the right
  payload. The last id still carries, because that is what a reconnection
  resumes from.
- The Anthropic adapter stamped its usage as reported whether or not a usage
  object arrived, so a call the backend said nothing about booked as one it
  said was free.
- A tool's `strict` flag was dropped at parse on both OpenAI wires, so a caller
  that bound the model to a schema was quietly given a suggestion instead. It
  travels to the two vendors that have a field for it, and the rest say they
  have none.
- A generation the Responses backend reported as failed was delivered as a
  completed one whenever it carried no error object: the stop reason had no
  case and fell through to complete. The thinking signature arrives on the item
  that closes the reasoning and was read only on the unary path, so a streamed
  turn could not be replayed to the backend that issued it.
- On the Ollama wire a tool call's arguments sent as a string of JSON reached
  the next backend as a string containing a string; a tool result with no name
  was given an id that matched no call, which the backend refuses the turn for
  rather than naming what was unpaired; and an answer with no content carried
  no `message` key at all, which a typed client dereferences to nothing.
- A response writer that cannot flush turned a proxy that streams into one
  that does not, with no error and no line anywhere: every test that reads the
  body afterwards still passes. The operator put the wrapper there and is now
  told.
- Metadata a backend puts beside the answer rather than in it — Gemini's
  grounding and safety ratings, an OpenAI choice's log probabilities — has
  nowhere in the neutral model to ride and was dropped in silence, so a client
  that asked for grounding could not tell it had not come back. It is named as
  a loss now, on both paths.
- A stream frame carrying a whole message where the standard shape carries a
  delta was read for the delta alone, so the frame looked empty and the answer
  it carried was gone. Some compatible servers send exactly that.
- A built-in call the Responses adapter does not model — a search, a file
  lookup, a hosted interpreter — was dropped on both paths, and so was an
  unmodelled Converse block. Each is carried whole now, which reaches a client
  on the same wire unchanged and any other as a reported drop.
- A content block the Anthropic adapter does not model was carried verbatim on
  the unary path and dropped on the streamed one, so the same answer lost
  content on one and kept it on the other. The streamed path also read the
  event name off the event line alone, so a gateway relaying this shape with
  the name only in the body — where this wire also puts it — parsed to nothing
  at all.
- A cache breakpoint on a block other than the last one was quietly moved to
  the end of the turn, caching more of the prompt than was asked for at the
  rate a cache write is billed. It is declined and reported instead.
- A tool call whose arguments were valid JSON but not an object — a bare
  string where the schema asked for an object — was put in the object slot of
  three wires that refuse the whole turn for it. It is kept under a key.
- A thinking budget of "decide for yourself" reached the one backend that
  spells it as no budget at all, so the caller got the model's default.
- Gemini carries no index on a tool call and goose carried none either, so two
  parallel calls reached a client that reassembles by index as one call with
  two sets of arguments run together. They are numbered from a counter that
  spans the stream.
- A streamed tool-call fragment that carried no index was numbered by its place
  in the chunk, which is always nought: a backend that omits the index — the
  compatible ones do — had two parallel calls merged into one whose arguments
  were two sets run together. An explicit index still wins, a fragment carrying
  an id opens the next call, and one carrying only arguments continues the call
  already open.
- A route that put tools in front of a backend took the caller's away. The two
  are one list by the time a backend sees them, deduplicated by name with the
  caller's winning, so a client that sent its own tools and landed on such a
  route no longer loses them without a word.
- A ledger row for a cancelled call was written against the cancelled context,
  so a sink that reads one refused the row for a call that was billed.
- The Responses wire rebuilds the whole response object at the end, so every
  delta was held in memory with no bound: an upstream that never stopped was
  held until the process died. Past the budget the deltas still go out and the
  terminal event says the object is short of them, which is what a client
  rebuilding from it needs to know. A generated image is not charged to the
  allowance the text is rebuilt from.
- The event counter started at one where this wire starts it at zero.
- The Ollama wire pairs a tool result to its call by name and transmits no id,
  and goose carried neither. Every other wire pairs by an id and refuses a turn
  whose id is empty, so a conversation that had called a tool could not be
  replayed anywhere else, in either direction. One is minted from the name both
  ends do carry, by the same rule the Gemini wire already used.
- A provider's `options` map took any key at all, so a misspelt one was a
  setting written and silently dropped, and a whole number that did not parse
  became zero. Keys are checked against what the kind reads, as every other
  setting in the file already was, and so are `dialect`, `version`, `betas` and
  `prefix` on a kind that never looks at them. An unknown MCP `transport` fell
  through to HTTP rather than saying so.
- An agent CLI took any `system_mode` and read an unknown one as the default,
  so an operator who asked for the cached system channel paid full price every
  call and one who asked for `off` sent the prompt anyway.
- `server.forwarded_header` was accepted by the file and never passed to the
  limiter, so every caller behind a load balancer shared one limit.
- Vertex would not build without a project, while the service account key it
  had been handed named one.
- Only the OpenAI model list was served. A client of every other wire
  populates its picker from the endpoint its own wire defines — `/api/tags`,
  `/v1beta/models` — and got a 404, and an Anthropic client reading
  `/v1/models` got a list shaped for another vendor. Each wire's endpoint is
  served in that wire's shape; the one both wires claim carries both spellings
  of the same facts. Nothing is invented: an upstream that did not say when a
  model was published is answered without the key rather than with a zero.
- A provider that offers no model list, or whose list could not be read, was
  left out of `/v1/models` without a word, so a short picker read as a backend
  that serves nothing. `Client.Models` returns what it left out, and the list
  endpoints carry the codes on the diagnostics header.
- An accounting column the client's own wire has no field for was dropped
  without a word. The Ollama wire carries neither cache nor reasoning, no wire
  carries a price, and a prompt count with no cache column reads as a call that
  cached nothing. The figures still reach the operator; the client is now told
  what its format could not hold.
- `-m` and `-u` were read only alongside `-p`, so either on its own was a flag
  the operator passed and nothing applied.
- A failure inside the proxy was served as 502 on every wire, which reads as
  the backend's fault: it sends an operator looking at one that did nothing
  wrong and invites a gateway in front to fail the call over to another. It is
  500 now. The five byte-identical copies of the status map that let it drift
  are one `wire.StatusFor`, and the cross-wire test names every kind rather
  than the subset someone remembered.
- The proxy's own refusals — no key, too many requests, at capacity — were
  written in the OpenAI shape whatever wire the client spoke, so an Anthropic
  or Gemini SDK reported a parse failure rather than the thing the refusal was
  for. They go through the codec serving the path; a path no codec claims keeps
  the generic shape.
- A path the proxy does not serve, and a method it does not serve on one it
  does, were answered with the standard library's plain text. Every client here
  parses what comes back as JSON, so both reached an SDK as an unreadable body
  rather than a not-found or a wrong method. The status and the `Allow` header
  are unchanged; only the body is written in the caller's wire.
- The OpenAI wires wrote `"code": ""` on a failure that carried no vendor code,
  which claims a code that is empty rather than one that was never there. It is
  null now, as it is on the API this wire is shaped after.
- A Gemini stream that failed after its headers named the failure with goose's
  own kind rather than this wire's canonical status, and carried whatever code
  the upstream had left behind — nothing, where the failure was goose's. A
  client matching on the name saw one it did not know, mid-stream, where it
  could act on it least. Both paths write one body now.
- An upstream's `Retry-After` reached an OpenAI or Anthropic client and was
  dropped for a Gemini or Ollama one, so the same rate limit was actionable or
  not depending on which client asked. It is a standard header rather than any
  vendor's field and every wire carries it. A wait under a second is served as
  one rather than rounded to no wait at all.
- `server.passthrough` naming a provider that cannot forward an opaque request
  — Bedrock signs a payload it has to read — was accepted at the start and
  refused at the first request that needed it, where only the client saw it.
- The passthrough door handed a caller's own credential to the backend whenever
  the operator had configured none of their own, whatever `allow_client_key`
  said. The routed path has always obeyed that setting; both do now. Set on a
  kind that cannot take a caller's credential at all, it is refused rather than
  quietly ignored.
- An output ceiling goose supplied because the Anthropic wire refuses a request
  without one was invented in silence, so a client of a wire with no such
  requirement was answered under a limit it never chose and stopped at a length
  it did not ask for. `cap.ceiling_supplied` says so; the same code replaces the
  `drop.extra_field` the explicit-zero case was filed under, which dropped
  nothing.
- `Stream.OnFrame` was called while the stream read a frame in, not on its way
  out, so an observer installed after the router had primed the stream — which
  is every observer, because priming is how the router learns the backend
  answered — never saw the first frame, and saw the rest a frame early.
- `Stream.OnEnd` replaced whatever hook was already there. The router puts the
  ledger row on that same stream before an embedder ever sees it, so an
  embedder setting its own left a delivered call off the record with nothing to
  say so. Hooks add and run in the order they were set.
- The OpenAI and Ollama adapters read the timestamp the backend put on its own
  answer and then answered with goose's clock, reporting the proxy's moment as
  the backend's. What the backend said is carried; a backend that said nothing
  is still stamped, because that is the closest thing anyone knows.
- A streamed answer went out without the stamp its unary twin carries: the
  Responses object said `"created_at": 0`, which a client reads as 1970, and
  every Ollama frame went out without the `created_at` that wire puts on all of
  them. Both are stamped now, from what the backend said where it said
  anything.
- What the backend said about its own answer was read and then dropped: the
  tier it served under, which changes what the tokens cost and which the OpenAI
  wires report on every call; the id Gemini gives its own answer; and the
  Responses object's own timestamp. All three are carried, and where the
  client's wire has a field for one it is written there rather than left to
  ride as an unmodelled extra.
- A loss the OpenAI adapter caused went on the record with no backend named,
  where the Ollama and CLI adapters name theirs, so a ledger row read as a loss
  nothing caused.
- `Retry-After` was set on a rate-limited reply and not exposed to CORS, so the
  one header a browser client can act on was the one it could not read.
- A tool result marked as a failure reached a backend with no field for one and
  lost the mark in silence: only the Anthropic wire and Bedrock carry it, and a
  model weighs a flagged failure differently from a result that merely reads
  like one. `Capabilities.ToolErrors` says which backends have the field and
  `drop.tool_error` says when it went missing.
- `WithRenders` promised that a request for something the client's format could
  never read is refused before a backend is asked, and only the proxy enforced
  it: an embedded caller asking for four completions on a format that carries
  one, or for thinking a format cannot show, paid for an answer it would then
  drop. The rule is the router's now, and the proxy no longer keeps a second
  copy of it.
- Every provider took a header set for the upstream and no kind could be given
  one from the file, so a deployment behind a gateway that wants a header of
  its own could use the library and not the binary. `[[provider]] headers` sets
  them; the credential is written after them, so one named there cannot replace
  it.

- `options.num_predict` below zero is this wire's way of saying there is no
  ceiling, which the neutral request says by leaving it unset — so the setting
  the client wrote reached the backend as nothing at all, with no word that it
  had gone. Found by the fuzzer, whose corpus now keeps the case.

- A vendor's own knobs that sit inside a field the codec reads were dropped at
  parse, on a proxy whose whole claim is that they are not: Ollama's `options`
  bag past the handful goose models — `num_ctx`, `repeat_penalty`, `mirostat`,
  the rest — Gemini's `generationConfig` past its own, and the Responses wire's
  `reasoning` and `text`, where goose also wrote a summary of its own over the
  one the client had asked for. Nothing else would have carried them, because
  the field around them is one goose reads. They travel to a backend that
  speaks the same wire, like every other unmodelled field.

- The Gemini and Ollama wires both carry a frequency and a presence penalty,
  goose models both, and every adapter has a field for them — and neither codec
  read them, so a client that set one reached a backend without it. The Gemini
  one reached no backend at all, because `generationConfig` is a field this
  codec reads and what it did not read inside it went nowhere.

- Both major vendors let a caller ask for one tool call at a time, and each
  puts the control somewhere else: beside the tools on the OpenAI wire, inside
  the tool choice on the Anthropic one. Modelled in neither, it crossed no wire
  at all — the OpenAI spelling reached only an OpenAI backend, as an unmodelled
  field, and the Anthropic one reached nothing. `Request.ParallelTools` carries
  it, each adapter writes its own spelling, and a backend with no field for it
  says so.

- Three losses on the Bedrock and Responses adapters went on the record with no
  backend named, which reads in a ledger as a loss nothing caused.

- `Limits.MaxRequestBytes` and `MaxResponseBytes` bounded every backend that
  speaks HTTP and not the one that spawns a binary, so a request the route
  called too large was still run against the operator's own subscription and a
  binary that never stopped talking was held only by the clock. A read failure
  that already knew what it was keeps its own kind and code rather than being
  rewrapped as a transport fault.

- An agent CLI provider took a `timeout`, defaulted it to five minutes like
  every HTTP backend, and read it nowhere: a binary that never returned held a
  slot, a child process and the caller for as long as the process lived, and
  the setting the operator wrote did nothing. It bounds the invocation now and
  is released when the stream ends.

- The tool endpoint answered a JSON-RPC notification, which takes no answer of
  any kind: only `notifications/initialized` was recognised, so a cancellation
  or a progress report came back as "unknown method" on a connection the client
  was still using.

- Bedrock declares that it honours a tool choice and the Converse API has no
  way to say "do not call one", so a caller asking for none had the setting
  dropped in silence and the backend's own default — that the model may — left
  in its place.

- A tool choice naming a specific tool with an empty name reached one wire as a
  request the backend refuses and was dropped in silence by another. It is
  removed once, at the routing step, and reported.

- `make release` built nothing and then failed: the per-platform rules were
  written as variable assignments rather than recipes, so every target was
  "nothing to be done" and the checksum step ran against a directory that had
  never been created. The checksum file also listed itself.

- An agent CLI reports how many turns it billed inside one invocation and the
  figure was parsed and dropped, so a call whose turns goose could not tell
  apart was booked as one where the binary had billed five.

- Both halves of the MCP handshake named a hard-coded `0.1.0` as goose's own
  version, in the `clientInfo` sent to every tool server and the `serverInfo`
  answered to every client. It follows the binary's version now, and a library
  embedded without one says it does not know rather than naming a release it
  may since have moved past.

### Changed

- The credentials a Bedrock or a Vertex provider needs can be built by anyone
  who imports the package. Every source that resolves one — the container and
  instance metadata services, a service-account key, the Google metadata
  server, and trying several in order — lived behind `internal/`, so the binary
  could reach them and an embedder could not, and had to reimplement IMDSv2 and
  a signed JWT exchange to register the same provider. They are
  `bedrock.StaticCredentials`, `ContainerCredentials`, `InstanceCredentials`
  and `FirstWorkingCredentials`, and `vertex.StaticToken`,
  `ServiceAccountToken`, `MetadataToken` and `FirstWorkingToken`. The binary resolves its own through the same functions,
  so the two cannot drift.
- `mcp.Version` was a mutable package variable, which raced against any server
  still handshaking when a late caller named the binary. It is `mcp.SetVersion`
  and `mcp.Version()` now.
- `agentcli.DefaultEnvAllowlist` was an exported slice, so a caller appending
  its own names widened what every other caller in the process passed to a
  child. It is a function returning a copy.
- A link the breaker skipped was noted and the note discarded, so a chain that
  quietly dropped a backend was indistinguishable to the caller from one that
  never had it. It reaches the response, the stream and the refusal.
- Capabilities the adapters declared and the router never read are now read:
  `SystemPrompt`, `ToolResults`, `ParallelTools`, `ThinkingSignatures`,
  `RedactedThinking`, `Extensions`, `MaxInputBytes` and the cache breakpoint
  budget. `AudioIn` and `ImageOut` are gone: nothing could consume them, and
  `llm.Part` has no audio kind to make the first true.
- Diagnostic codes with no emitter are gone (`drop.refusal`, `acct.clamped`,
  `acct.unmeasured`, `wire.frame_truncated`) or have one
  (`policy.saturated`, `acct.row_dropped`, `drop.cache_hint`,
  `wire.no_finish_reason` on the streamed path). `cap.parallel_tools` is new.
- A chain returned by `WithSelector` runs under the table's tools, policy and
  limits instead of silently under none.
- `Response.Diagnostics` and friends are `llm.Diagnostics`, which gained `Has`
  and `Find`.
- Gemini candidates carry an empty `parts` list rather than `null`.
- The openai adapter derives the output-limit spelling and the streamed-usage
  opt-in from its own declared capabilities, so what a caller can inspect is
  what the request is built from.

- A backend that never says why it stopped can be tolerated through the loss
  policy: the truncation is reported either way and only the failing is the
  caller's to choose.
- Reconciling a backend's summary against its turns records an anomaly where
  the two views disagree, because the figure was arrived at rather than
  measured.

### Added

- `server.tls_cert` and `server.tls_key`. Off loopback without them the binary
  warns that keys and prompts cross the network in the clear.
- `server.max_concurrent` and `server.max_per_minute`. One shared API key
  otherwise leaves nothing between a client with a loop and the whole quota.
- `server.metrics`, serving `GET /metrics` from the ledger.
- The MCP endpoint refuses a request carrying a browser `Origin`.
- The usage ledger carries the upstream's own error text in `detail`.
- `WithBreaker` skips a link that has failed its last several calls, for a
  cooldown, rather than trying it on every request and making every caller wait
  for a backend everybody already knows is down.
- `X-Goose-Usage` reaches a streamed reply as a trailer: what a stream cost is
  not known when its headers are written.
- `server.forwarded_header` names a header carrying the client's address, read
  only because the operator named it.
- A conformance suite asks the real backends, behind a build tag, with each one
  skipped unless its credential is in the environment. Everything else runs
  against a stand-in, which says nothing about whether the vendor agrees.
- The decoders are fuzzed against a hostile body, and so are the event-stream
  parser, the SSE scanner, the line reader, the fold, the accounting and the
  configuration loader.
- Every wire is checked against the claim it is proxied under: what the client
  set reaches the backend under that backend's names, what the backend answered
  reaches the client, and every token of a streamed answer arrives in order —
  or the difference is on the diagnostics header. Silence is the failure.

- The deprecated OpenAI function-calling surface is read: `functions` becomes
  the tool list, `function_call` the tool choice, and an assistant turn's
  `function_call` is paired with the `role: "function"` turn that answers it by
  an id minted for both, because every other wire matches them by one.
- Ollama's graded thinking. `"think": "high"` is a level this endpoint accepts
  and goose refused with a 400; it now travels to a backend that grades it and
  is reported to one that does not.
- `goose_throttled_total`. A rate-limited request never reaches the handler
  that counts requests, so an operator who set a limit could not see it bite.
- `cap.session_elsewhere`, `acct.multi_turn` and the `price_missing` anomaly.
- A generated image on the Responses wire, in both directions: the item on the
  unary path, and the previews and the finished picture on the streamed one.
- `deny` on a route, and `llm.Codes` for anything checking a policy against the
  codes that exist.
- `Client.Record`, for a proxy booking a call goose did not translate.
- The conformance suite is compiled in CI. Behind a build tag nothing else
  builds it, and it would rot until somebody spent real tokens finding out.
- The keys an AWS runtime hands its own workload: an ECS task, an EKS pod
  identity, and an EC2 instance role read over IMDSv2. A Bedrock deployment on
  AWS infrastructure had no way to authenticate without a static key, while the
  Vertex side has read the instance metadata all along.
- The startup banner names the tool servers that answered, with what each said
  about itself and how many tools it offers. The file said which were
  configured; a server that spawned, handshook and offers nothing read exactly
  like a working one.
- `server.mcp_tools` and `goose mcp`. The proxy could be told about tool servers
  and could serve their tools back, but could not itself be used as one: an
  agent that speaks the tool protocol and no vendor wire had no way in, even
  though every provider, route and fallback chain behind goose would have served
  it. `ask` and `models` are answered by goose, with the model, the provider and
  the bill beside the answer, and a backend's refusal comes back as a failed
  tool call rather than a protocol error nothing can read. `goose mcp` serves
  the same protocol on stdin and stdout, because most agents spawn a command
  rather than opening a socket; the endpoint is unchanged and its loopback and
  API-key rules cover the new tools. A model is never offered them: it would be
  offered itself.
- A tool server the backend dials itself now crosses the proxy. OpenAI spells it
  as an `mcp` tool and Anthropic as `mcp_servers` beside the messages; goose
  refused the first outright and dropped the second as an unknown field, so a
  client that had arranged its tools that way could not use the proxy at all.
  Both are read into one shape and written back out on either wire — including
  the `mcp_toolset` half of the Anthropic spelling, which was refused as an
  unportable tool type and so rejected every request from a client that had one
  configured. A request carrying a server reaches the endpoint that can serve it
  rather than the one the caller happened to speak, and the beta flag that opens
  the field travels with the request that needs it. A restriction that cannot be
  carried — an approval step, a read-only filter, a denylist on a wire that
  names only what is allowed — is refused before the call is spent rather than
  quietly dropped, because serving it would run what the caller withheld;
  anything merely descriptive is reported with `drop.mcp_server`, and a backend
  with no such field at all says so once.
- The tool protocol handshake is an agreement again. Every client was answered
  with the newest revision goose knows whatever it asked for, which a client
  pinned to an older one reads as a refusal and disconnects on; goose now agrees
  to any handshake-based revision it speaks, `2024-11-05` through `2025-11-25`,
  and names its own only when it cannot.
- Both halves speak the current protocol revision, `2026-07-28`, which replaced
  the handshake with per-request metadata and has no session at all. goose spoke
  only the older era, so by the specification's own compatibility matrix a
  client of the current revision could not use its endpoint and a server of it
  could not be dialled — the failure being an implementation-defined error at
  one end and a silent mis-parse at the other. The endpoint now serves both eras
  side by side, including `server/discover`, the header mirroring and its
  `-32020` refusal, `-32022` naming the versions on offer, `404` for a method
  that does not exist, `403` for a browser origin and `405` for the GET stream
  and session the revision removed. The client probes with `server/discover`
  before anything else and falls back to the handshake on any other answer or on
  silence, bounded by a deadline of its own so a quiet server costs one short
  wait rather than a whole call timeout. A tool result that asks the client for
  input reaches the caller as a refusal rather than as an empty answer. A client
  that closed its end of the pipe no longer holds the process open: the answers
  in flight are given a moment to finish and then it exits, which is what the
  specification asks of a server whose input has closed.
- Arguments a tool asks to have mirrored into headers are mirrored. The current
  revision requires a client to copy the values a tool's `x-mcp-header`
  annotations designate, and a server that designated them refuses a call that
  arrives without them — so goose could not call such a tool at all. They are
  now computed from the tool's own schema on the way out and held against the
  body on the way in, with the sentinel encoding for a value that cannot travel
  as plain ASCII; a definition whose annotations break the rule is left out of
  the list rather than called incorrectly, and the banner names what was left
  out, because a tool that vanishes without a word is undiagnosable.
- A client is told when the set of tools moves. goose re-reads its servers on a
  timer and was the only one that knew: a client that listed once at startup
  never learned of a tool that appeared after, for the life of the session.
  `subscriptions/listen` is served on both transports — the acknowledgement
  first, every frame naming the subscription it belongs to, a heartbeat on a
  quiet HTTP stream and a graceful close when goose is the one ending it — and a
  client of the handshake era is told on the channel it already has. The
  capability is claimed only where the notification can actually arrive, and a
  refresh that read the same list again announces nothing.
- Progress on a long call reaches the caller. A tool server reporting on a call
  in flight had those frames stepped over as noise, so a five-minute tool looked
  identical to a lost one; they are now relayed under the token the caller
  chose. A request carrying a `progressToken` is answered on a stream — the
  reports in front of the result — on the endpoint and on the pipe alike, and
  `ask` reports the answer forming as the model writes it. `mcp.Report` is how
  a tool served here says so, and `mcp.Watched` tells it whether anyone is
  listening before it does the extra work.
- A cancellation on the pipe stops the call it names. The transport carries
  every request on one channel, so there is no stream to close per request and
  `notifications/cancelled` is the only way a client can withdraw one; it was
  read as a notification and dropped, so a tool — including one that spends an
  operator's credit on a completion — kept working on an answer nobody would
  read, and answered it afterwards, which the specification forbids.
- A tool server's definition crosses the proxy whole. Each one was rebuilt from
  the three fields goose models, so a title, an icon, an annotation — anything
  the client rather than the model reads — was dropped at the seam. The
  server's own object is now passed on under the name goose advertises it by.
