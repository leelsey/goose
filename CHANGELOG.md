# Changelog

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
