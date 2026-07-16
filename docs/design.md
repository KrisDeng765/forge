# Forge Design

## Goal

Forge is a command-line Agent Runtime that operates without relying on heavy Agent
frameworks, interacting directly with LLM APIs via raw HTTP calls. It manages
conversation state, registers and executes tools, and safely drives the agent loop by
interpreting the model's content blocks and stop reasons.

## Non-goals

- Do not use existing Agent frameworks like LangChain or LangGraph.
- Do not develop a graphical user interface (GUI).
- Do not support multi-user or distributed execution.
- Do not implement e-commerce business logic (e.g., Listing, PPC, replenishment).
- Do not implement a cross-session long-term memory system.
- Do not attempt to replace mature, production-grade Agent frameworks.

## Components

### API Client

The API Client reads authentication credentials from environment variables, serializes
one fully formed typed request into JSON, invokes the Anthropic Messages API via HTTPX,
and parses the response into either a typed message or a typed exception. It is a dumb,
single-attempt transport boundary: it does not assemble prompts, mutate conversation
state, retry requests, execute tools, or decide whether the loop should continue.

### Conversation State

The Conversation State sequentially stores user, assistant, and tool-result messages for
the active run and returns a replayable snapshot of that history. It does not own the
top-level `system` prompt, tool definitions, request budgets, or API calls. Phase A only
provides short-term, in-process state and excludes cross-session long-term memory.
Snapshots are deep copies so request assembly cannot mutate the state-owned transcript;
assistant blocks are retained verbatim, including provider-added fields accepted by the
replay experiment.

### Tool Registry

The Tool Registry maintains tool names, descriptions, JSON Schemas, and their
corresponding Python callables. It exposes typed tool definitions for requests and
resolves execution by the names supplied by the model. Before execution, it validates
tool arguments using Pydantic, rejecting unknown tools and malformed inputs. It does not
assemble API requests or mutate conversation history. Each registration supplies an
explicit Pydantic input model: its JSON Schema becomes the wire definition, parameter
descriptions live in `Field(description=...)`, and the decorator supplies the tool-level
description.

### Agent Loop

The Agent Loop owns orchestration and full request assembly. On every iteration it
combines the run's immutable top-level `system` prompt, the Tool Registry's current tool
definitions, and the Conversation State's message snapshot into a
`CreateMessageRequest`, then passes that complete request to the API Client. This keeps
protocol transport in the client and state storage in Conversation State while placing
cross-component policy in the one component that has all required dependencies.

The Loop interprets content blocks and `stop_reason`, executes requested tools, appends
tool results, and decides whether to continue, retry, or terminate. It owns maximum
iteration limits, bounded validation self-correction, and terminal states. A wrapping
MessageClient owns retry and budget policy so the loop remains focused on orchestration.
Phase A defines an approval-policy interface that the Loop calls before side-effecting tool
execution; interactive human approval is deferred to Phase D.

### CLI

The CLI captures user instructions, boots the Agent Loop, and streams model text, tool
calls, errors, and final statuses to the terminal. It strictly manages user interaction
and contains no model protocol or e-commerce business logic.

## Failure Modes

1. **Missing or invalid API key**
   Check environment variables before transmitting requests; halt immediately on
   authentication failure without initiating futile retries.
2. **API rate limiting (`429`)**
   Read the `Retry-After` header and retry using exponential backoff with jitter up to a
   defined ceiling. Reserve budget again before every attempt.
3. **Provider overload (`529`)**
   Classify this as a transient service fault, perform a capped number of backoff
   retries, and terminate clearly if the limit is exceeded.
4. **Network timeout or connection failure**
   Do not treat `POST /v1/messages` as idempotent. A connection timeout occurs while
   establishing TCP/TLS, before an HTTP request is delivered, so the Loop may retry it
   with bounded backoff. A read timeout occurs after the connection is established and
   the request has been sent: the provider may have generated and billed a complete
   response that the runtime never received. Blindly retrying then pays twice for one
   logical turn. Connection resets or transport failures after transmission begins are
   classified as the same ambiguous-completion case.

   The Loop permits at most one automatic ambiguous-completion retry. It does not append
   an unseen assistant turn or execute any tool from it, retains the first attempt's full
   worst-case budget reservation, and starts the retry only if a second full reservation
   still fits under the hard cap. If that second reservation does not fit,
   `RetryingMessageClient` raises `AmbiguousCompletionBudgetError` chained from the first
   ambiguous connection error; the Loop returns `ambiguous_completion`, rather than
   concealing the unknown outcome behind `budget_exceeded`. This is acceptable because
   duplicate effects are bounded to explicitly reserved model spend, no external tool side
   effect can occur from the lost response, and the event is recorded for audit.
5. **Truncated model output**
   Inspect `max_tokens` and `model_context_window_exceeded`. Never dispatch a truncated
   `tool_use` block. The exact action for each reason is defined in the dispatch table.
6. **Invalid tool arguments**
   Validate inputs via Pydantic, return structured validation errors back to the model,
   and allow a bounded number of self-correction attempts.
7. **Unknown tool request**
   Reject execution of unregistered tools and return a structured error to the model
   instead of dynamically invoking arbitrary names.
8. **Tool timeout or exception**
   Enforce a timeout for every tool execution, catch exceptions, and map failures into
   explicit tool results or initiate a safe shutdown.
9. **Infinite agent loop**
   Enforce a hard limit on API iterations, tool rounds, retries, and `pause_turn`
   continuations; halt at the relevant threshold and preserve an audit trail.
10. **Spend limit exceeded**
    Enforce the cap by reserving the maximum possible cost before each API attempt. Let
    `S` be confirmed spend, `B` the hard cap, `I` the estimated input-token count,
    `p_in` the input price per token, `M` the requested `max_tokens`, and `p_out` the
    output price per token. All prices are expressed in the same currency:

    ```text
    reserved_call_cost = I * p_in + M * p_out
    permit_call        = S + reserved_call_cost <= B
    max_affordable_M   = floor((B - S - I * p_in) / p_out)
    ```

    Input counting must include every billable request component and use a conservative
    upper bound; prompt-cache prices must be added when caching is enabled. Clamp `M` to
    `max_affordable_M` before sending and reject the call if no useful positive output
    allowance remains. Because the API cannot emit more than `max_tokens`, this reserves
    the unknown output cost at its worst case. After a response, settle against actual
    usage and release the unused reservation. If an attempt has ambiguous completion,
    keep its full reservation because actual usage is unknowable.
11. **User cancellation**
    Trap terminal interrupts, abort active tasks gracefully, and preserve recoverable
    state.

## Stop-reason Dispatch

The Loop must use explicit exhaustive dispatch. A successful HTTP response is not a
completed agent run until this table has been applied.

| `stop_reason` | Classification | Required Loop action | Conversation-state mutation |
| --- | --- | --- | --- |
| `end_turn` | Complete | Surface the final assistant content and terminate successfully. | Append the complete assistant turn. |
| `stop_sequence` | Complete by caller rule | Surface the generated content and the matching `stop_sequence`, then terminate successfully. | Append the complete assistant turn. |
| `tool_use` | Tool round | Require at least one valid client `tool_use` block. Append the assistant turn verbatim, pass every call through the approval-policy seam, execute all calls, and correlate each result or structured error by `tool_use_id`. Put all results from this response into one user message, then continue. A reason/block mismatch is a protocol error and is surfaced instead of guessed around. | Append one assistant message followed by exactly one user message containing all correlated `tool_result` blocks and no other content. |
| `pause_turn` | Provider-side continuation | Append the paused assistant content verbatim and call the API again with the same tools and request configuration. Continue only within the dedicated continuation and total-iteration limits. Do not synthesize client tool results. | Append the paused assistant turn; add no user message. |
| `max_tokens` | Truncation in Phase A | Treat the response as incomplete and never execute tool blocks from it. Surface a terminal truncation result without appending the assistant turn. Phase B will add the bounded larger-`max_tokens` retry and budget checks. | Do not append the truncated assistant turn. |
| `model_context_window_exceeded` | Non-retriable truncation in Phase A | Treat the response as incomplete, execute no tools, surface a context-limit status with any safe partial text, and terminate. A future compaction policy may create a new request, but increasing `max_tokens` cannot fix an exhausted context window. | Do not append the truncated assistant turn. |
| `refusal` | Refused | Surface the refusal and available `stop_details`, then terminate without tool execution. A future explicitly configured fallback-model policy may retry; Phase A does not do so implicitly. | Append the refusal turn for audit, but do not continue it. |
| `null` | Incomplete protocol state | Accept only while assembling a streaming response. If a completed non-streaming response still has `null`, surface a protocol error and terminate. | Do not append an incomplete response. |
| Any unknown string | Forward-compatible wire value | Preserve the raw response, surface an unsupported stop reason, and terminate. Never reinterpret it as `end_turn`. | Do not append a turn whose completion semantics are unknown. |

## Decisions

### Phase B reliability policy

Retry is a `MessageClient` wrapper, rather than a concern of `AgentLoop` or
`AnthropicClient`. The transport remains single-attempt and the loop still performs one
logical turn; the wrapper can be combined with the budget wrapper as
`RetryingMessageClient(BudgetedMessageClient(raw_client))`. This order makes every retry a
fresh budget reservation. Sleeps and random sampling are injected so tests record exact
delays without real waiting.

| Failure | Retry decision | Delay | Reservation treatment |
| --- | --- | --- | --- |
| `429` | Retry, up to 4 total attempts | Use valid `Retry-After`; otherwise full jitter | Release the failed attempt, reserve again |
| `529` | Retry, up to 4 total attempts | Full jitter | Release the failed attempt, reserve again |
| Connect/pool failure | Retry, up to 4 total attempts | Full jitter | Release the failed attempt, reserve again |
| Received `500`–`599` | Retry, up to 4 total attempts | Full jitter | Release the failed attempt, reserve again |
| Read/write timeout or post-send transport failure | At most one retry | Full jitter | Keep the first full reservation; reserve the retry independently |
| `400`, `401`, `403`, other `4xx` | Never retry | None | Release the failed attempt |

Full jitter is `uniform(0, min(8, 0.5 × 2^(n-1)))`, measured in seconds after failed
attempt `n`. A run has a 30-second cumulative delay ceiling. A bare `500` is treated as a
transient, explicit provider failure (not an ambiguous lost response), matching the
provider's documented handling of 5xx failures. The retry wrapper still rejects a second
ambiguous completion even when its overall attempt cap has not been reached.

### Phase B budget ledger

The ledger is a second `MessageClient` wrapper. Before every transport attempt it computes
an input estimate `I`, reserves `I · p_in + M · p_out`, clamps `max_tokens` to the largest
positive affordable value, and settles from actual response usage. A successful call
releases unused reservation; an ambiguous completion retains its entire reservation.
`BudgetExceededError` becomes the terminal `budget_exceeded` run status.

Forge deliberately avoids the token-counting endpoint in Phase B: it is more accurate but
is still an estimate and adds a network failure mode. For client-supplied JSON, compact
UTF-8 byte length is deliberately pessimistic. That rule alone is not sufficient for tool
requests: the provider injects a billable tool-use preamble which has no corresponding
request byte. `conservative_input_token_estimate` therefore adds the model-specific
`TOOL_USE_PREAMBLE_TOKEN_ALLOWANCE` of 512 tokens whenever `tools` is non-empty.

The calibration attempt on 2026-07-15 built the smallest valid tool-bearing request
(`model=claude-haiku-4-5-20251001`, `max_tokens=1`, user content `x`, one one-fieldless
tool). Its compact JSON estimate was 214 tokens; the pre-registered prediction was that
actual input would exceed 214 because of the injected preamble. The live response reported
544 input tokens, confirming a 330-token provider-side addition. Forge reserves 512 tokens
instead of exactly 330, retaining a 182-token guard band for this model/API-version pair
while `BudgetAccountingError` remains a loud calibration tripwire. Re-run this exact probe
before changing the default model, API version, or tool protocol; if observed usage exceeds
the estimate, raise the allowance before enabling that configuration. Images, server tools,
prompt caching, or other separately priced request features remain out of scope until they
have dedicated estimators and pricing rules.

### Phase B ambiguous-completion trace

For an ambiguous first attempt, the budget wrapper retains reservation `R1`; the retry
wrapper permits one retry, which must reserve `R2` independently. If `R2` cannot fit, the
budget wrapper raises `BudgetExceededError` before sending a second POST. The retry wrapper
recognizes that an earlier attempt remains unknown and raises
`AmbiguousCompletionBudgetError` from that original `APIConnectionError`; the cause is
preserved for logs and the Loop reports `ambiguous_completion` (exit code 8). Thus the user
learns both facts: no retry was affordable and the first request may already have completed.

The CLI uses a `$0.05` default hard cap for the default Haiku 4.5 model; users can set a
positive per-run cap with `--budget-usd`. It records standard prices of `$1 / MTok` input
and `$5 / MTok` output as decimal per-token values. These prices must be revisited when
the default model or provider pricing changes.

### Phase B bounded correction and sync tool timeout

The loop records validation error signatures as `(tool name, error text)`. It stops with
`tool_validation_stalled` after two consecutive identical validation errors or three
validation errors total, instead of spending the general API-iteration allowance on a
deterministically broken tool request.

Every synchronous tool runs through `ThreadPoolExecutor` and `future.result(timeout=10)`.
Timeout returns a correlated `is_error` tool result to the model. Python cannot kill a
running thread: the worker is abandoned and its eventual output discarded. This was an
explicit Phase B caveat. Phase C now executes awaitable tools directly and cancels them at
their await points. Synchronous callables run through `asyncio.to_thread` for compatibility,
so Python still cannot force-stop a running thread; its late result is discarded.

### Phase C streaming and async ADR

Phase C makes one complete migration to async rather than keeping parallel synchronous and
asynchronous paths. `AnthropicClient`, the budget and retry wrappers, `AgentLoop`, tool
execution, and the CLI all use `async def`; the CLI is the single `asyncio.run` boundary.
This prevents two subtly different retry, budget, and transcript implementations from
drifting apart.

Every Messages request sends `stream: true`. The raw client passes received bytes to a
hand-written SSE parser that keeps partial lines across arbitrary HTTP chunk boundaries and
emits events only after an SSE blank-line delimiter. The accumulator then accepts the
ordered `message_start`, content-block, `message_delta`, and `message_stop` events. Text
deltas are sent immediately to a stream observer; tool JSON is held per block index and is
parsed only when `content_block_stop` arrives. The accumulator creates the same completed
`MessageResponse` that the non-streaming loop dispatch table already understands.

`message_start` provides actual input usage early. The budget wrapper compares it with its
reservation before output is generated; an underestimated request retains its full
reservation conservatively and raises `BudgetAccountingError`. Final usage still settles
the reservation. A malformed or cut-off successful SSE response is an ambiguous completion,
because the provider may already have generated and billed it. If a retry follows text that
was already shown, the terminal prints a retry boundary; Forge never silently presents the
second attempt as a continuation of the first. No tool executes until an entire response has
been accumulated.

Tools from a single `tool_use` response are scheduled with `asyncio.gather`. Completion may
be out of order, but the gathered `tool_result` list remains in model block order. The
assistant turn and all tool results are committed to `ConversationState` together only after
the entire tool round finishes. On cancellation, the transcript therefore stops at its last
complete turn; the snapshot separately lists unresolved tool calls for manual reconciliation.

`--state-file PATH` enables an atomic JSON snapshot on Ctrl-C and returns exit code `130`.
The snapshot has a version, UTC interruption time, run configuration, committed messages,
and unresolved tool identifiers. It excludes credentials and deliberately does not resume
automatically: a cancelled side-effecting tool may have changed the outside world before it
observed cancellation.

The checked-in SSE fixtures are deterministic, synthetic protocol fixtures used for
byte-by-byte and random-chunk tests. The planned real `curl -N` capture and the two live
milestone transcripts remain an acceptance task until an `ANTHROPIC_API_KEY` is available;
their response-only fixture must be scrubbed of personal prompt content before committing.

### Raw HTTPX instead of the Anthropic SDK

Using HTTPX in Phase A forces a direct understanding of headers, request bodies, content
blocks, stop reasons, and error responses, preventing the SDK from hiding these protocol
mechanics. The trade-off is manually maintaining serialization, error classification,
and compatibility; however, production projects may transition back to the official SDK
once the underlying protocol is fully understood.

### Tolerant wire reader, strict runtime dispatch

Models for provider-owned responses allow unknown fields and preserve genuinely unknown
content blocks in their original wire shape so that a future provider addition does not
crash parsing or corrupt transcript replay. Known block types still validate strictly: a
malformed known block indicates contract failure or transcript corruption and must not
fall through to the unknown-block model. `stop_reason` remains `str | None` at the wire
boundary, while the Loop restores exhaustiveness through the dispatch table above and
surfaces unknown values. This keeps Pyright strict over internal branches without making
the external parser brittle.

Request models follow the opposite policy and reject unknown fields. Forge controls
their shape, so accepting a typo such as `max_token` would only defer a local error into
an avoidable remote `400`.

### Request assembly belongs to the Loop

The top-level `system` prompt is run configuration, not a conversation message.
Conversation State owns only replayable messages, the Tool Registry owns tool
definitions and callables, and the API Client owns one HTTP exchange. The Loop is the
only component that legitimately depends on all three, so it assembles the complete
`system + tools + messages` request on each iteration.

### Pyright strict mode

Strict type checking establishes explicit contracts for function inputs, outputs, and
nullability. This aids in auditing both human-written and AI-generated code, catching
type mismatches before runtime. It complements, but does not replace, runtime validation
and automated testing.

### Haiku for development

Developing the Agent Loop, tool protocols, and error-handling mechanics does not demand
the strongest reasoning model. Utilizing Haiku reduces operational costs and shortens
feedback latency. While subsequent evaluations will baseline against stronger models,
the runtime architecture itself must remain model-agnostic.

## CLI exit codes

| Exit code | Meaning |
| --- | --- |
| `0` | Completed or caller `stop_sequence` |
| `1` | Runtime, configuration, or protocol error |
| `2` | Argument usage error (owned by `argparse`) |
| `3` | Truncated at `max_tokens` |
| `4` | Context-window limit |
| `5` | Model refusal |
| `6` | Budget exceeded before an API attempt |
| `7` | Bounded validation self-correction stalled |
| `8` | A request may have completed, but its one allowed retry could not be funded |
| `130` | User interrupted the async run; optional state snapshot was written |

## Deferred register

| Item | Status | Trigger |
| --- | --- | --- |
| Approval-defense comment in `loop.py` | Resolved in this Phase B close-out | No trigger; keep the policy fail-closed and model-visible errors generic. |
| Registry callable-signature versus Pydantic-model contract check | Deferred | Before Project 1 registers its first real tool. |
| Six dispatch-table tests | Resolved in this Phase B opening | No trigger; maintain when a `stop_reason` changes. |

## Protocol References

- [Anthropic stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Anthropic tool-result handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
- [Anthropic API errors and long-request guidance](https://platform.claude.com/docs/en/api/errors)
