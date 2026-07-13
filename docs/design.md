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
tool results, and decides whether to continue, retry, or terminate. It also owns maximum
iteration limits and budget reservations. Phase A defines an approval-policy interface
that the Loop calls before side-effecting tool execution; interactive human approval is
deferred to Phase D. Phase A implements explicit stop-reason dispatch and the API-call
limit only: retries, backoff, and budget reservations are deferred to Phase B.

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
   still fits under the hard cap. Otherwise it halts and surfaces the ambiguity. This is
   acceptable because duplicate effects are bounded to explicitly reserved model spend,
   no external tool side effect can occur from the lost response, and the event is
   recorded for audit.
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

## Protocol References

- [Anthropic stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
- [Anthropic tool-result handling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)
- [Anthropic API errors and long-request guidance](https://platform.claude.com/docs/en/api/errors)
