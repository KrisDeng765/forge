# Forge Design
## Goal
Forge is a command-line Agent Runtime that operates without relying on heavy Agent frameworks, interacting directly with LLM APIs via raw HTTP calls. It manages conversation state, registers and executes tools, and safely drives the agent loop by interpreting the model's content blocks and stop reasons.
## Non-goals
- Do not use existing Agent frameworks like LangChain or LangGraph.
- Do not develop a graphical user interface (GUI).
- Do not support multi-user or distributed execution.
- Do not implement e-commerce business logic (e.g., Listing, PPC, replenishment).
- Do not implement a cross-session long-term memory system.
- Do not attempt to replace mature, production-grade Agent frameworks.
## Components
### API Client
The API Client reads authentication credentials from environment variables, serializes typed requests into JSON, invokes the Anthropic Messages API via HTTPX, and parses successful or error responses into explicit data models. It strictly handles network protocols and does not decide whether to execute tools or continue the loop.
### Conversation State
The Conversation State sequentially stores user, assistant, and tool result messages for the active run, generating the complete message history required for subsequent API requests. Phase A only provides short-term, in-process state and excludes cross-session long-term memory.
### Tool Registry
The Tool Registry maintains tool names, descriptions, JSON Schemas, and their corresponding Python callables, resolving and dispatching lookups based on the tool names provided by the model. Prior to execution, it validates tool arguments using Pydantic, rejecting unknown tools and malformed inputs.
### Agent Loop
The Agent Loop orchestrates the execution flow: it invokes the API Client, interprets content blocks and `stop_reason` values in the response, executes tools requested by the model, appends tool results back to the Conversation State, and re-invokes the model. It also governs control logic, including maximum iteration limits, budget caps, human-in-the-loop approvals, and graceful teardown.
### CLI
The CLI captures user instructions, boots the Agent Loop, and streams model text, tool calls, errors, and final statuses to the terminal. It strictly manages user interaction and contains no model protocol or e-commerce business logic.
## Failure Modes
1. **Missing or invalid API key**  
   Check environment variables before transmitting requests; halt immediately on authentication failure without initiating futile retries.
2. **API rate limiting (`429`)**  
   Read the `Retry-After` header and retry using exponential backoff with jitter up to a defined ceiling.
3. **Provider overload (`529`)**  
   Classify this as a transient service fault, perform a capped number of backoff retries, and terminate clearly if the limit is exceeded.
4. **Network timeout or connection reset**  
   Catch network exceptions, perform bounded retries strictly for safe and idempotent API requests, and log each failure.
5. **Truncated model output**  
   Inspect `max_tokens` and `model_context_window_exceeded`. Mark the response as incomplete and block malformed text or truncated tool parameters from flowing downstream.
6. **Invalid tool arguments**  
   Validate inputs via Pydantic, return structured validation errors back to the model, and allow a bounded number of self-correction attempts.
7. **Unknown tool request**  
   Reject execution of unregistered tools and return a structured error to the model instead of dynamically invoking arbitrary names.
8. **Tool timeout or exception**  
   Enforce a timeout for every tool execution, catch exceptions, and map failures into explicit tool results or initiate a safe shutdown.
9. **Infinite agent loop**  
   Enforce a hard limit on maximum iterations; halt execution upon reaching the threshold and persist an audit trail capturing the stop state.
10. **Spend limit exceeded**  
    Track input and output tokens per turn; intercept and block subsequent model invocations before breaching the operational budget.
11. **User cancellation**  
    Trap terminal interrupts, abort active tasks gracefully, and preserve recoverable state.
## Decisions
### Raw HTTPX instead of the Anthropic SDK
Using HTTPX in Phase A forces a direct understanding of headers, request bodies, content blocks, stop reasons, and error responses, preventing the SDK from hiding these protocol mechanics. The trade-off is manually maintaining serialization, error classification, and compatibility; however, production projects may transition back to the official SDK once the underlying protocol is fully understood.
### Pyright strict mode
Strict type checking establishes explicit contracts for function inputs, outputs, and nullability. This aids in auditing both human-written and AI-generated code, catching type mismatches before runtime. It complements, but does not replace, runtime validation and automated testing.
### Haiku for development
Developing the Agent Loop, tool protocols, and error-handling mechanics does not demand the strongest reasoning model. Utilizing Haiku reduces operational costs and shortens feedback latency. While subsequent evaluations will baseline against stronger models, the runtime architecture itself must remain model-agnostic.
