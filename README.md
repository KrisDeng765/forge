# Forge

Forge is a command-line agent runtime built from raw HTTPX and typed Pydantic models. It sends typed Messages API requests, preserves provider response data, and will orchestrate tool-using conversations without an agent framework.

**Status:** Phase A complete — typed API client, ConversationState, ToolRegistry, Agent
Loop, CLI, and live-tool milestone are complete. Phase B is in progress: bounded retries,
spend reservations, self-correction limits, and synchronous tool timeouts are implemented
with deterministic tests.
