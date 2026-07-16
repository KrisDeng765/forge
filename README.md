# Forge

Forge is a command-line agent runtime built from raw HTTPX and typed Pydantic models. It sends typed Messages API requests, preserves provider response data, and will orchestrate tool-using conversations without an agent framework.

**Status:** Phase C implementation complete — Forge now uses async HTTP streaming, renders
text as SSE deltas arrive, runs independent local tools concurrently, and can persist a
safe interruption snapshot. The offline suite verifies these behaviours; the outstanding
live capture/transcript acceptance requires an `ANTHROPIC_API_KEY` in the execution
environment.

Run one task with an optional Ctrl-C snapshot path:

```sh
ANTHROPIC_API_KEY=... uv run python main.py \
  --state-file ./forge-interrupted.json \
  "What is the weather in London and what time is it?"
```

The runtime writes response text to standard output as it arrives. Tool progress and any
stream-retry boundary are written to standard error. A snapshot is created only after an
interrupt and only when `--state-file` is provided; it never contains the API key and is an
audit/reconciliation record, not an automatic-resume file.
