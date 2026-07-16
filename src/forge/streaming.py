"""SSE parsing and Messages-stream accumulation without an SDK."""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast

from forge.models import MessageResponse


class StreamProtocolError(ValueError):
    """Raised when a successful SSE response cannot form one complete message."""


class StreamObserver(Protocol):
    """Receive safe, incremental runtime events while a message is streaming."""

    def on_text_delta(self, text: str) -> None: ...

    def on_input_tokens(self, input_tokens: int) -> None: ...

    def on_stream_retry(self) -> None: ...


class NullStreamObserver:
    def on_text_delta(self, text: str) -> None:
        pass

    def on_input_tokens(self, input_tokens: int) -> None:
        pass

    def on_stream_retry(self) -> None:
        pass


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: str


class SSEParser:
    """Incrementally parse the small SSE subset emitted by Messages streaming.

    HTTP chunks are unrelated to SSE event boundaries, so bytes are retained until a
    complete line and then a blank-line event delimiter have arrived.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._event_name: str | None = None
        self._data_lines: list[str] = []

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        self._buffer.extend(chunk)
        events: list[SSEEvent] = []

        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                break

            line_bytes = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if line_bytes.endswith(b"\r"):
                line_bytes = line_bytes[:-1]

            try:
                line = line_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StreamProtocolError("SSE line is not valid UTF-8.") from exc

            if not line:
                event = self._finish_event()
                if event is not None:
                    events.append(event)
                continue
            if line.startswith(":"):
                continue

            field, separator, value = line.partition(":")
            if separator and value.startswith(" "):
                value = value[1:]
            if field == "event":
                self._event_name = value
            elif field == "data":
                self._data_lines.append(value)

        return events

    def finish(self) -> None:
        """Reject an EOF that cuts off a line or an SSE event."""

        if self._buffer or self._event_name is not None or self._data_lines:
            raise StreamProtocolError("SSE stream ended before an event delimiter.")

    def _finish_event(self) -> SSEEvent | None:
        if not self._data_lines:
            self._event_name = None
            return None

        event = SSEEvent(
            event=self._event_name or "message",
            data="\n".join(self._data_lines),
        )
        self._event_name = None
        self._data_lines = []
        return event


@dataclass
class _PendingBlock:
    index: int
    payload: dict[str, Any]
    json_fragments: list[str]
    stopped: bool = False
    parsed_input: dict[str, Any] | None = None
    input_error: StreamProtocolError | None = None


class StreamAccumulator:
    """Turn Messages API events into Forge's existing completed response model."""

    _TRUNCATION_REASONS = {"max_tokens", "model_context_window_exceeded"}

    def __init__(self, observer: StreamObserver | None = None) -> None:
        self._observer = observer if observer is not None else NullStreamObserver()
        self._message_start: dict[str, Any] | None = None
        self._input_tokens: int | None = None
        self._output_tokens: int | None = None
        self._stop_reason: str | None = None
        self._stop_sequence: str | None = None
        self._stop_details: dict[str, Any] | None = None
        self._blocks: dict[int, _PendingBlock] = {}
        self._message_delta_seen = False
        self._message_stop_seen = False

    def consume(self, event: SSEEvent) -> None:
        try:
            raw: object = json.loads(event.data)
        except json.JSONDecodeError as exc:
            raise StreamProtocolError("SSE event data is not valid JSON.") from exc
        if not isinstance(raw, dict):
            raise StreamProtocolError("SSE event data must be a JSON object.")

        payload: dict[str, Any] = cast(dict[str, Any], raw)
        payload_type = payload.get("type")
        if payload_type != event.event:
            raise StreamProtocolError(
                f"SSE event name {event.event!r} does not match payload type {payload_type!r}."
            )

        match event.event:
            case "ping":
                return
            case "message_start":
                self._consume_message_start(payload)
            case "content_block_start":
                self._consume_block_start(payload)
            case "content_block_delta":
                self._consume_block_delta(payload)
            case "content_block_stop":
                self._consume_block_stop(payload)
            case "message_delta":
                self._consume_message_delta(payload)
            case "message_stop":
                self._message_stop_seen = True
            case _:
                raise StreamProtocolError(f"Unsupported SSE event type: {event.event!r}.")

    def finish(self) -> MessageResponse:
        if self._message_start is None:
            raise StreamProtocolError("SSE stream did not include message_start.")
        if not self._message_delta_seen or not self._message_stop_seen:
            raise StreamProtocolError("SSE stream ended before message completion.")
        if self._input_tokens is None or self._output_tokens is None:
            raise StreamProtocolError("SSE stream did not include complete usage.")

        blocks = self._final_blocks()
        response_payload = deepcopy(self._message_start)
        response_payload.update(
            {
                "content": blocks,
                "stop_reason": self._stop_reason,
                "stop_sequence": self._stop_sequence,
                "stop_details": self._stop_details,
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
            }
        )
        try:
            return MessageResponse.model_validate(response_payload)
        except ValueError as exc:
            raise StreamProtocolError("Completed SSE response failed model validation.") from exc

    def _consume_message_start(self, payload: dict[str, Any]) -> None:
        if self._message_start is not None:
            raise StreamProtocolError("SSE stream contains more than one message_start.")
        message: object = payload.get("message")
        if not isinstance(message, dict):
            raise StreamProtocolError("message_start requires a message object.")
        message_payload: dict[str, Any] = cast(dict[str, Any], message)
        usage: object = message_payload.get("usage")
        if not isinstance(usage, dict):
            raise StreamProtocolError("message_start requires integer input_tokens.")
        usage_payload: dict[str, Any] = cast(dict[str, Any], usage)
        input_tokens: object = usage_payload.get("input_tokens")
        if not isinstance(input_tokens, int):
            raise StreamProtocolError("message_start requires integer input_tokens.")

        self._message_start = message_payload
        self._input_tokens = input_tokens
        self._observer.on_input_tokens(self._input_tokens)

    def _consume_block_start(self, payload: dict[str, Any]) -> None:
        self._require_started("content_block_start")
        index = self._require_index(payload)
        block: object = payload.get("content_block")
        if not isinstance(block, dict):
            raise StreamProtocolError("content_block_start requires a typed content block.")
        block_payload: dict[str, Any] = cast(dict[str, Any], block)
        if not isinstance(block_payload.get("type"), str):
            raise StreamProtocolError("content_block_start requires a typed content block.")
        if index in self._blocks:
            raise StreamProtocolError(f"Content block index {index} started twice.")

        self._blocks[index] = _PendingBlock(
            index=index,
            payload=deepcopy(block_payload),
            json_fragments=[],
        )

    def _consume_block_delta(self, payload: dict[str, Any]) -> None:
        self._require_started("content_block_delta")
        block = self._require_open_block(self._require_index(payload))
        delta: object = payload.get("delta")
        if not isinstance(delta, dict):
            raise StreamProtocolError("content_block_delta requires a typed delta.")
        delta_payload: dict[str, Any] = cast(dict[str, Any], delta)
        if not isinstance(delta_payload.get("type"), str):
            raise StreamProtocolError("content_block_delta requires a typed delta.")

        delta_type: object = delta_payload["type"]
        block_type = block.payload.get("type")
        if delta_type == "text_delta" and block_type == "text":
            text: object = delta_payload.get("text")
            if not isinstance(text, str):
                raise StreamProtocolError("text_delta requires text.")
            existing = block.payload.get("text", "")
            if not isinstance(existing, str):
                raise StreamProtocolError("text block started with non-string text.")
            block.payload["text"] = existing + text
            self._observer.on_text_delta(text)
            return

        if delta_type == "input_json_delta" and block_type == "tool_use":
            partial_json: object = delta_payload.get("partial_json")
            if not isinstance(partial_json, str):
                raise StreamProtocolError("input_json_delta requires partial_json.")
            block.json_fragments.append(partial_json)
            return

        raise StreamProtocolError(
            f"Delta {delta_type!r} cannot be applied to block type {block_type!r}."
        )

    def _consume_block_stop(self, payload: dict[str, Any]) -> None:
        self._require_started("content_block_stop")
        block = self._require_open_block(self._require_index(payload))
        block.stopped = True
        if block.payload.get("type") != "tool_use":
            return

        # Argument-free tools commonly start with `input: {}` and emit no
        # input_json_delta at all. In that valid case the start block already
        # carries the complete input; only non-empty streamed inputs need JSON
        # reconstruction from their fragments.
        streamed_input = "".join(block.json_fragments)
        if not streamed_input:
            initial_input = block.payload.get("input")
            if isinstance(initial_input, dict):
                block.parsed_input = deepcopy(cast(dict[str, Any], initial_input))
                return
            block.input_error = StreamProtocolError(
                f"Tool block {block.index} has neither input JSON nor an input object."
            )
            return

        try:
            parsed = json.loads(streamed_input)
        except json.JSONDecodeError:
            block.input_error = StreamProtocolError(
                f"Tool block {block.index} ended with incomplete JSON."
            )
            return
        if not isinstance(parsed, dict):
            block.input_error = StreamProtocolError(
                f"Tool block {block.index} input must decode to an object."
            )
            return
        block.parsed_input = cast(dict[str, Any], parsed)

    def _consume_message_delta(self, payload: dict[str, Any]) -> None:
        self._require_started("message_delta")
        if self._message_delta_seen:
            raise StreamProtocolError("SSE stream contains more than one message_delta.")
        delta: object = payload.get("delta")
        usage: object = payload.get("usage")
        if not isinstance(delta, dict) or not isinstance(usage, dict):
            raise StreamProtocolError("message_delta requires delta and usage objects.")
        delta_payload: dict[str, Any] = cast(dict[str, Any], delta)
        usage_payload: dict[str, Any] = cast(dict[str, Any], usage)
        if not isinstance(usage_payload.get("output_tokens"), int):
            raise StreamProtocolError("message_delta requires integer output_tokens.")
        output_tokens: object = usage_payload["output_tokens"]
        if not isinstance(output_tokens, int):
            raise StreamProtocolError("message_delta requires integer output_tokens.")
        stop_reason: object = delta_payload.get("stop_reason")
        stop_sequence: object = delta_payload.get("stop_sequence")
        stop_details: object = delta_payload.get("stop_details")
        if stop_reason is not None and not isinstance(stop_reason, str):
            raise StreamProtocolError("message_delta stop_reason must be a string or null.")
        if stop_sequence is not None and not isinstance(stop_sequence, str):
            raise StreamProtocolError("message_delta stop_sequence must be a string or null.")
        if stop_details is not None and not isinstance(stop_details, dict):
            raise StreamProtocolError("message_delta stop_details must be an object or null.")

        self._stop_reason = stop_reason
        self._stop_sequence = stop_sequence
        self._stop_details = cast(dict[str, Any] | None, stop_details)
        self._output_tokens = output_tokens
        self._message_delta_seen = True

    def _final_blocks(self) -> list[dict[str, Any]]:
        truncation = self._stop_reason in self._TRUNCATION_REASONS
        final_blocks: list[dict[str, Any]] = []
        for index in sorted(self._blocks):
            block = self._blocks[index]
            block_type = block.payload.get("type")
            if not block.stopped:
                if truncation and block_type == "tool_use":
                    continue
                if truncation and block_type == "text":
                    final_blocks.append(deepcopy(block.payload))
                    continue
                raise StreamProtocolError(f"Content block {index} did not stop.")
            if block_type == "tool_use":
                if block.input_error is not None:
                    if truncation:
                        continue
                    raise block.input_error
                if block.parsed_input is None:
                    raise StreamProtocolError(f"Tool block {index} has no parsed input.")
                complete = deepcopy(block.payload)
                complete["input"] = block.parsed_input
                final_blocks.append(complete)
                continue
            final_blocks.append(deepcopy(block.payload))
        return final_blocks

    def _require_started(self, event_name: str) -> None:
        if self._message_start is None:
            raise StreamProtocolError(f"{event_name} arrived before message_start.")

    @staticmethod
    def _require_index(payload: dict[str, Any]) -> int:
        index = payload.get("index")
        if not isinstance(index, int) or index < 0:
            raise StreamProtocolError("Content block event requires a non-negative index.")
        return index

    def _require_open_block(self, index: int) -> _PendingBlock:
        block = self._blocks.get(index)
        if block is None:
            raise StreamProtocolError(f"Content block index {index} was not started.")
        if block.stopped:
            raise StreamProtocolError(f"Content block index {index} already stopped.")
        return block
