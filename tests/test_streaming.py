import random
from pathlib import Path

import pytest

from forge.models import MessageResponse, ToolUseBlock
from forge.streaming import (
    SSEEvent,
    SSEParser,
    StreamAccumulator,
    StreamObserver,
    StreamProtocolError,
)

FIXTURES = Path(__file__).parent / "fixtures"


class RecordingObserver(StreamObserver):
    def __init__(self) -> None:
        self.text: list[str] = []
        self.input_tokens: list[int] = []
        self.retry_count = 0

    def on_text_delta(self, text: str) -> None:
        self.text.append(text)

    def on_input_tokens(self, input_tokens: int) -> None:
        self.input_tokens.append(input_tokens)

    def on_stream_retry(self) -> None:
        self.retry_count += 1


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def parse_chunks(chunks: list[bytes]) -> list[SSEEvent]:
    parser = SSEParser()
    events: list[SSEEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    parser.finish()
    return events


def accumulate(data: bytes, observer: StreamObserver | None = None) -> MessageResponse:
    accumulator = StreamAccumulator(observer)
    for event in parse_chunks([data]):
        accumulator.consume(event)
    return accumulator.finish()


def test_sse_parser_is_invariant_to_byte_and_random_chunk_boundaries() -> None:
    data = fixture_bytes("streaming_tool_use.sse")
    whole = parse_chunks([data])
    byte_by_byte = parse_chunks([data[index : index + 1] for index in range(len(data))])

    random_source = random.Random(20260715)
    chunks: list[bytes] = []
    cursor = 0
    while cursor < len(data):
        width = random_source.randint(1, 17)
        chunks.append(data[cursor : cursor + width])
        cursor += width
    randomized = parse_chunks(chunks)

    assert byte_by_byte == whole
    assert randomized == whole
    assert [event.event for event in whole] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_sse_parser_rejects_a_truncated_line_or_event() -> None:
    parser = SSEParser()
    parser.feed(b"event: ping\ndata: {\"type\":\"ping\"}")

    with pytest.raises(StreamProtocolError, match="delimiter"):
        parser.finish()


def test_accumulator_emits_text_immediately_and_matches_complete_response() -> None:
    observer = RecordingObserver()
    response = accumulate(fixture_bytes("streaming_text.sse"), observer)

    expected = MessageResponse.model_validate(
        {
            "id": "msg_stream_text",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "Hello from Forge."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
    )

    assert response.model_dump(mode="json") == expected.model_dump(mode="json")
    assert observer.text == ["Hello ", "from Forge."]
    assert observer.input_tokens == [7]


def test_accumulator_reassembles_tool_json_only_after_block_stop() -> None:
    response = accumulate(fixture_bytes("streaming_tool_use.sse"))

    assert response.stop_reason == "tool_use"
    assert len(response.content) == 1
    tool_use = response.content[0]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.input == {"city": "London"}


def test_accumulator_accepts_an_argument_free_tool_without_json_deltas() -> None:
    accumulator = StreamAccumulator()
    events = [
        SSEEvent(
            event="message_start",
            data=(
                '{"type":"message_start","message":{"id":"msg_empty_tool",'
                '"type":"message","role":"assistant","model":"claude-haiku-4-5-20251001",'
                '"content":[],"stop_reason":null,"stop_sequence":null,'
                '"usage":{"input_tokens":4}}}'
            ),
        ),
        SSEEvent(
            event="content_block_start",
            data=(
                '{"type":"content_block_start","index":0,"content_block":'
                '{"type":"tool_use","id":"toolu_clock","name":"get_utc_time",'
                '"input":{}}}'
            ),
        ),
        SSEEvent(
            event="content_block_delta",
            data=(
                '{"type":"content_block_delta","index":0,"delta":'
                '{"type":"input_json_delta","partial_json":""}}'
            ),
        ),
        SSEEvent(
            event="content_block_stop",
            data='{"type":"content_block_stop","index":0}',
        ),
        SSEEvent(
            event="message_delta",
            data=(
                '{"type":"message_delta","delta":{"stop_reason":"tool_use",'
                '"stop_sequence":null},"usage":{"output_tokens":1}}'
            ),
        ),
        SSEEvent(event="message_stop", data='{"type":"message_stop"}'),
    ]

    for event in events:
        accumulator.consume(event)
    response = accumulator.finish()

    tool_use = response.content[0]
    assert isinstance(tool_use, ToolUseBlock)
    assert tool_use.name == "get_utc_time"
    assert tool_use.input == {}


def test_truncated_tool_json_is_dropped_and_cannot_be_dispatched() -> None:
    response = accumulate(fixture_bytes("streaming_truncated_tool.sse"))

    assert response.stop_reason == "max_tokens"
    assert response.content == []
