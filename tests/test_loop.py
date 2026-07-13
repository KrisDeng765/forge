import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import Field

from forge.loop import (
    AgentLoop,
    LoopProtocolError,
    MaxIterationsExceeded,
    UnsupportedStopReasonError,
)
from forge.models import CreateMessageRequest, JsonObject, MessageResponse
from forge.registry import ToolInputModel, ToolRegistry
from forge.state import ConversationState

FIXTURES = Path(__file__).parent / "fixtures"


class FakeClient:
    def __init__(self, responses: list[MessageResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CreateMessageRequest] = []

    def create_message(self, request: CreateMessageRequest) -> MessageResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeClient received more requests than scripted responses.")
        return self._responses.pop(0)


class CityInput(ToolInputModel):
    city: str = Field(description="The city whose weather is requested.")


class EmptyInput(ToolInputModel):
    pass


class DenyAll:
    def approve(self, tool_name: str, tool_input: JsonObject) -> bool:
        return False


def load_response(name: str) -> MessageResponse:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return MessageResponse.model_validate(cast(dict[str, Any], data))


def make_response(
    *,
    content: list[dict[str, Any]],
    stop_reason: str | None,
    stop_sequence: str | None = None,
) -> MessageResponse:
    return MessageResponse.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": stop_sequence,
            "stop_details": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def tool_use_payload(
    *,
    tool_use_id: str,
    name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": tool_use_id,
        "name": name,
        "input": tool_input,
    }


def make_state() -> ConversationState:
    state = ConversationState()
    state.append_user_text("What is the weather in London?")
    return state


def register_weather_tool(registry: ToolRegistry, calls: list[str]) -> None:
    def get_weather(city: str) -> str:
        calls.append(f"weather:{city}")
        return f"Weather for {city}: rainy"

    registry.tool(
        description="Get the current weather for a city.",
        input_model=CityInput,
    )(get_weather)


def register_clock_tool(registry: ToolRegistry, calls: list[str]) -> None:
    def get_clock() -> str:
        calls.append("clock")
        return "12:00 UTC"

    registry.tool(
        description="Get the current UTC time.",
        input_model=EmptyInput,
    )(get_clock)


def make_loop(
    *,
    client: FakeClient,
    state: ConversationState,
    registry: ToolRegistry,
    max_iterations: int = 4,
    approval_policy: DenyAll | None = None,
) -> AgentLoop:
    return AgentLoop(
        client=client,
        state=state,
        registry=registry,
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        system="Answer concisely.",
        max_iterations=max_iterations,
        approval_policy=approval_policy,
    )


def test_tool_round_builds_fresh_requests_and_replayable_transcript() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    register_weather_tool(registry, calls)
    state = make_state()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_b_tool_result_answer.json"),
        ]
    )

    result = make_loop(client=client, state=state, registry=registry).run()

    assert result.status == "completed"
    assert "14°C" in result.text
    assert calls == ["weather:London"]
    assert len(client.requests) == 2
    assert client.requests[0].system == "Answer concisely."
    assert [tool.name for tool in client.requests[0].tools or []] == ["get_weather"]
    assert [message.role for message in client.requests[0].messages] == ["user"]
    assert [message.role for message in client.requests[1].messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert client.requests[1].messages[1].model_dump(mode="json")["content"][0][
        "caller"
    ] == {"type": "direct"}
    assert [message.role for message in state.snapshot()] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_unknown_tool_returns_an_error_result_and_continues() -> None:
    state = make_state()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_b_tool_result_answer.json"),
        ]
    )

    result = make_loop(
        client=client,
        state=state,
        registry=ToolRegistry(),
    ).run()

    tool_result = state.snapshot()[2].model_dump(mode="json", exclude_none=True)["content"][0]
    assert result.status == "completed"
    assert tool_result["is_error"] is True
    assert "Unknown tool" in tool_result["content"]
    assert len(client.requests) == 2


def test_max_iterations_counts_api_requests() -> None:
    state = make_state()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_a_tool_use.json"),
        ]
    )

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        make_loop(
            client=client,
            state=state,
            registry=ToolRegistry(),
            max_iterations=2,
        ).run()

    assert exc_info.value.max_iterations == 2
    assert len(client.requests) == 2


def test_multiple_tool_uses_create_one_ordered_results_message() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    register_weather_tool(registry, calls)
    register_clock_tool(registry, calls)
    state = make_state()
    tool_round = make_response(
        content=[
            tool_use_payload(
                tool_use_id="toolu_weather",
                name="get_weather",
                tool_input={"city": "London"},
            ),
            tool_use_payload(
                tool_use_id="toolu_clock",
                name="get_clock",
                tool_input={},
            ),
        ],
        stop_reason="tool_use",
    )
    client = FakeClient(
        [tool_round, load_response("exercise_b_tool_result_answer.json")]
    )

    result = make_loop(client=client, state=state, registry=registry).run()

    transcript = state.snapshot()
    results_message = transcript[2].model_dump(mode="json", exclude_none=True)
    assert result.status == "completed"
    assert calls == ["weather:London", "clock"]
    assert [message.role for message in transcript] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [block["tool_use_id"] for block in results_message["content"]] == [
        "toolu_weather",
        "toolu_clock",
    ]


def test_tool_use_reason_without_a_tool_block_is_a_protocol_error() -> None:
    state = make_state()
    client = FakeClient(
        [
            make_response(
                content=[{"type": "text", "text": "I need a tool."}],
                stop_reason="tool_use",
            )
        ]
    )

    with pytest.raises(LoopProtocolError):
        make_loop(client=client, state=state, registry=ToolRegistry()).run()

    assert [message.role for message in state.snapshot()] == ["user"]


def test_denied_tool_is_returned_as_an_error_without_execution() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    register_weather_tool(registry, calls)
    state = make_state()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_b_tool_result_answer.json"),
        ]
    )

    result = make_loop(
        client=client,
        state=state,
        registry=registry,
        approval_policy=DenyAll(),
    ).run()

    tool_result = state.snapshot()[2].model_dump(mode="json", exclude_none=True)["content"][0]
    assert result.status == "completed"
    assert calls == []
    assert tool_result["is_error"] is True
    assert "not approved" in tool_result["content"]


def test_truncated_tool_response_never_executes_or_mutates_state() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    register_weather_tool(registry, calls)
    state = make_state()
    client = FakeClient(
        [
            make_response(
                content=[
                    tool_use_payload(
                        tool_use_id="toolu_123",
                        name="get_weather",
                        tool_input={"city": "London"},
                    )
                ],
                stop_reason="max_tokens",
            )
        ]
    )

    result = make_loop(client=client, state=state, registry=registry).run()

    assert result.status == "truncated"
    assert calls == []
    assert [message.role for message in state.snapshot()] == ["user"]


def test_unknown_stop_reason_preserves_state_and_raises_a_typed_error() -> None:
    state = make_state()
    response = make_response(
        content=[{"type": "text", "text": "Unexpected."}],
        stop_reason="future_stop_reason",
    )
    client = FakeClient([response])

    with pytest.raises(UnsupportedStopReasonError) as exc_info:
        make_loop(client=client, state=state, registry=ToolRegistry()).run()

    assert exc_info.value.response is response
    assert [message.role for message in state.snapshot()] == ["user"]


def test_duplicate_tool_use_ids_fail_before_any_tool_executes() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    register_weather_tool(registry, calls)
    state = make_state()
    client = FakeClient(
        [
            make_response(
                content=[
                    tool_use_payload(
                        tool_use_id="toolu_duplicate",
                        name="get_weather",
                        tool_input={"city": "London"},
                    ),
                    tool_use_payload(
                        tool_use_id="toolu_duplicate",
                        name="get_weather",
                        tool_input={"city": "Paris"},
                    ),
                ],
                stop_reason="tool_use",
            )
        ]
    )

    with pytest.raises(LoopProtocolError):
        make_loop(client=client, state=state, registry=registry).run()

    assert calls == []
    assert [message.role for message in state.snapshot()] == ["user"]
