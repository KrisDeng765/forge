import asyncio
import json
from pathlib import Path
from threading import Event
from typing import Any, cast

import pytest
from pydantic import Field

from forge.errors import AmbiguousCompletionBudgetError, BudgetExceededError
from forge.loop import (
    AgentLoop,
    ApprovalPolicy,
    LoopProtocolError,
    MaxIterationsExceeded,
    MessageClient,
    UnsupportedStopReasonError,
)
from forge.models import (
    CreateMessageRequest,
    JsonObject,
    MessageResponse,
    ToolResultBlock,
    ToolUseBlock,
)
from forge.registry import ToolInputModel, ToolRegistry
from forge.state import ConversationState
from forge.streaming import StreamObserver

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeClient:
    def __init__(self, responses: list[MessageResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[CreateMessageRequest] = []

    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse:
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

class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_tool_call(self, tool_use: ToolUseBlock) -> None:
        self.events.append(f"call:{tool_use.name}")

    def on_tool_result(self, result: ToolResultBlock) -> None:
        self.events.append(f"result:{result.tool_use_id}:{result.is_error}")

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
    client: MessageClient,
    state: ConversationState,
    registry: ToolRegistry,
    max_iterations: int = 4,
    approval_policy: ApprovalPolicy | None = None,
    observer: RecordingObserver | None = None,
    tool_timeout_seconds: float = 10.0,
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
        observer=observer,
        tool_timeout_seconds=tool_timeout_seconds,
    )


async def test_tool_round_builds_fresh_requests_and_replayable_transcript() -> None:
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

    result = await make_loop(client=client, state=state, registry=registry).run()

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


async def test_unknown_tool_returns_an_error_result_and_continues() -> None:
    state = make_state()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_b_tool_result_answer.json"),
        ]
    )

    result = await make_loop(
        client=client,
        state=state,
        registry=ToolRegistry(),
    ).run()

    tool_result = state.snapshot()[2].model_dump(mode="json", exclude_none=True)["content"][0]
    assert result.status == "completed"
    assert tool_result["is_error"] is True
    assert "Unknown tool" in tool_result["content"]
    assert len(client.requests) == 2


async def test_max_iterations_counts_api_requests() -> None:
    state = make_state()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_a_tool_use.json"),
        ]
    )

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        await make_loop(
            client=client,
            state=state,
            registry=ToolRegistry(),
            max_iterations=2,
        ).run()

    assert exc_info.value.max_iterations == 2
    assert len(client.requests) == 2


async def test_multiple_tool_uses_create_one_ordered_results_message() -> None:
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

    result = await make_loop(client=client, state=state, registry=registry).run()

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


async def test_tool_use_reason_without_a_tool_block_is_a_protocol_error() -> None:
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
        await make_loop(client=client, state=state, registry=ToolRegistry()).run()

    assert [message.role for message in state.snapshot()] == ["user"]


async def test_denied_tool_is_returned_as_an_error_without_execution() -> None:
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

    result = await make_loop(
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


async def test_broken_approval_fails_closed_without_leaking_its_exception() -> None:
    class BrokenApproval:
        def approve(self, tool_name: str, tool_input: JsonObject) -> bool:
            raise RuntimeError("internal policy secret")

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

    result = await make_loop(
        client=client,
        state=state,
        registry=registry,
        approval_policy=BrokenApproval(),
    ).run()

    tool_result = state.snapshot()[2].model_dump(mode="json", exclude_none=True)["content"][0]
    assert result.status == "completed"
    assert calls == []
    assert tool_result["is_error"] is True
    assert "internal policy secret" not in tool_result["content"]
    assert "the tool was not run" in tool_result["content"]


async def test_truncated_tool_response_never_executes_or_mutates_state() -> None:
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

    result = await make_loop(client=client, state=state, registry=registry).run()

    assert result.status == "truncated"
    assert calls == []
    assert [message.role for message in state.snapshot()] == ["user"]


async def test_unknown_stop_reason_preserves_state_and_raises_a_typed_error() -> None:
    state = make_state()
    response = make_response(
        content=[{"type": "text", "text": "Unexpected."}],
        stop_reason="future_stop_reason",
    )
    client = FakeClient([response])

    with pytest.raises(UnsupportedStopReasonError) as exc_info:
        await make_loop(client=client, state=state, registry=ToolRegistry()).run()

    assert exc_info.value.response is response
    assert [message.role for message in state.snapshot()] == ["user"]


async def test_duplicate_tool_use_ids_fail_before_any_tool_executes() -> None:
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
        await make_loop(client=client, state=state, registry=registry).run()

    assert calls == []
    assert [message.role for message in state.snapshot()] == ["user"]

async def test_observer_sees_a_tool_call_and_its_result() -> None:
    calls: list[str] = []
    registry = ToolRegistry()
    register_weather_tool(registry, calls)
    observer = RecordingObserver()
    client = FakeClient(
        [
            load_response("exercise_a_tool_use.json"),
            load_response("exercise_b_tool_result_answer.json"),
        ]
    )

    result = await make_loop(
        client=client,
        state=make_state(),
        registry=registry,
        observer=observer,
    ).run()

    assert result.status == "completed"
    assert observer.events == [
        "call:get_weather",
        "result:toolu_01VpMy2As7txuxpAgFDgBEKN:None",
    ]


@pytest.mark.parametrize(
    ("stop_reason", "stop_sequence", "expected_status", "expected_messages"),
    [
        ("end_turn", None, "completed", ["user", "assistant"]),
        ("stop_sequence", "END", "stop_sequence", ["user", "assistant"]),
        ("refusal", None, "refusal", ["user", "assistant"]),
        ("max_tokens", None, "truncated", ["user"]),
        (
            "model_context_window_exceeded",
            None,
            "context_limit",
            ["user"],
        ),
    ],
)
async def test_terminal_stop_reason_dispatches_with_documented_state_mutation(
    stop_reason: str,
    stop_sequence: str | None,
    expected_status: str,
    expected_messages: list[str],
) -> None:
    state = make_state()
    client = FakeClient(
        [
            make_response(
                content=[{"type": "text", "text": "Terminal."}],
                stop_reason=stop_reason,
                stop_sequence=stop_sequence,
            )
        ]
    )

    result = await make_loop(client=client, state=state, registry=ToolRegistry()).run()

    assert result.status == expected_status
    assert [message.role for message in state.snapshot()] == expected_messages


async def test_pause_turn_appends_then_continues_without_a_tool_result() -> None:
    state = make_state()
    client = FakeClient(
        [
            make_response(
                content=[{"type": "text", "text": "Still working."}],
                stop_reason="pause_turn",
            ),
            make_response(
                content=[{"type": "text", "text": "Done."}],
                stop_reason="end_turn",
            ),
        ]
    )

    result = await make_loop(client=client, state=state, registry=ToolRegistry()).run()

    assert result.status == "completed"
    assert [message.role for message in state.snapshot()] == [
        "user",
        "assistant",
        "assistant",
    ]


async def test_repeated_validation_error_stops_before_the_iteration_limit() -> None:
    state = make_state()
    invalid_tool_round = make_response(
        content=[
            tool_use_payload(
                tool_use_id="toolu_invalid",
                name="get_weather",
                tool_input={"city": 42},
            )
        ],
        stop_reason="tool_use",
    )
    client = FakeClient([invalid_tool_round, invalid_tool_round])
    registry = ToolRegistry()
    register_weather_tool(registry, [])

    result = await make_loop(client=client, state=state, registry=registry).run()

    assert result.status == "tool_validation_stalled"
    assert len(client.requests) == 2
    assert [message.role for message in state.snapshot()] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]


async def test_budget_exceeded_has_a_terminal_result_without_a_response() -> None:
    class OverBudgetClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            raise BudgetExceededError("Budget exhausted.")

    result = await make_loop(
        client=OverBudgetClient(),
        state=make_state(),
        registry=ToolRegistry(),
    ).run()

    assert result.status == "budget_exceeded"
    assert result.response is None
    assert result.text == ""


async def test_ambiguous_completion_is_a_distinct_terminal_result() -> None:
    class AmbiguousClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            raise AmbiguousCompletionBudgetError("A retry could not be funded.")

    result = await make_loop(
        client=AmbiguousClient(),
        state=make_state(),
        registry=ToolRegistry(),
    ).run()

    assert result.status == "ambiguous_completion"
    assert result.response is None
    assert result.text == ""


async def test_tool_timeout_returns_a_correlated_error_result() -> None:
    started = Event()
    release = Event()
    registry = ToolRegistry()

    @registry.tool(
        description="A deliberately slow weather lookup.",
        input_model=CityInput,
    )
    def get_weather(city: str) -> str:
        started.set()
        release.wait(timeout=1.0)
        return f"Weather for {city}"

    assert callable(get_weather)

    client = FakeClient(
        [
            make_response(
                content=[
                    tool_use_payload(
                        tool_use_id="toolu_slow",
                        name="get_weather",
                        tool_input={"city": "London"},
                    )
                ],
                stop_reason="tool_use",
            ),
            make_response(
                content=[{"type": "text", "text": "Tool unavailable."}],
                stop_reason="end_turn",
            ),
        ]
    )
    state = make_state()

    try:
        result = await make_loop(
            client=client,
            state=state,
            registry=registry,
            tool_timeout_seconds=0.01,
        ).run()
        tool_result = state.snapshot()[2].content

        assert result.status == "completed"
        assert started.wait(timeout=1.0)
        assert isinstance(tool_result, list)
        timeout_result = tool_result[0]
        assert isinstance(timeout_result, ToolResultBlock)
        assert timeout_result.is_error is True
        assert "exceeded its 0.01-second timeout" in str(timeout_result.content)
    finally:
        release.set()


async def test_async_tools_run_concurrently_but_results_keep_block_order() -> None:
    completion_order: list[str] = []
    registry = ToolRegistry()

    @registry.tool(
        description="A slow async weather lookup.",
        input_model=CityInput,
    )
    async def get_weather(city: str) -> str:
        await asyncio.sleep(0.20)
        completion_order.append("weather")
        return f"Weather for {city}"

    @registry.tool(
        description="A quick async clock lookup.",
        input_model=EmptyInput,
    )
    async def get_clock() -> str:
        await asyncio.sleep(0.05)
        completion_order.append("clock")
        return "12:00 UTC"

    assert callable(get_weather)
    assert callable(get_clock)

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
    state = make_state()
    loop = make_loop(
        client=FakeClient(
            [
                tool_round,
                make_response(
                    content=[{"type": "text", "text": "Done."}],
                    stop_reason="end_turn",
                ),
            ]
        ),
        state=state,
        registry=registry,
    )

    started_at = asyncio.get_running_loop().time()
    result = await loop.run()
    elapsed = asyncio.get_running_loop().time() - started_at

    result_blocks = state.snapshot()[2].model_dump(mode="json")["content"]
    assert result.status == "completed"
    assert completion_order == ["clock", "weather"]
    assert [block["tool_use_id"] for block in result_blocks] == [
        "toolu_weather",
        "toolu_clock",
    ]
    assert elapsed < 0.23


async def test_cancelling_inflight_tools_keeps_the_transcript_at_a_safe_boundary() -> None:
    started = asyncio.Event()
    never_finishes = asyncio.Event()
    registry = ToolRegistry()

    @registry.tool(
        description="A cancellable async weather lookup.",
        input_model=CityInput,
    )
    async def get_weather(city: str) -> str:
        started.set()
        await never_finishes.wait()
        return city

    assert callable(get_weather)

    tool_round = make_response(
        content=[
            tool_use_payload(
                tool_use_id="toolu_cancelled",
                name="get_weather",
                tool_input={"city": "London"},
            )
        ],
        stop_reason="tool_use",
    )
    loop = make_loop(
        client=FakeClient([tool_round]),
        state=make_state(),
        registry=registry,
    )
    task = asyncio.create_task(loop.run())
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    snapshot = loop.interruption_snapshot()
    messages = cast(list[object], snapshot["messages"])
    unresolved = cast(list[dict[str, object]], snapshot["unresolved_tool_uses"])
    assert len(messages) == 1
    assert unresolved[0]["id"] == "toolu_cancelled"
