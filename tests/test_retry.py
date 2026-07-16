import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from forge.client import AnthropicClient
from forge.errors import (
    AmbiguousCompletionBudgetError,
    APIConnectionError,
    AuthenticationError,
    BudgetExceededError,
)
from forge.models import CreateMessageRequest, Message, MessageResponse
from forge.retry import RetryingMessageClient
from forge.streaming import StreamObserver

FIXTURES = Path(__file__).parent / "fixtures"
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_request() -> CreateMessageRequest:
    return CreateMessageRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        messages=[Message(role="user", content="Hello")],
    )


def load_success() -> dict[str, Any]:
    data = json.loads((FIXTURES / "exercise_c_no_tool.json").read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def load_stream(name: str = "streaming_text.sse") -> bytes:
    return (FIXTURES / name).read_bytes()


def error_body(error_type: str, message: str) -> dict[str, object]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }


async def test_chaos_sequence_retries_with_injected_sleep_and_jitter() -> None:
    calls = 0
    sleeps: list[float] = []
    jitter_bounds: list[tuple[float, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        match calls:
            case 1:
                return httpx.Response(
                    429,
                    headers={"retry-after": "1"},
                    json=error_body("rate_limit_error", "Slow down."),
                )
            case 2:
                return httpx.Response(
                    529,
                    json=error_body("overloaded_error", "Busy."),
                )
            case 3:
                raise httpx.ReadTimeout("Timed out.", request=request)
            case 4:
                return httpx.Response(200, content=load_stream())
            case _:
                raise AssertionError("Too many attempts")

    def half_jitter(lower: float, upper: float) -> float:
        jitter_bounds.append((lower, upper))
        return upper / 2

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as raw_client:
        client = RetryingMessageClient(
            raw_client,
            sleep=record_sleep,
            uniform=half_jitter,
            max_attempts=4,
            base_delay_seconds=0.5,
            max_delay_seconds=8.0,
            max_total_delay_seconds=10.0,
        )
        response = await client.create_message(make_request())

    assert isinstance(response, MessageResponse)
    assert calls == 4
    assert sleeps == [1.0, 0.5, 1.0]
    assert jitter_bounds == [(0.0, 1.0), (0.0, 2.0)]


async def test_ambiguous_completion_is_retried_at_most_once() -> None:
    calls = 0
    sleeps: list[float] = []

    class ScriptedClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            nonlocal calls
            calls += 1
            raise APIConnectionError(
                "Response lost.",
                request_may_have_completed=True,
            )

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = RetryingMessageClient(
        ScriptedClient(),
        sleep=record_sleep,
        uniform=lambda lower, upper: upper,
        max_attempts=4,
    )

    with pytest.raises(APIConnectionError, match="Response lost"):
        await client.create_message(make_request())

    assert calls == 2
    assert len(sleeps) == 1


async def test_budget_refusal_after_an_ambiguous_attempt_preserves_its_cause() -> None:
    calls = 0
    original = APIConnectionError("Response lost.", request_may_have_completed=True)

    class ScriptedClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise original
            raise BudgetExceededError("Retry cannot fit under the cap.")

    async def no_sleep(seconds: float) -> None:
        pass

    client = RetryingMessageClient(
        ScriptedClient(),
        sleep=no_sleep,
        uniform=lambda lower, upper: upper,
        max_attempts=4,
    )

    with pytest.raises(AmbiguousCompletionBudgetError) as exc_info:
        await client.create_message(make_request())

    assert calls == 2
    assert exc_info.value.__cause__ is original


async def test_non_retryable_authentication_error_is_propagated_immediately() -> None:
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            json=error_body("authentication_error", "Invalid API key."),
        )

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as raw_client:
        client = RetryingMessageClient(raw_client, sleep=record_sleep)
        with pytest.raises(AuthenticationError):
            await client.create_message(make_request())

    assert calls == 1
    assert sleeps == []
