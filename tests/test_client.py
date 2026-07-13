import json
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from forge.client import AnthropicClient
from forge.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OverloadedError,
    RateLimitError,
)
from forge.models import CreateMessageRequest, Message, ToolUseBlock

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def make_request() -> CreateMessageRequest:
    return CreateMessageRequest(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[Message(role="user", content="What is the weather in London?")],
    )


def error_body(error_type: str, message: str) -> dict[str, object]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def test_create_message_sends_request_and_parses_success() -> None:
    request_model = make_request()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "fake-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == request_model.model_dump(
            mode="json",
            exclude_none=True,
        )
        return httpx.Response(200, json=load_fixture("exercise_a_tool_use.json"))

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        response = client.create_message(request_model)

    assert response.stop_reason == "tool_use"
    assert any(isinstance(block, ToolUseBlock) for block in response.content)


def test_malformed_success_response_is_a_typed_ambiguous_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>Bad Gateway</html>")

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(APIConnectionError) as exc_info:
            client.create_message(make_request())

    assert exc_info.value.request_may_have_completed is True


def test_missing_api_key_fails_during_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Anthropic API key is required"):
        AnthropicClient()


def test_api_key_defaults_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "environment-key"
        return httpx.Response(200, json=load_fixture("exercise_c_no_tool.json"))

    with AnthropicClient(transport=httpx.MockTransport(handler)) as client:
        response = client.create_message(make_request())

    assert response.stop_reason == "end_turn"


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_statuses_are_classified(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"request-id": "req_auth"},
            json=error_body("authentication_error", "Invalid API key."),
        )

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(AuthenticationError) as exc_info:
            client.create_message(make_request())

    assert exc_info.value.status_code == status_code
    assert exc_info.value.request_id == "req_auth"
    assert exc_info.value.error_response.error.message == "Invalid API key."


def test_rate_limit_preserves_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"request-id": "req_rate", "retry-after": "7"},
            json=error_body("rate_limit_error", "Rate limit exceeded."),
        )

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RateLimitError) as exc_info:
            client.create_message(make_request())

    assert exc_info.value.retry_after == 7.0
    assert exc_info.value.request_id == "req_rate"


def test_overload_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529,
            json=error_body("overloaded_error", "API is overloaded."),
        )

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(OverloadedError) as exc_info:
            client.create_message(make_request())

    assert exc_info.value.status_code == 529


def test_other_non_success_status_uses_generic_status_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json=error_body("api_error", "Internal server error."),
        )

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(APIStatusError) as exc_info:
            client.create_message(make_request())

    assert type(exc_info.value) is APIStatusError
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize(
    ("transport_error", "phase", "request_may_have_completed"),
    [
        (httpx.ConnectTimeout, "connect", False),
        (httpx.ReadTimeout, "read", True),
    ],
)
def test_timeout_errors_preserve_completion_semantics(
    transport_error: type[httpx.TimeoutException],
    phase: str,
    request_may_have_completed: bool,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise transport_error("Timed out.", request=request)

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(APITimeoutError) as exc_info:
            client.create_message(make_request())

    assert exc_info.value.phase == phase
    assert (
        exc_info.value.request_may_have_completed is request_may_have_completed
    )


def test_connection_error_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection failed.", request=request)

    with AnthropicClient(
        api_key="fake-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(APIConnectionError) as exc_info:
            client.create_message(make_request())

    assert type(exc_info.value) is APIConnectionError
    assert exc_info.value.request_may_have_completed is False
