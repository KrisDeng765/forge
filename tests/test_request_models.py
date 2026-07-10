from typing import Any

import pytest
from pydantic import ValidationError

from forge.models import (
    AnyToolChoice,
    AutoToolChoice,
    CreateMessageRequest,
    Message,
    NamedToolChoice,
    NoneToolChoice,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)


def test_create_message_request_dumps_exercise_a_shape() -> None:
    request = CreateMessageRequest(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get the current weather for a city.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ],
        messages=[
            Message(
                role="user",
                content="What is the weather in London?",
            )
        ],
    )

    assert request.model_dump(mode="json", exclude_none=True) == {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in London?",
            }
        ],
    }


def test_create_message_request_dumps_tool_result_followup_shape() -> None:
    request = CreateMessageRequest(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get the current weather for a city.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ],
        messages=[
            Message(
                role="user",
                content="What is the weather in London?",
            ),
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        type="tool_use",
                        id="toolu_123",
                        name="get_weather",
                        input={"city": "London"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        tool_use_id="toolu_123",
                        content="14°C, raining",
                    )
                ],
            ),
        ],
    )

    assert request.model_dump(mode="json", exclude_none=True) == {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in London?",
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "London"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "14°C, raining",
                    }
                ],
            },
        ],
    }


def test_system_is_serialized_as_a_top_level_parameter() -> None:
    request = CreateMessageRequest(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system="Answer briefly.",
        messages=[Message(role="user", content="Hello")],
    )

    payload = request.model_dump(mode="json", exclude_none=True)

    assert payload["system"] == "Answer briefly."
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"type": "auto"}, AutoToolChoice),
        ({"type": "any"}, AnyToolChoice),
        ({"type": "none"}, NoneToolChoice),
        ({"type": "tool", "name": "get_weather"}, NamedToolChoice),
    ],
)
def test_tool_choice_variants_use_the_discriminator(
    payload: dict[str, str],
    expected_type: type[
        AutoToolChoice | AnyToolChoice | NoneToolChoice | NamedToolChoice
    ],
) -> None:
    request_data: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "tool_choice": payload,
    }

    request = CreateMessageRequest.model_validate(request_data)

    assert type(request.tool_choice) is expected_type
    assert request.model_dump(mode="json", exclude_none=True)["tool_choice"] == payload


def test_unknown_tool_choice_type_is_rejected() -> None:
    request_data = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "tool_choice": {"type": "future_choice"},
    }

    with pytest.raises(ValidationError):
        CreateMessageRequest.model_validate(request_data)


def test_request_rejects_unknown_fields() -> None:
    request_data = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_token": 1024,
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CreateMessageRequest.model_validate(request_data)


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_request_requires_positive_max_tokens(max_tokens: int) -> None:
    request_data = {
        "model": "claude-haiku-4-5",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": "Hello"}],
    }

    with pytest.raises(ValidationError):
        CreateMessageRequest.model_validate(request_data)


@pytest.mark.parametrize("name", ["", "get weather", "x" * 65])
def test_tool_definition_rejects_invalid_names(name: str) -> None:
    tool_data = {
        "name": name,
        "description": "Get the current weather for a city.",
        "input_schema": {"type": "object"},
    }

    with pytest.raises(ValidationError):
        ToolDefinition.model_validate(tool_data)
