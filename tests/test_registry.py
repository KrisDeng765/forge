from typing import Any, cast

import pytest
from pydantic import Field, ValidationError

from forge.models import ToolUseBlock
from forge.registry import ToolInputModel, ToolRegistry


class CityInput(ToolInputModel):
    city: str = Field(description="The city whose weather is requested.")


def make_tool_use(name: str, tool_input: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(
        type="tool_use",
        id="toolu_123",
        name=name,
        input=tool_input,
    )


def test_registering_a_tool_creates_a_definition_and_executes_it() -> None:
    registry = ToolRegistry()

    @registry.tool(
        description="Get the current weather for a city.",
        input_model=CityInput,
    )
    def get_weather(city: str) -> str:
        return f"Weather for {city}: rainy"

    assert get_weather.__name__ == "get_weather"
    definitions = registry.definitions()
    assert len(definitions) == 1
    assert definitions[0].name == "get_weather"
    assert definitions[0].description == "Get the current weather for a city."
    assert definitions[0].input_schema["properties"]["city"]["description"] == (
        "The city whose weather is requested."
    )

    result = registry.execute(make_tool_use("get_weather", {"city": "London"}))

    assert result.model_dump(mode="json", exclude_none=True) == {
        "type": "tool_result",
        "tool_use_id": "toolu_123",
        "content": "Weather for London: rainy",
    }


def test_invalid_input_returns_an_error_without_calling_the_tool() -> None:
    registry = ToolRegistry()
    called = False

    @registry.tool(
        description="Get the current weather for a city.",
        input_model=CityInput,
    )
    def get_weather(city: str) -> str:
        nonlocal called
        called = True
        return city

    assert callable(get_weather)
    result = registry.execute(
        make_tool_use(
            "get_weather",
            {"city": "London", "unexpected": True},
        )
    )

    assert result.is_error is True
    assert result.tool_use_id == "toolu_123"
    assert isinstance(result.content, str)
    assert "Invalid input" in result.content
    assert called is False


def test_unknown_tool_returns_a_structured_error_result() -> None:
    result = ToolRegistry().execute(make_tool_use("missing_tool", {}))

    assert result.is_error is True
    assert result.tool_use_id == "toolu_123"
    assert isinstance(result.content, str)
    assert "Unknown tool" in result.content


def test_tool_exception_returns_a_structured_error_result() -> None:
    registry = ToolRegistry()

    @registry.tool(
        description="Get the current weather for a city.",
        input_model=CityInput,
    )
    def broken_weather(city: str) -> str:
        raise RuntimeError("weather backend unavailable")

    assert callable(broken_weather)
    result = registry.execute(make_tool_use("broken_weather", {"city": "London"}))

    assert result.is_error is True
    assert isinstance(result.content, str)
    assert "weather backend unavailable" in result.content


def test_invalid_tool_return_is_a_structured_error_result() -> None:
    registry = ToolRegistry()

    def invalid_weather(city: str) -> str:
        return cast(str, 42)

    registry.tool(
        description="Get the current weather for a city.",
        input_model=CityInput,
    )(invalid_weather)

    result = registry.execute(make_tool_use("invalid_weather", {"city": "London"}))

    assert result.is_error is True
    assert isinstance(result.content, str)
    assert "must return a string" in result.content


def test_invalid_tool_name_fails_during_registration() -> None:
    registry = ToolRegistry()

    def get_weather(city: str) -> str:
        return city

    with pytest.raises(ValidationError):
        registry.tool(
            name="bad name",
            description="Get the current weather for a city.",
            input_model=CityInput,
        )(get_weather)
