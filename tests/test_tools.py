from datetime import datetime, timedelta, timezone

from forge.models import ToolUseBlock
from forge.registry import ToolRegistry
from forge.tools import (
    format_utc_time,
    register_calculator,
    register_clock,
    register_default_tools,
    register_weather,
)


def test_calculator_schema_constrains_the_operator_and_returns_a_result() -> None:
    registry = ToolRegistry()
    register_calculator(registry)

    definition = registry.definitions()[0]
    assert definition.name == "calculate"
    assert definition.input_schema["properties"]["op"]["enum"] == [
        "+",
        "-",
        "*",
        "/",
    ]

    result = registry.execute(
        ToolUseBlock(
            type="tool_use",
            id="toolu_calculator",
            name="calculate",
            input={"a": 480, "op": "*", "b": 0.125},
        )
    )

    assert result.content == "60.0"
    assert result.is_error is None


def test_calculator_rejects_an_operator_outside_its_schema() -> None:
    registry = ToolRegistry()
    register_calculator(registry)

    result = registry.execute(
        ToolUseBlock(
            type="tool_use",
            id="toolu_invalid_operator",
            name="calculate",
            input={"a": 1, "op": "%", "b": 2},
        )
    )

    assert result.is_error is True
    assert result.content is not None
    assert "Invalid input" in result.content


def test_calculator_turns_division_by_zero_into_a_tool_error() -> None:
    registry = ToolRegistry()
    register_calculator(registry)

    result = registry.execute(
        ToolUseBlock(
            type="tool_use",
            id="toolu_divide_zero",
            name="calculate",
            input={"a": 1, "op": "/", "b": 0},
        )
    )

    assert result.is_error is True
    assert result.content is not None
    assert "Cannot divide by zero." in result.content


def test_weather_is_deterministic_and_normalizes_location() -> None:
    registry = ToolRegistry()
    register_weather(registry)

    first = registry.execute(
        ToolUseBlock(
            type="tool_use",
            id="toolu_weather_first",
            name="get_weather",
            input={"location": "London"},
        )
    )
    second = registry.execute(
        ToolUseBlock(
            type="tool_use",
            id="toolu_weather_second",
            name="get_weather",
            input={"location": "  LONDON  "},
        )
    )

    assert first.content == "Fictional forecast for London: 14°C and raining."
    assert second.content == first.content
    assert first.is_error is None
    assert second.is_error is None


def test_format_utc_time_converts_an_aware_time_to_utc() -> None:
    london_summer_time = datetime(
        2026,
        7,
        13,
        10,
        5,
        6,
        tzinfo=timezone(timedelta(hours=1)),
    )

    assert (
        format_utc_time(london_summer_time)
        == "Current UTC time: 2026-07-13T09:05:06+00:00"
    )


def test_clock_tool_returns_a_utc_timestamp_without_testing_the_current_second() -> None:
    registry = ToolRegistry()
    register_clock(registry)

    result = registry.execute(
        ToolUseBlock(
            type="tool_use",
            id="toolu_clock",
            name="get_utc_time",
            input={},
        )
    )

    assert result.is_error is None
    assert isinstance(result.content, str)
    assert result.content.startswith("Current UTC time: ")
    assert result.content.endswith("+00:00")


def test_register_default_tools_exposes_the_cli_tool_set_in_a_stable_order() -> None:
    registry = ToolRegistry()
    register_default_tools(registry)

    assert [definition.name for definition in registry.definitions()] == [
        "calculate",
        "get_weather",
        "get_utc_time",
    ]
