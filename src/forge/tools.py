from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import StringConstraints

from forge.registry import ToolInputModel, ToolRegistry


class CalculatorInput(ToolInputModel):
    a: float
    op: Literal["+", "-", "*", "/"]
    b: float


class WeatherInput(ToolInputModel):
    location: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ClockInput(ToolInputModel):
    pass


def register_calculator(registry: ToolRegistry) -> None:
    def calculate(
        a: float,
        op: Literal["+", "-", "*", "/"],
        b: float,
    ) -> str:
        match op:
            case "+":
                value = a + b
            case "-":
                value = a - b
            case "*":
                value = a * b
            case "/":
                if b == 0:
                    raise ValueError("Cannot divide by zero.")
                value = a / b
            case _:
                raise AssertionError(f"Unsupported operator: {op!r}")

        return str(value)

    registry.tool(
        name="calculate",
        description="Calculate two numbers using +, -, *, or /.",
        input_model=CalculatorInput,
    )(calculate)


_FAKE_FORECASTS: dict[str, str] = {
    "london": "Fictional forecast for London: 14°C and raining.",
}


def register_weather(registry: ToolRegistry) -> None:
    def get_weather(location: str) -> str:
        return _FAKE_FORECASTS.get(
            location.casefold(),
            f"No fictional forecast is configured for {location!r}.",
        )

    registry.tool(
        name="get_weather",
        description=(
            "Return a deterministic fictional weather forecast. "
            "It is not real-time weather data."
        ),
        input_model=WeatherInput,
    )(get_weather)


def format_utc_time(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("UTC formatting requires a timezone-aware datetime.")

    return f"Current UTC time: {moment.astimezone(UTC).isoformat(timespec='seconds')}"


def register_clock(registry: ToolRegistry) -> None:
    def get_utc_time() -> str:
        return format_utc_time(datetime.now(UTC))

    registry.tool(
        name="get_utc_time",
        description="Return the current time in UTC.",
        input_model=ClockInput,
    )(get_utc_time)


def register_default_tools(registry: ToolRegistry) -> None:
    register_calculator(registry)
    register_weather(registry)
    register_clock(registry)
