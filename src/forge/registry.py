from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec

from pydantic import BaseModel, ConfigDict, ValidationError

from forge.models import ToolDefinition, ToolResultBlock, ToolUseBlock

P = ParamSpec("P")
ToolFunction = Callable[..., str]


class ToolInputModel(BaseModel):
    """Base input model for a locally registered tool."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class _RegisteredTool:
    definition: ToolDefinition
    input_model: type[ToolInputModel]
    function: ToolFunction


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _RegisteredTool] = {}

    def tool(
        self,
        *,
        description: str,
        input_model: type[ToolInputModel],
        name: str | None = None,
    ) -> Callable[[Callable[P, str]], Callable[P, str]]:
        """Register a typed function and return it unchanged."""

        def register(function: Callable[P, str]) -> Callable[P, str]:
            tool_name = name if name is not None else function.__name__
            definition = ToolDefinition(
                name=tool_name,
                description=description,
                input_schema=input_model.model_json_schema(),
            )

            if tool_name in self._tools:
                raise ValueError(f"A tool named {tool_name!r} is already registered.")

            self._tools[tool_name] = _RegisteredTool(
                definition=definition,
                input_model=input_model,
                function=function,
            )
            return function

        return register

    def definitions(self) -> list[ToolDefinition]:
        """Return independent definitions for request assembly."""

        return [tool.definition.model_copy(deep=True) for tool in self._tools.values()]

    def execute(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        """Validate and execute one tool call without leaking tool failures."""

        tool = self._tools.get(tool_use.name)
        if tool is None:
            return _error_result(
                tool_use.id,
                f"Unknown tool: {tool_use.name}",
            )

        try:
            validated_input = tool.input_model.model_validate(tool_use.input)
        except ValidationError as exc:
            return _error_result(
                tool_use.id,
                f"Invalid input for tool {tool_use.name!r}: {exc}",
            )

        try:
            output = tool.function(**validated_input.model_dump(mode="python"))
        except Exception as exc:
            return _error_result(
                tool_use.id,
                f"Tool {tool_use.name!r} failed: {exc}",
            )

        return ToolResultBlock(
            type="tool_result",
            tool_use_id=tool_use.id,
            content=output,
        )


def _error_result(tool_use_id: str, message: str) -> ToolResultBlock:
    return ToolResultBlock(
        type="tool_result",
        tool_use_id=tool_use_id,
        content=message,
        is_error=True,
    )
