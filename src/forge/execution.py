"""Bounded asynchronous execution for local tools."""

import asyncio
from typing import Protocol

from forge.models import ToolResultBlock, ToolUseBlock
from forge.registry import ToolRegistry


class ToolExecutor(Protocol):
    async def execute(
        self,
        registry: ToolRegistry,
        tool_use: ToolUseBlock,
    ) -> ToolResultBlock: ...


class TimedToolExecutor:
    """Run one tool with an async timeout and a correlated timeout result."""

    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive.")
        self._timeout_seconds = timeout_seconds

    async def execute(
        self,
        registry: ToolRegistry,
        tool_use: ToolUseBlock,
    ) -> ToolResultBlock:
        try:
            return await asyncio.wait_for(
                registry.execute(tool_use),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            # async tools receive cancellation at their next await point. A sync tool
            # called through asyncio.to_thread may still finish in its abandoned thread.
            return ToolResultBlock(
                type="tool_result",
                tool_use_id=tool_use.id,
                content=(
                    f"Tool {tool_use.name!r} exceeded its "
                    f"{self._timeout_seconds:g}-second timeout."
                ),
                is_error=True,
            )
