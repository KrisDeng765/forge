"""Bounded synchronous execution for local tools."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Protocol

from forge.models import ToolResultBlock, ToolUseBlock
from forge.registry import ToolRegistry


class ToolExecutor(Protocol):
    def execute(self, registry: ToolRegistry, tool_use: ToolUseBlock) -> ToolResultBlock: ...


class TimedToolExecutor:
    """Run one synchronous tool in a worker and return a correlated timeout result."""

    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("tool timeout must be positive.")
        self._timeout_seconds = timeout_seconds

    def execute(self, registry: ToolRegistry, tool_use: ToolUseBlock) -> ToolResultBlock:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="forge-tool")
        future = executor.submit(registry.execute, tool_use)
        try:
            result = future.result(timeout=self._timeout_seconds)
        except TimeoutError:
            future.cancel()
            # Python cannot kill a running thread. The worker may finish later, but its
            # output is deliberately abandoned; Phase C's async tools will cancel work.
            executor.shutdown(wait=False, cancel_futures=True)
            return ToolResultBlock(
                type="tool_result",
                tool_use_id=tool_use.id,
                content=(
                    f"Tool {tool_use.name!r} exceeded its "
                    f"{self._timeout_seconds:g}-second timeout."
                ),
                is_error=True,
            )
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=True)
            return result
