import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import TextIO

from forge.client import AnthropicClient
from forge.errors import ForgeError
from forge.loop import AgentLoop, MessageClient, RunStatus
from forge.models import ToolResultBlock, ToolUseBlock
from forge.registry import ToolRegistry
from forge.state import ConversationState
from forge.tools import register_default_tools

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_SYSTEM = (
    "Use the available tools for arithmetic, UTC time, and weather requests. "
    "Weather tool results are fictional and must be labeled as such."
)


class TerminalObserver:
    def __init__(self, output: TextIO) -> None:
        self._output = output
        self._tool_names: dict[str, str] = {}

    def on_tool_call(self, tool_use: ToolUseBlock) -> None:
        self._tool_names[tool_use.id] = tool_use.name
        print(f"→ Calling {tool_use.name}", file=self._output)

    def on_tool_result(self, result: ToolResultBlock) -> None:
        tool_name = self._tool_names.pop(result.tool_use_id, result.tool_use_id)
        outcome = "failed" if result.is_error else "completed"
        print(f"← {tool_name} {outcome}", file=self._output)


def exit_code_for(status: RunStatus) -> int:
    match status:
        case "completed" | "stop_sequence":
            return 0
        case "refusal":
            return 2
        case "truncated":
            return 3
        case "context_limit":
            return 4
        case _:
            raise AssertionError(f"Unexpected run status: {status!r}")


def run_task(
    task: str,
    client: MessageClient,
    *,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    state = ConversationState()
    state.append_user_text(task)
    registry = ToolRegistry()
    register_default_tools(registry)
    loop = AgentLoop(
        client=client,
        state=state,
        registry=registry,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=DEFAULT_SYSTEM,
        observer=TerminalObserver(stderr),
    )

    result = loop.run()
    if result.text:
        print(result.text, file=stdout)
    if result.status not in {"completed", "stop_sequence"}:
        print(f"Run ended with status: {result.status}", file=stderr)

    return exit_code_for(result.status)


def _parse_task(argv: Sequence[str] | None) -> str:
    parser = ArgumentParser(description="Run Forge on one task.")
    parser.add_argument("task", help="The task to send to the agent.")
    parsed = parser.parse_args(argv)
    task = getattr(parsed, "task", None)
    if not isinstance(task, str):
        raise AssertionError("argparse did not produce a string task.")
    if not task.strip():
        parser.error("task must not be blank")

    return task


def main(argv: Sequence[str] | None = None) -> int:
    task = _parse_task(argv)

    try:
        with AnthropicClient() as client:
            return run_task(
                task,
                client,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
    except (ForgeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
