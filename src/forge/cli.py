import asyncio
import json
import os
import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

from forge.budget import BudgetedMessageClient, BudgetLedger, ModelPricing
from forge.client import AnthropicClient
from forge.errors import ForgeError
from forge.loop import AgentLoop, MessageClient, RunStatus
from forge.models import ToolResultBlock, ToolUseBlock
from forge.registry import ToolRegistry
from forge.retry import RetryingMessageClient
from forge.state import ConversationState
from forge.streaming import StreamObserver
from forge.tools import register_default_tools

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_HARD_CAP_USD = Decimal("0.05")
# Claude Haiku 4.5 standard input/output pricing, expressed per token.
DEFAULT_MODEL_PRICING = ModelPricing(
    input_token_price=Decimal("0.000001"),
    output_token_price=Decimal("0.000005"),
)
DEFAULT_SYSTEM = (
    "Use the available tools for arithmetic, UTC time, and weather requests. "
    "Weather tool results are fictional and must be labeled as such."
)


class TerminalObserver(StreamObserver):
    def __init__(self, text_output: TextIO, status_output: TextIO) -> None:
        self._text_output = text_output
        self._status_output = status_output
        self._tool_names: dict[str, str] = {}

    def on_text_delta(self, text: str) -> None:
        print(text, end="", flush=True, file=self._text_output)

    def on_input_tokens(self, input_tokens: int) -> None:
        pass

    def on_stream_retry(self) -> None:
        print("\n[Stream interrupted; retrying the request.]", file=self._status_output)

    def on_tool_call(self, tool_use: ToolUseBlock) -> None:
        self._tool_names[tool_use.id] = tool_use.name
        print(f"→ Calling {tool_use.name}", file=self._status_output)

    def on_tool_result(self, result: ToolResultBlock) -> None:
        tool_name = self._tool_names.pop(result.tool_use_id, result.tool_use_id)
        outcome = "failed" if result.is_error else "completed"
        print(f"← {tool_name} {outcome}", file=self._status_output)


def exit_code_for(status: RunStatus) -> int:
    match status:
        case "completed" | "stop_sequence":
            return 0
        case "refusal":
            return 5
        case "truncated":
            return 3
        case "context_limit":
            return 4
        case "budget_exceeded":
            return 6
        case "tool_validation_stalled":
            return 7
        case "ambiguous_completion":
            return 8
        case _:
            raise AssertionError(f"Unexpected run status: {status!r}")


async def run_task(
    task: str,
    client: MessageClient,
    *,
    stdout: TextIO,
    stderr: TextIO,
    hard_cap: Decimal = DEFAULT_HARD_CAP_USD,
    state_file: Path | None = None,
) -> int:
    state = ConversationState()
    state.append_user_text(task)
    registry = ToolRegistry()
    register_default_tools(registry)
    ledger = BudgetLedger(
        hard_cap=hard_cap,
        pricing=DEFAULT_MODEL_PRICING,
    )
    budgeted_client = BudgetedMessageClient(client, ledger)
    resilient_client = RetryingMessageClient(budgeted_client)
    observer = TerminalObserver(stdout, stderr)
    loop = AgentLoop(
        client=resilient_client,
        state=state,
        registry=registry,
        model=DEFAULT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=DEFAULT_SYSTEM,
        observer=observer,
        stream_observer=observer,
    )

    try:
        result = await loop.run()
    except asyncio.CancelledError:
        if state_file is not None:
            _write_interrupt_snapshot(state_file, loop.interruption_snapshot())
            print(f"Interrupted; state saved to {state_file}.", file=stderr)
        else:
            print("Interrupted; no state file was requested.", file=stderr)
        return 130

    if result.text:
        print(file=stdout)
    if result.status not in {"completed", "stop_sequence"}:
        print(f"Run ended with status: {result.status}", file=stderr)

    return exit_code_for(result.status)


def parse_options(argv: Sequence[str] | None) -> tuple[str, Decimal, Path | None]:
    parser = ArgumentParser(description="Run Forge on one task.")
    parser.add_argument("task", help="The task to send to the agent.")
    parser.add_argument(
        "--budget-usd",
        type=_positive_decimal,
        default=DEFAULT_HARD_CAP_USD,
        help="Hard per-run Messages API spend cap in USD (default: 0.05).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Write an atomic JSON state snapshot here if the run is interrupted.",
    )
    parsed = parser.parse_args(argv)
    task = getattr(parsed, "task", None)
    if not isinstance(task, str):
        raise AssertionError("argparse did not produce a string task.")
    if not task.strip():
        parser.error("task must not be blank")

    budget = getattr(parsed, "budget_usd", None)
    if not isinstance(budget, Decimal):
        raise AssertionError("argparse did not produce a Decimal budget.")
    state_file = getattr(parsed, "state_file", None)
    if state_file is not None and not isinstance(state_file, Path):
        raise AssertionError("argparse did not produce a Path state file.")
    return task, budget, state_file


def _positive_decimal(value: str) -> Decimal:
    try:
        budget = Decimal(value)
    except ArithmeticError as exc:
        raise ValueError("budget must be a decimal value") from exc
    if not budget.is_finite() or budget <= 0:
        raise ValueError("budget must be a positive finite decimal")
    return budget


async def _main_async(task: str, budget: Decimal, state_file: Path | None) -> int:
    async with AnthropicClient() as client:
        return await run_task(
            task,
            client,
            stdout=sys.stdout,
            stderr=sys.stderr,
            hard_cap=budget,
            state_file=state_file,
        )


def main(argv: Sequence[str] | None = None) -> int:
    task, budget, state_file = parse_options(argv)

    try:
        return asyncio.run(_main_async(task, budget, state_file))
    except (ForgeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted before Forge could save state.", file=sys.stderr)
        return 130


def _write_interrupt_snapshot(path: Path, snapshot: dict[str, object]) -> None:
    """Atomically persist parseable, non-secret interruption state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "interrupted_at": datetime.now(UTC).isoformat(),
        "status": "interrupted",
        **snapshot,
    }
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(payload, temporary, ensure_ascii=False, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


if __name__ == "__main__":
    raise SystemExit(main())
