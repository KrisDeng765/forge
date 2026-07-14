import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from decimal import Decimal
from typing import TextIO

from forge.budget import BudgetedMessageClient, BudgetLedger, ModelPricing
from forge.client import AnthropicClient
from forge.errors import ForgeError
from forge.loop import AgentLoop, MessageClient, RunStatus
from forge.models import ToolResultBlock, ToolUseBlock
from forge.registry import ToolRegistry
from forge.retry import RetryingMessageClient
from forge.state import ConversationState
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
            return 5
        case "truncated":
            return 3
        case "context_limit":
            return 4
        case "budget_exceeded":
            return 6
        case "tool_validation_stalled":
            return 7
        case _:
            raise AssertionError(f"Unexpected run status: {status!r}")


def run_task(
    task: str,
    client: MessageClient,
    *,
    stdout: TextIO,
    stderr: TextIO,
    hard_cap: Decimal = DEFAULT_HARD_CAP_USD,
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
    loop = AgentLoop(
        client=resilient_client,
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


def parse_options(argv: Sequence[str] | None) -> tuple[str, Decimal]:
    parser = ArgumentParser(description="Run Forge on one task.")
    parser.add_argument("task", help="The task to send to the agent.")
    parser.add_argument(
        "--budget-usd",
        type=_positive_decimal,
        default=DEFAULT_HARD_CAP_USD,
        help="Hard per-run Messages API spend cap in USD (default: 0.05).",
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
    return task, budget


def _positive_decimal(value: str) -> Decimal:
    try:
        budget = Decimal(value)
    except ArithmeticError as exc:
        raise ValueError("budget must be a decimal value") from exc
    if not budget.is_finite() or budget <= 0:
        raise ValueError("budget must be a positive finite decimal")
    return budget


def main(argv: Sequence[str] | None = None) -> int:
    task, budget = parse_options(argv)

    try:
        with AnthropicClient() as client:
            return run_task(
                task,
                client,
                stdout=sys.stdout,
                stderr=sys.stderr,
                hard_cap=budget,
            )
    except (ForgeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
