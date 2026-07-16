import asyncio
import json
from io import StringIO
from pathlib import Path

from forge.cli import TerminalObserver, exit_code_for, parse_options, run_task
from forge.models import (
    CreateMessageRequest,
    MessageResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from forge.streaming import StreamObserver


class FakeClient:
    def __init__(self, response: MessageResponse) -> None:
        self._response = response
        self.requests: list[CreateMessageRequest] = []

    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse:
        self.requests.append(request)
        if observer is not None:
            for block in self._response.content:
                if isinstance(block, TextBlock):
                    observer.on_text_delta(block.text)
        return self._response


def make_final_response(text: str) -> MessageResponse:
    return MessageResponse.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "stop_details": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def test_exit_codes_reflect_run_status() -> None:
    assert exit_code_for("completed") == 0
    assert exit_code_for("stop_sequence") == 0
    assert exit_code_for("refusal") == 5
    assert exit_code_for("truncated") == 3
    assert exit_code_for("context_limit") == 4
    assert exit_code_for("budget_exceeded") == 6
    assert exit_code_for("ambiguous_completion") == 8


def test_cli_options_allow_a_positive_budget_override() -> None:
    task, budget, state_file = parse_options(["--budget-usd", "0.10", "Say hello."])

    assert task == "Say hello."
    assert budget.as_tuple().exponent == -2
    assert str(budget) == "0.10"
    assert state_file is None

    _, _, configured_state_file = parse_options(
        ["--state-file", "run-state.json", "Say hello."]
    )
    assert configured_state_file == Path("run-state.json")


def test_terminal_observer_prints_tool_progress() -> None:
    output = StringIO()
    text_output = StringIO()
    observer = TerminalObserver(text_output, output)

    observer.on_tool_call(
        ToolUseBlock(
            type="tool_use",
            id="toolu_weather",
            name="get_weather",
            input={"location": "London"},
        )
    )
    observer.on_tool_result(
        ToolResultBlock(
            type="tool_result",
            tool_use_id="toolu_weather",
            content="London is rainy.",
        )
    )
    observer.on_tool_call(
        ToolUseBlock(
            type="tool_use",
            id="toolu_calculator",
            name="calculate",
            input={"a": 1, "op": "/", "b": 0},
        )
    )
    observer.on_tool_result(
        ToolResultBlock(
            type="tool_result",
            tool_use_id="toolu_calculator",
            content="division by zero",
            is_error=True,
        )
    )

    assert output.getvalue().splitlines() == [
        "→ Calling get_weather",
        "← get_weather completed",
        "→ Calling calculate",
        "← calculate failed",
    ]


def test_run_task_assembles_default_tools_and_prints_the_final_answer() -> None:
    client = FakeClient(make_final_response("Hello from Forge."))
    stdout = StringIO()
    stderr = StringIO()

    exit_code = asyncio.run(
        run_task(
            "Say hello.",
            client,
            stdout=stdout,
            stderr=stderr,
        )
    )

    assert exit_code == 0
    assert stdout.getvalue() == "Hello from Forge.\n"
    assert stderr.getvalue() == ""
    assert [tool.name for tool in client.requests[0].tools or []] == [
        "calculate",
        "get_weather",
        "get_utc_time",
    ]
    assert client.requests[0].messages[0].content == "Say hello."


def test_cancelled_run_writes_a_parseable_non_secret_snapshot(tmp_path: Path) -> None:
    class BlockingClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            await asyncio.Event().wait()
            raise AssertionError("The blocking request should be cancelled.")

    state_file = tmp_path / "forge-state.json"
    stdout = StringIO()
    stderr = StringIO()

    async def cancel_run() -> int:
        task = asyncio.create_task(
            run_task(
                "Say hello.",
                BlockingClient(),
                stdout=stdout,
                stderr=stderr,
                state_file=state_file,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        return await task

    assert asyncio.run(cancel_run()) == 130
    snapshot = json.loads(state_file.read_text(encoding="utf-8"))
    assert snapshot["status"] == "interrupted"
    assert snapshot["snapshot_version"] == 1
    assert snapshot["messages"][0]["content"] == "Say hello."
    assert "api_key" not in json.dumps(snapshot).lower()
    assert "state saved" in stderr.getvalue()
