from io import StringIO

from forge.cli import TerminalObserver, exit_code_for, run_task
from forge.models import (
    CreateMessageRequest,
    MessageResponse,
    ToolResultBlock,
    ToolUseBlock,
)


class FakeClient:
    def __init__(self, response: MessageResponse) -> None:
        self._response = response
        self.requests: list[CreateMessageRequest] = []

    def create_message(self, request: CreateMessageRequest) -> MessageResponse:
        self.requests.append(request)
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
    assert exit_code_for("refusal") == 2
    assert exit_code_for("truncated") == 3
    assert exit_code_for("context_limit") == 4


def test_terminal_observer_prints_tool_progress() -> None:
    output = StringIO()
    observer = TerminalObserver(output)

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

    exit_code = run_task(
        "Say hello.",
        client,
        stdout=stdout,
        stderr=stderr,
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
