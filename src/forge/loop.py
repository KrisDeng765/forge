from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Protocol

from forge.errors import ForgeError
from forge.models import (
    CreateMessageRequest,
    JsonObject,
    MessageResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from forge.registry import ToolRegistry
from forge.state import ConversationState

type RunStatus = Literal[
    "completed",
    "stop_sequence",
    "refusal",
    "truncated",
    "context_limit",
]


class MessageClient(Protocol):
    def create_message(self, request: CreateMessageRequest) -> MessageResponse: ...


class ApprovalPolicy(Protocol):
    def approve(self, tool_name: str, tool_input: JsonObject) -> bool: ...


class AlwaysApprove:
    def approve(self, tool_name: str, tool_input: JsonObject) -> bool:
        return True


@dataclass(frozen=True)
class RunResult:
    response: MessageResponse
    status: RunStatus

    @property
    def text(self) -> str:
        return "".join(
            block.text
            for block in self.response.content
            if isinstance(block, TextBlock)
        )


class LoopError(ForgeError):
    """Base exception for Agent Loop protocol failures."""


class LoopProtocolError(LoopError):
    """Raised when a provider response violates the loop protocol."""


class MaxIterationsExceeded(LoopError):
    def __init__(self, max_iterations: int) -> None:
        self.max_iterations = max_iterations
        super().__init__(
            f"Agent loop reached its limit of {max_iterations} API requests."
        )


class UnsupportedStopReasonError(LoopProtocolError):
    def __init__(self, response: MessageResponse) -> None:
        self.response = response
        super().__init__(f"Unsupported stop reason: {response.stop_reason!r}.")


class AgentLoop:
    """Orchestrate one run without owning transport, tools, or transcript storage."""

    def __init__(
        self,
        *,
        client: MessageClient,
        state: ConversationState,
        registry: ToolRegistry,
        model: str,
        max_tokens: int,
        system: str | None = None,
        max_iterations: int = 10,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        self._client = client
        self._state = state
        self._registry = registry
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._max_iterations = max_iterations
        self._approval_policy = (
            approval_policy if approval_policy is not None else AlwaysApprove()
        )

    def run(self) -> RunResult:
        """Run until a terminal response, protocol error, or API-call limit."""

        for _ in range(self._max_iterations):
            response = self._client.create_message(self._build_request())
            result = self._dispatch(response)
            if result is not None:
                return result

        raise MaxIterationsExceeded(self._max_iterations)

    def _build_request(self) -> CreateMessageRequest:
        definitions = self._registry.definitions()
        return CreateMessageRequest(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=definitions or None,
            messages=self._state.snapshot(),
        )

    def _dispatch(self, response: MessageResponse) -> RunResult | None:
        tool_uses = [
            block for block in response.content if isinstance(block, ToolUseBlock)
        ]

        match response.stop_reason:
            case "end_turn":
                self._require_no_tool_uses(response, tool_uses)
                self._state.append_assistant_blocks(response.content)
                return RunResult(response=response, status="completed")
            case "stop_sequence":
                self._require_no_tool_uses(response, tool_uses)
                if response.stop_sequence is None:
                    raise LoopProtocolError(
                        "stop_sequence requires a matching stop_sequence value."
                    )
                self._state.append_assistant_blocks(response.content)
                return RunResult(response=response, status="stop_sequence")
            case "tool_use":
                if not tool_uses:
                    raise LoopProtocolError(
                        "stop_reason 'tool_use' requires at least one tool_use block."
                    )
                self._require_unique_tool_use_ids(tool_uses)
                self._state.append_assistant_blocks(response.content)
                results = [self._execute_tool_use(tool_use) for tool_use in tool_uses]
                self._state.append_tool_results(results)
                return None
            case "pause_turn":
                self._require_no_tool_uses(response, tool_uses)
                self._state.append_assistant_blocks(response.content)
                return None
            case "refusal":
                self._require_no_tool_uses(response, tool_uses)
                self._state.append_assistant_blocks(response.content)
                return RunResult(response=response, status="refusal")
            case "max_tokens":
                # Phase A never retries a truncated response or executes its tool blocks.
                return RunResult(response=response, status="truncated")
            case "model_context_window_exceeded":
                return RunResult(response=response, status="context_limit")
            case None:
                raise LoopProtocolError(
                    "A completed non-streaming response cannot have a null stop reason."
                )
            case _:
                raise UnsupportedStopReasonError(response)

    def _execute_tool_use(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        try:
            approved = self._approval_policy.approve(
                tool_use.name,
                deepcopy(tool_use.input),
            )
        except Exception as exc:
            return _error_result(
                tool_use.id,
                f"Approval for tool {tool_use.name!r} failed: {exc}",
            )

        if not approved:
            return _error_result(
                tool_use.id,
                f"Tool {tool_use.name!r} was not approved.",
            )

        return self._registry.execute(tool_use)

    @staticmethod
    def _require_no_tool_uses(
        response: MessageResponse,
        tool_uses: list[ToolUseBlock],
    ) -> None:
        if tool_uses:
            raise LoopProtocolError(
                f"stop_reason {response.stop_reason!r} cannot include tool_use blocks."
            )

    @staticmethod
    def _require_unique_tool_use_ids(tool_uses: list[ToolUseBlock]) -> None:
        tool_use_ids = [tool_use.id for tool_use in tool_uses]
        if len(set(tool_use_ids)) != len(tool_use_ids):
            raise LoopProtocolError("A response cannot contain duplicate tool_use ids.")


def _error_result(tool_use_id: str, message: str) -> ToolResultBlock:
    return ToolResultBlock(
        type="tool_result",
        tool_use_id=tool_use_id,
        content=message,
        is_error=True,
    )
