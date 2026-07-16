import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Protocol

from forge.errors import (
    AmbiguousCompletionBudgetError,
    BudgetExceededError,
    ForgeError,
)
from forge.execution import TimedToolExecutor, ToolExecutor
from forge.models import (
    CreateMessageRequest,
    JsonObject,
    MessageResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from forge.registry import ToolRegistry, is_validation_error
from forge.state import ConversationState
from forge.streaming import StreamObserver

type RunStatus = Literal[
    "completed",
    "stop_sequence",
    "refusal",
    "truncated",
    "context_limit",
    "budget_exceeded",
    "ambiguous_completion",
    "tool_validation_stalled",
]


class MessageClient(Protocol):
    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse: ...


class ApprovalPolicy(Protocol):
    def approve(self, tool_name: str, tool_input: JsonObject) -> bool: ...


class ToolObserver(Protocol):
    def on_tool_call(self, tool_use: ToolUseBlock) -> None: ...

    def on_tool_result(self, result: ToolResultBlock) -> None: ...


class NullToolObserver:
    def on_tool_call(self, tool_use: ToolUseBlock) -> None:
        pass

    def on_tool_result(self, result: ToolResultBlock) -> None:
        pass


class AlwaysApprove:
    def approve(self, tool_name: str, tool_input: JsonObject) -> bool:
        return True


@dataclass(frozen=True)
class RunResult:
    response: MessageResponse | None
    status: RunStatus

    @property
    def text(self) -> str:
        content = self.response.content if self.response is not None else []
        return "".join(
            block.text for block in content if isinstance(block, TextBlock)
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
        observer: ToolObserver | None = None,
        tool_timeout_seconds: float = 10.0,
        max_consecutive_validation_errors: int = 2,
        max_validation_errors: int = 3,
        tool_executor: ToolExecutor | None = None,
        stream_observer: StreamObserver | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if max_consecutive_validation_errors <= 0:
            raise ValueError("max_consecutive_validation_errors must be positive.")
        if max_validation_errors <= 0:
            raise ValueError("max_validation_errors must be positive.")

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
        self._observer = observer if observer is not None else NullToolObserver()
        self._tool_executor = (
            tool_executor
            if tool_executor is not None
            else TimedToolExecutor(tool_timeout_seconds)
        )
        self._stream_observer = stream_observer
        self._max_consecutive_validation_errors = max_consecutive_validation_errors
        self._max_validation_errors = max_validation_errors
        self._validation_error_count = 0
        self._last_validation_error: tuple[str, str] | None = None
        self._consecutive_validation_errors = 0
        self._inflight_tool_uses: list[ToolUseBlock] = []

    async def run(self) -> RunResult:
        """Run until a terminal response, protocol error, or API-call limit."""

        for _ in range(self._max_iterations):
            try:
                response = await self._client.create_message(
                    self._build_request(),
                    observer=self._stream_observer,
                )
            except AmbiguousCompletionBudgetError:
                return RunResult(response=None, status="ambiguous_completion")
            except BudgetExceededError:
                return RunResult(response=None, status="budget_exceeded")
            result = await self._dispatch(response)
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

    async def _dispatch(self, response: MessageResponse) -> RunResult | None:
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
                results = await self._execute_tool_uses(tool_uses)
                # Commit only after every result exists: Ctrl-C snapshots therefore
                # retain a complete transcript, never an orphaned tool_use turn.
                self._state.append_tool_round(response.content, results)
                if self._validation_limit_reached(tool_uses, results):
                    return RunResult(
                        response=response,
                        status="tool_validation_stalled",
                    )
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

    async def _execute_tool_uses(
        self,
        tool_uses: list[ToolUseBlock],
    ) -> list[ToolResultBlock]:
        self._inflight_tool_uses = [tool_use.model_copy(deep=True) for tool_use in tool_uses]
        try:
            results = await asyncio.gather(
                *(self._execute_tool_use(tool_use) for tool_use in tool_uses)
            )
        except asyncio.CancelledError:
            # Preserve these ids in the interrupt snapshot for manual reconciliation;
            # completed local calls may already have had side effects.
            raise
        else:
            self._inflight_tool_uses = []
            return results

    async def _execute_tool_use(self, tool_use: ToolUseBlock) -> ToolResultBlock:
        self._observer.on_tool_call(tool_use.model_copy(deep=True))

        try:
            approved = self._approval_policy.approve(
                tool_use.name,
                deepcopy(tool_use.input),
            )
        except Exception:
            # Fail closed: an approver error must never permit a side-effecting tool.
            # Returning a generic error lets the model choose another safe path without
            # disclosing policy implementation details in a model-visible tool result.
            result = _error_result(
                tool_use.id,
                f"Approval for tool {tool_use.name!r} failed; the tool was not run.",
            )
        else:
            if not approved:
                result = _error_result(
                    tool_use.id,
                    f"Tool {tool_use.name!r} was not approved.",
                )
            else:
                result = await self._tool_executor.execute(self._registry, tool_use)

        self._observer.on_tool_result(result.model_copy(deep=True))
        return result

    def interruption_snapshot(self) -> dict[str, object]:
        """Return only replayable state plus unresolved work; never include secrets."""

        return {
            "snapshot_version": 1,
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": self._system,
            "messages": [
                message.model_dump(mode="json") for message in self._state.snapshot()
            ],
            "unresolved_tool_uses": [
                tool_use.model_dump(mode="json")
                for tool_use in self._inflight_tool_uses
            ],
        }

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

    def _validation_limit_reached(
        self,
        tool_uses: list[ToolUseBlock],
        results: list[ToolResultBlock],
    ) -> bool:
        validation_errors = [
            (tool_use.name, result.content)
            for tool_use, result in zip(tool_uses, results, strict=True)
            if is_validation_error(result) and isinstance(result.content, str)
        ]
        if not validation_errors:
            self._last_validation_error = None
            self._consecutive_validation_errors = 0
            return False

        self._validation_error_count += len(validation_errors)
        if len(validation_errors) == 1:
            signature = validation_errors[0]
            if signature == self._last_validation_error:
                self._consecutive_validation_errors += 1
            else:
                self._last_validation_error = signature
                self._consecutive_validation_errors = 1
        else:
            self._last_validation_error = None
            self._consecutive_validation_errors = 0

        return (
            self._consecutive_validation_errors >= self._max_consecutive_validation_errors
            or self._validation_error_count >= self._max_validation_errors
        )


def _error_result(tool_use_id: str, message: str) -> ToolResultBlock:
    return ToolResultBlock(
        type="tool_result",
        tool_use_id=tool_use_id,
        content=message,
        is_error=True,
    )
