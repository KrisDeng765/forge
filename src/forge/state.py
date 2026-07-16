from collections.abc import Sequence
from copy import deepcopy

from forge.models import ContentBlock, Message, ToolResultBlock


class ConversationState:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def append_user_text(self, text: str) -> None:
        self._messages.append(Message(role="user", content=text))

    def append_assistant_blocks(self, blocks: Sequence[ContentBlock]) -> None:
        self._messages.append(
            Message(
                role="assistant",
                content=deepcopy(list(blocks)),
            )
        )

    def append_tool_results(self, results: Sequence[ToolResultBlock]) -> None:
        content: list[ContentBlock] = [result for result in results]
        self._messages.append(
            Message(
                role="user",
                content=deepcopy(content),
            )
        )

    def append_tool_round(
        self,
        assistant_blocks: Sequence[ContentBlock],
        results: Sequence[ToolResultBlock],
    ) -> None:
        """Commit one completed tool round without exposing a half-round state."""

        result_content: list[ContentBlock] = [result for result in results]
        assistant = Message(role="assistant", content=deepcopy(list(assistant_blocks)))
        results_message = Message(role="user", content=deepcopy(result_content))
        self._messages.extend([assistant, results_message])

    def snapshot(self) -> list[Message]:
        # State owns transcript integrity; callers must not receive mutable aliases.
        return deepcopy(self._messages)
