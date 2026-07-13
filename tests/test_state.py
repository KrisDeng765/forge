from forge.models import ToolResultBlock, ToolUseBlock
from forge.state import ConversationState


def make_tool_use() -> ToolUseBlock:
    return ToolUseBlock.model_validate(
        {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"city": "London"},
            "caller": {"type": "direct"},
        }
    )


def test_state_builds_a_verbatim_replayable_transcript() -> None:
    state = ConversationState()

    state.append_user_text("What is the weather in London?")
    state.append_assistant_blocks([make_tool_use()])
    state.append_tool_results(
        [
            ToolResultBlock(
                type="tool_result",
                tool_use_id="toolu_123",
                content="London: 14 C and raining.",
            )
        ]
    )

    assert [
        message.model_dump(mode="json", exclude_none=True)
        for message in state.snapshot()
    ] == [
        {
            "role": "user",
            "content": "What is the weather in London?",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "London"},
                    "caller": {"type": "direct"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "London: 14 C and raining.",
                }
            ],
        },
    ]


def test_snapshot_is_a_deep_copy_of_the_internal_transcript() -> None:
    state = ConversationState()
    state.append_user_text("What is the weather in London?")
    state.append_assistant_blocks([make_tool_use()])

    snapshot = state.snapshot()
    assert isinstance(snapshot[1].content, list)
    block = snapshot[1].content[0]
    assert isinstance(block, ToolUseBlock)

    block.input["city"] = "Paris"
    snapshot.pop()

    fresh_snapshot = state.snapshot()
    assert len(fresh_snapshot) == 2
    assert fresh_snapshot[1].model_dump(
        mode="json",
        exclude_none=True,
    )["content"][0]["input"] == {"city": "London"}