from forge.models import (
    CreateMessageRequest,
    Message,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)

def test_create_message_request_dumps_exercise_a_shape() -> None:
    request = CreateMessageRequest(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get the current weather for a city.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ],
        messages=[
            Message(
                role="user",
                content="What is the weather in London?",
            )
        ],
    )

    assert request.model_dump(mode="json", exclude_none=True) == {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in London?",
            }
        ],
    }

def test_create_message_request_dumps_tool_result_followup_shape() -> None:
    request = CreateMessageRequest(
        model="claude-haiku-4-5",
        max_tokens=1024,
        tools=[
            ToolDefinition(
                name="get_weather",
                description="Get the current weather for a city.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            )
        ],
        messages=[
            Message(
                role="user",
                content="What is the weather in London?",
            ),
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        type="tool_use",
                        id="toolu_123",
                        name="get_weather",
                        input={"city": "London"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        type="tool_result",
                        tool_use_id="toolu_123",
                        content="14°C, raining",
                    )
                ],
            ),
        ],
    )

    assert request.model_dump(mode="json", exclude_none=True) == {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            }
        ],
        "messages": [
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
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "14°C, raining",
                    }
                ],
            },
        ],
    }