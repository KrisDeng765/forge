import json
from pathlib import Path
from typing import Any, cast

from forge.models import MessageResponse, TextBlock, ToolUseBlock, UnknownContentBlock

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def test_parse_exercise_a_tool_use_response() -> None:
    data = load_fixture("exercise_a_tool_use.json")

    response = MessageResponse.model_validate(data)
    parsed = MessageResponse.model_validate(data)

    assert response.stop_reason == "tool_use"
    assert parsed.model_dump(mode="json") == data
    assert any(isinstance(block, ToolUseBlock) for block in response.content)


def test_parse_exercise_b_final_answer_response() -> None:
    data = load_fixture("exercise_b_tool_result_answer.json")

    response = MessageResponse.model_validate(data)
    parsed = MessageResponse.model_validate(data)
    assert parsed.model_dump(mode="json") == data
    assert response.stop_reason == "end_turn"
    assert any(isinstance(block, TextBlock) for block in response.content)
    
def test_parse_exercise_c_tools_attached_but_unused() -> None:
    data = load_fixture("exercise_c_no_tool.json")

    response = MessageResponse.model_validate(data)
    parsed = MessageResponse.model_validate(data)
    assert parsed.model_dump(mode="json") == data
    assert response.stop_reason == "end_turn"
    assert not any(isinstance(block, ToolUseBlock) for block in response.content)

def test_unknown_block_type_is_preserved() -> None:
    data = load_fixture("test_unknown_block_type_is_preserved.json")
    original_block = data["content"][0]

    response = MessageResponse.model_validate(data)

    unknown_blocks = [
        block for block in response.content
        if isinstance(block, UnknownContentBlock)
    ]
    parsed = MessageResponse.model_validate(data)
    assert parsed.model_dump(mode="json") == data
    assert len(unknown_blocks) == 1
    assert unknown_blocks[0].type == "telepathy"
    assert unknown_blocks[0].raw == original_block