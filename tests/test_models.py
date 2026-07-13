import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from forge.models import MessageResponse, TextBlock, ToolUseBlock, UnknownContentBlock

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    path = FIXTURES / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], data)


def test_parse_exercise_a_tool_use_response() -> None:
    data = load_fixture("exercise_a_tool_use.json")

    response = MessageResponse.model_validate(data)

    assert response.stop_reason == "tool_use"
    assert response.model_dump(mode="json") == data
    assert any(isinstance(block, ToolUseBlock) for block in response.content)


def test_parse_exercise_b_final_answer_response() -> None:
    data = load_fixture("exercise_b_tool_result_answer.json")

    response = MessageResponse.model_validate(data)
    assert response.model_dump(mode="json") == data
    assert response.stop_reason == "end_turn"
    assert any(isinstance(block, TextBlock) for block in response.content)
    
def test_parse_exercise_c_tools_attached_but_unused() -> None:
    data = load_fixture("exercise_c_no_tool.json")

    response = MessageResponse.model_validate(data)
    assert response.model_dump(mode="json") == data
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
    assert response.model_dump(mode="json") == data
    assert len(unknown_blocks) == 1
    assert unknown_blocks[0].type == "telepathy"
    assert unknown_blocks[0].raw == original_block

def test_malformed_known_block_is_rejected() -> None:
    data = deepcopy(load_fixture("exercise_incomplete_response.json"))
    data["content"][0] = {"type": "text"}

    with pytest.raises(ValidationError):
        MessageResponse.model_validate(data)

def test_stop_sequence_is_exposed_as_a_typed_field() -> None:
    data = load_fixture("exercise_b_tool_result_answer.json")
    data["stop_sequence"] = "\n\nEND"

    response = MessageResponse.model_validate(data)

    assert response.stop_sequence == "\n\nEND"


def test_stop_details_is_exposed_as_a_typed_field() -> None:
    data = load_fixture("exercise_b_tool_result_answer.json")
    data["stop_details"] = {"reason": "policy"}

    response = MessageResponse.model_validate(data)

    assert response.stop_details == {"reason": "policy"}
