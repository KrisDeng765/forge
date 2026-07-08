from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TextBlock(BaseModel):
    type: Literal["text"]
    text: str

class ToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]

class ToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str

class UnknownContentBlock(BaseModel):
    type: str
    raw: dict[str, Any]

ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | UnknownContentBlock

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int

class MessageResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[ContentBlock]
    stop_reason: str | None
    usage: Usage