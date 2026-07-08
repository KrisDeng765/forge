from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

JsonObject = dict[str, Any]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class TextBlock(WireModel):
    type: Literal["text"]
    text: str


class ToolUseBlock(WireModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: JsonObject


class ToolResultBlock(WireModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[TextBlock] | None = None
    is_error: bool | None = None


class UnknownContentBlock(BaseModel):
    type: str
    raw: JsonObject

    @model_validator(mode="before")
    @classmethod
    def wrap_wire_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw = cast(dict[str, Any], data)
        block_type = raw.get("type", "unknown")
        if not isinstance(block_type, str):
            block_type = "unknown"

        return {
            "type": block_type,
            "raw": raw,
        }

    @model_serializer(mode="plain")
    def serialize_original_payload(self) -> JsonObject:
        return self.raw


KnownContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]

ContentBlock = KnownContentBlock | UnknownContentBlock


class Message(WireModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = Field(min_length=1)
    input_schema: JsonObject


class AutoToolChoice(BaseModel):
    type: Literal["auto"]


class AnyToolChoice(BaseModel):
    type: Literal["any"]


class NoneToolChoice(BaseModel):
    type: Literal["none"]


class NamedToolChoice(BaseModel):
    type: Literal["tool"]
    name: str


ToolChoice = Annotated[
    AutoToolChoice | AnyToolChoice | NoneToolChoice | NamedToolChoice,
    Field(discriminator="type"),
]


class CreateMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    max_tokens: int = Field(gt=0)
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None


class Usage(WireModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(WireModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[ContentBlock]
    stop_reason: str | None
    usage: Usage