from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

type JsonObject = dict[str, Any]
KNOWN_CONTENT_BLOCK_TYPES: frozenset[str] = frozenset(
    {"text", "tool_use", "tool_result"}
)


class WireModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


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

        if "raw" in raw and isinstance(raw.get("raw"), dict):
            return raw

        block_type = raw.get("type", "unknown")
        if not isinstance(block_type, str):
            block_type = "unknown"

        # Known block types are runtime protocol, not forward-compatibility surface.
        # If one is malformed, fail validation instead of hiding it as unknown.
        if block_type in KNOWN_CONTENT_BLOCK_TYPES:
            raise ValueError(
                "known content block types must validate as their concrete model"
            )

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


class ToolDefinition(RequestModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = Field(min_length=1)
    input_schema: JsonObject


class AutoToolChoice(RequestModel):
    type: Literal["auto"]


class AnyToolChoice(RequestModel):
    type: Literal["any"]


class NoneToolChoice(RequestModel):
    type: Literal["none"]


class NamedToolChoice(RequestModel):
    type: Literal["tool"]
    name: str


ToolChoice = Annotated[
    AutoToolChoice | AnyToolChoice | NoneToolChoice | NamedToolChoice,
    Field(discriminator="type"),
]


class CreateMessageRequest(RequestModel):
    model: str
    max_tokens: int = Field(gt=0)
    messages: list[Message]
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None


class ErrorDetail(WireModel):
    type: str
    message: str


class ErrorResponse(WireModel):
    type: Literal["error"]
    error: ErrorDetail


class Usage(WireModel):
    input_tokens: int
    output_tokens: int


class MessageResponse(WireModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[ContentBlock]
    # Wire parsing stays tolerant; the loop dispatch layer reintroduces strict
    # handling by mapping known stop_reason values and surfacing unknown ones.
    stop_reason: str | None
    stop_sequence: str | None
    usage: Usage
