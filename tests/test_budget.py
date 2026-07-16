import asyncio
import json
from decimal import Decimal

import pytest

from forge.budget import (
    TOOL_USE_PREAMBLE_TOKEN_ALLOWANCE,
    BudgetedMessageClient,
    BudgetLedger,
    ModelPricing,
    conservative_input_token_estimate,
)
from forge.errors import APIConnectionError, BudgetAccountingError, BudgetExceededError
from forge.models import CreateMessageRequest, Message, MessageResponse, ToolDefinition
from forge.streaming import StreamObserver


def make_request(max_tokens: int = 40) -> CreateMessageRequest:
    return CreateMessageRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[Message(role="user", content="Hello")],
    )


def make_response(input_tokens: int, output_tokens: int) -> MessageResponse:
    return MessageResponse.model_validate(
        {
            "id": "msg_budget",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "Done."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def make_ledger(cap: str = "0.55") -> BudgetLedger:
    return BudgetLedger(
        hard_cap=Decimal(cap),
        pricing=ModelPricing(
            input_token_price=Decimal("0.10"),
            output_token_price=Decimal("0.01"),
        ),
    )


def test_tool_requests_reserve_the_provider_preamble_allowance() -> None:
    without_tools = make_request()
    with_tools = without_tools.model_copy(
        update={
            "tools": [
                ToolDefinition(
                    name="x",
                    description="x",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                )
            ]
        },
        deep=True,
    )

    def compact_json_bytes(request: CreateMessageRequest) -> int:
        payload = request.model_dump(mode="json", exclude_none=True)
        return len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode())

    assert conservative_input_token_estimate(without_tools) == compact_json_bytes(
        without_tools
    )
    assert conservative_input_token_estimate(with_tools) == (
        compact_json_bytes(with_tools) + TOOL_USE_PREAMBLE_TOKEN_ALLOWANCE
    )


def test_budget_clamps_output_and_settles_actual_usage() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.requests: list[CreateMessageRequest] = []

        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            self.requests.append(request)
            return make_response(input_tokens=3, output_tokens=4)

    raw_client = RecordingClient()
    ledger = make_ledger()
    client = BudgetedMessageClient(
        raw_client,
        ledger,
        estimate_input_tokens=lambda request: 4,
    )

    response = asyncio.run(client.create_message(make_request()))

    assert response.usage.output_tokens == 4
    assert raw_client.requests[0].max_tokens == 15
    assert ledger.confirmed_spend == Decimal("0.34")
    assert ledger.active_reservations == 0
    assert ledger.ambiguous_reservations == Decimal("0")


def test_budget_keeps_reservation_after_ambiguous_completion() -> None:
    class AmbiguousClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            raise APIConnectionError("Response lost.", request_may_have_completed=True)

    ledger = make_ledger(cap="0.50")
    client = BudgetedMessageClient(
        AmbiguousClient(),
        ledger,
        estimate_input_tokens=lambda request: 1,
    )

    with pytest.raises(APIConnectionError, match="Response lost"):
        asyncio.run(client.create_message(make_request()))

    assert ledger.confirmed_spend == Decimal("0")
    assert ledger.ambiguous_reservations == Decimal("0.50")
    with pytest.raises(BudgetExceededError):
        asyncio.run(client.create_message(make_request()))


def test_message_start_reconciles_input_usage_before_output_is_generated() -> None:
    class UnderestimatedClient:
        async def create_message(
            self,
            request: CreateMessageRequest,
            *,
            observer: StreamObserver | None = None,
        ) -> MessageResponse:
            assert observer is not None
            observer.on_input_tokens(5)
            return make_response(input_tokens=5, output_tokens=1)

    ledger = make_ledger(cap="0.55")
    client = BudgetedMessageClient(
        UnderestimatedClient(),
        ledger,
        estimate_input_tokens=lambda request: 4,
    )

    with pytest.raises(BudgetAccountingError, match="input usage"):
        asyncio.run(client.create_message(make_request()))

    assert ledger.active_reservations == 0
    assert ledger.ambiguous_reservations == Decimal("0.55")
