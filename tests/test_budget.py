from decimal import Decimal

import pytest

from forge.budget import BudgetedMessageClient, BudgetLedger, ModelPricing
from forge.errors import APIConnectionError, BudgetExceededError
from forge.models import CreateMessageRequest, Message, MessageResponse


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


def test_budget_clamps_output_and_settles_actual_usage() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.requests: list[CreateMessageRequest] = []

        def create_message(self, request: CreateMessageRequest) -> MessageResponse:
            self.requests.append(request)
            return make_response(input_tokens=3, output_tokens=4)

    raw_client = RecordingClient()
    ledger = make_ledger()
    client = BudgetedMessageClient(
        raw_client,
        ledger,
        estimate_input_tokens=lambda request: 4,
    )

    response = client.create_message(make_request())

    assert response.usage.output_tokens == 4
    assert raw_client.requests[0].max_tokens == 15
    assert ledger.confirmed_spend == Decimal("0.34")
    assert ledger.active_reservations == 0
    assert ledger.ambiguous_reservations == Decimal("0")


def test_budget_keeps_reservation_after_ambiguous_completion() -> None:
    class AmbiguousClient:
        def create_message(self, request: CreateMessageRequest) -> MessageResponse:
            raise APIConnectionError("Response lost.", request_may_have_completed=True)

    ledger = make_ledger(cap="0.50")
    client = BudgetedMessageClient(
        AmbiguousClient(),
        ledger,
        estimate_input_tokens=lambda request: 1,
    )

    with pytest.raises(APIConnectionError, match="Response lost"):
        client.create_message(make_request())

    assert ledger.confirmed_spend == Decimal("0")
    assert ledger.ambiguous_reservations == Decimal("0.50")
    with pytest.raises(BudgetExceededError):
        client.create_message(make_request())
