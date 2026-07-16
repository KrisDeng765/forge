"""Conservative, per-attempt spend reservations for Messages API calls."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Protocol

from forge.errors import APIConnectionError, BudgetAccountingError, BudgetExceededError
from forge.models import CreateMessageRequest, MessageResponse, Usage
from forge.streaming import NullStreamObserver, StreamObserver

type InputTokenEstimator = Callable[[CreateMessageRequest], int]

# The API injects a tool-use system preamble that is billable but absent from the
# request JSON. This guard band is deliberately model-specific; see design.md for
# the recorded evidence and its recalibration trigger.
TOOL_USE_PREAMBLE_TOKEN_ALLOWANCE = 512


class MessageClient(Protocol):
    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse: ...


@dataclass(frozen=True)
class ModelPricing:
    """Prices in USD for one input or output token."""

    input_token_price: Decimal
    output_token_price: Decimal

    def __post_init__(self) -> None:
        if self.input_token_price <= 0 or self.output_token_price <= 0:
            raise ValueError("Token prices must be positive.")


@dataclass(frozen=True)
class BudgetReservation:
    identifier: int
    input_tokens: int
    max_tokens: int
    reserved_cost: Decimal


class BudgetLedger:
    """Track confirmed spend and worst-case reservations for one agent run."""

    def __init__(self, *, hard_cap: Decimal, pricing: ModelPricing) -> None:
        if hard_cap <= 0:
            raise ValueError("hard_cap must be positive.")

        self._hard_cap = hard_cap
        self._pricing = pricing
        self._confirmed_spend = Decimal("0")
        self._ambiguous_reservations = Decimal("0")
        self._active: dict[int, BudgetReservation] = {}
        self._next_identifier = 0

    @property
    def confirmed_spend(self) -> Decimal:
        return self._confirmed_spend

    @property
    def ambiguous_reservations(self) -> Decimal:
        return self._ambiguous_reservations

    @property
    def active_reservations(self) -> int:
        return len(self._active)

    def reserve(self, *, input_tokens: int, requested_max_tokens: int) -> BudgetReservation:
        """Reserve an input estimate plus the maximum permitted output cost."""

        if input_tokens < 0:
            raise ValueError("input_tokens cannot be negative.")
        if requested_max_tokens <= 0:
            raise ValueError("requested_max_tokens must be positive.")

        available = self._available_budget()
        input_cost = Decimal(input_tokens) * self._pricing.input_token_price
        output_allowance = available - input_cost
        if output_allowance <= 0:
            raise BudgetExceededError("The remaining budget cannot fund a positive output.")

        max_affordable = int(
            (output_allowance / self._pricing.output_token_price).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        allowed_max_tokens = min(requested_max_tokens, max_affordable)
        if allowed_max_tokens <= 0:
            raise BudgetExceededError("The remaining budget cannot fund a positive output.")

        reserved_cost = input_cost + (
            Decimal(allowed_max_tokens) * self._pricing.output_token_price
        )
        self._next_identifier += 1
        reservation = BudgetReservation(
            identifier=self._next_identifier,
            input_tokens=input_tokens,
            max_tokens=allowed_max_tokens,
            reserved_cost=reserved_cost,
        )
        self._active[reservation.identifier] = reservation
        return reservation

    def settle(self, reservation: BudgetReservation, usage: Usage) -> None:
        """Confirm actual provider usage and release the unused reservation."""

        self._remove_active(reservation)
        actual_cost = (
            Decimal(usage.input_tokens) * self._pricing.input_token_price
            + Decimal(usage.output_tokens) * self._pricing.output_token_price
        )
        if actual_cost > reservation.reserved_cost:
            raise BudgetAccountingError(
                "Provider usage exceeded the request's conservative budget reservation."
            )
        self._confirmed_spend += actual_cost

    def reconcile_input(
        self,
        reservation: BudgetReservation,
        actual_input_tokens: int,
    ) -> None:
        """Fail early when message_start disproves the request estimate."""

        if actual_input_tokens < 0:
            raise BudgetAccountingError("Provider reported negative input token usage.")
        if actual_input_tokens > reservation.input_tokens:
            raise BudgetAccountingError(
                "Provider input usage exceeded the request's conservative budget estimate."
            )

    def release(self, reservation: BudgetReservation) -> None:
        """Release a reservation for a response known not to have completed."""

        self._remove_active(reservation)

    def retain_ambiguous(self, reservation: BudgetReservation) -> None:
        """Keep the entire reservation when completion and billing are unknowable."""

        self._remove_active(reservation)
        self._ambiguous_reservations += reservation.reserved_cost

    def _available_budget(self) -> Decimal:
        active_cost = sum(
            (reservation.reserved_cost for reservation in self._active.values()),
            start=Decimal("0"),
        )
        return (
            self._hard_cap
            - self._confirmed_spend
            - self._ambiguous_reservations
            - active_cost
        )

    def _remove_active(self, reservation: BudgetReservation) -> None:
        removed = self._active.pop(reservation.identifier, None)
        if removed != reservation:
            raise BudgetAccountingError("Budget reservation is not active.")


class BudgetedMessageClient:
    """Apply a ledger reservation to every single transport attempt."""

    def __init__(
        self,
        client: MessageClient,
        ledger: BudgetLedger,
        *,
        estimate_input_tokens: InputTokenEstimator | None = None,
    ) -> None:
        self._client = client
        self._ledger = ledger
        self._estimate_input_tokens = (
            conservative_input_token_estimate
            if estimate_input_tokens is None
            else estimate_input_tokens
        )

    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse:
        input_tokens = self._estimate_input_tokens(request)
        reservation = self._ledger.reserve(
            input_tokens=input_tokens,
            requested_max_tokens=request.max_tokens,
        )
        bounded_request = request.model_copy(
            update={"max_tokens": reservation.max_tokens},
            deep=True,
        )
        budget_observer = _BudgetObserver(
            ledger=self._ledger,
            reservation=reservation,
            downstream=observer if observer is not None else NullStreamObserver(),
        )

        try:
            response = await self._client.create_message(
                bounded_request,
                observer=budget_observer,
            )
        except APIConnectionError as exc:
            if exc.request_may_have_completed:
                self._ledger.retain_ambiguous(reservation)
            else:
                self._ledger.release(reservation)
            raise
        except BudgetAccountingError:
            # message_start proves that at least input tokens were billed, while the
            # unseen remainder is unknown. Retain the full reservation conservatively.
            self._ledger.retain_ambiguous(reservation)
            raise
        except Exception:
            self._ledger.release(reservation)
            raise

        self._ledger.settle(reservation, response.usage)
        return response


class _BudgetObserver:
    def __init__(
        self,
        *,
        ledger: BudgetLedger,
        reservation: BudgetReservation,
        downstream: StreamObserver,
    ) -> None:
        self._ledger = ledger
        self._reservation = reservation
        self._downstream = downstream

    def on_text_delta(self, text: str) -> None:
        self._downstream.on_text_delta(text)

    def on_input_tokens(self, input_tokens: int) -> None:
        self._ledger.reconcile_input(self._reservation, input_tokens)
        self._downstream.on_input_tokens(input_tokens)

    def on_stream_retry(self) -> None:
        self._downstream.on_stream_retry()


def conservative_input_token_estimate(request: CreateMessageRequest) -> int:
    """Return Forge's conservative estimate for a JSON-only Messages request.

    Forge currently sends text, client-side-tool definitions, and JSON tool results only.
    Compact UTF-8 JSON byte length over-reserves the client-supplied wire payload. Tool
    requests additionally reserve a fixed allowance for the provider-injected tool-use
    preamble, whose tokens have no corresponding request bytes. Images, server tools,
    prompt caching, or other separately priced request features need dedicated estimators
    and pricing rules before they are added.
    """

    payload = request.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    preamble_allowance = TOOL_USE_PREAMBLE_TOKEN_ALLOWANCE if request.tools else 0
    return len(encoded) + preamble_allowance
