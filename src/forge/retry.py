"""Retry an async MessageClient without giving it ownership of loop policy or state."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Protocol

from forge.errors import (
    AmbiguousCompletionBudgetError,
    APIConnectionError,
    APIStatusError,
    BudgetExceededError,
    OverloadedError,
    RateLimitError,
)
from forge.models import CreateMessageRequest, MessageResponse
from forge.streaming import StreamObserver

type Sleep = Callable[[float], Awaitable[None]]
type Uniform = Callable[[float, float], float]


class MessageClient(Protocol):
    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse: ...


class RetryingMessageClient:
    """Add bounded retry policy around a single-attempt MessageClient.

    The wrapper deliberately sits outside the budget wrapper. Each retry therefore calls
    the budgeted client again, retaining the first full reservation after ambiguous
    completion and reserving the retry independently.
    """

    def __init__(
        self,
        client: MessageClient,
        *,
        sleep: Sleep = asyncio.sleep,
        uniform: Uniform = random.uniform,
        max_attempts: int = 4,
        base_delay_seconds: float = 0.5,
        max_delay_seconds: float = 8.0,
        max_total_delay_seconds: float = 30.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        if base_delay_seconds < 0 or max_delay_seconds < 0:
            raise ValueError("Retry delays cannot be negative.")
        if max_delay_seconds < base_delay_seconds:
            raise ValueError("max_delay_seconds cannot be lower than base_delay_seconds.")
        if max_total_delay_seconds < 0:
            raise ValueError("max_total_delay_seconds cannot be negative.")

        self._client = client
        self._sleep = sleep
        self._uniform = uniform
        self._max_attempts = max_attempts
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._max_total_delay_seconds = max_total_delay_seconds

    async def create_message(
        self,
        request: CreateMessageRequest,
        *,
        observer: StreamObserver | None = None,
    ) -> MessageResponse:
        attempts = 0
        ambiguous_retries = 0
        total_delay = 0.0
        pending_ambiguous_completion: APIConnectionError | None = None

        while True:
            attempts += 1
            try:
                return await self._client.create_message(request, observer=observer)
            except Exception as exc:
                if (
                    isinstance(exc, BudgetExceededError)
                    and pending_ambiguous_completion is not None
                ):
                    raise AmbiguousCompletionBudgetError(
                        "A prior Messages request may have completed, but the "
                        "budget cannot fund its one permitted retry."
                    ) from pending_ambiguous_completion

                retry_after, ambiguous_completion = self._retry_policy(exc)
                if retry_after is None or attempts >= self._max_attempts:
                    raise
                if ambiguous_completion and ambiguous_retries >= 1:
                    raise

                delay = (
                    retry_after
                    if retry_after >= 0
                    else self._full_jitter_delay(attempts)
                )
                if total_delay + delay > self._max_total_delay_seconds:
                    raise

                if observer is not None:
                    observer.on_stream_retry()
                await self._sleep(delay)
                total_delay += delay
                if ambiguous_completion:
                    if not isinstance(exc, APIConnectionError):
                        raise AssertionError(
                            "Only connection failures can be ambiguous completions."
                        ) from exc
                    pending_ambiguous_completion = exc
                    ambiguous_retries += 1

    def _retry_policy(self, exc: Exception) -> tuple[float | None, bool]:
        if isinstance(exc, RateLimitError):
            return (exc.retry_after if exc.retry_after is not None else -1.0), False
        if isinstance(exc, OverloadedError):
            return -1.0, False
        if isinstance(exc, APIConnectionError):
            return -1.0, exc.request_may_have_completed
        if isinstance(exc, APIStatusError) and 500 <= exc.status_code < 600:
            # A received 5xx is an explicit provider failure, unlike a lost response.
            return -1.0, False
        return None, False

    def _full_jitter_delay(self, failed_attempt: int) -> float:
        upper_bound = min(
            self._max_delay_seconds,
            self._base_delay_seconds * (2 ** (failed_attempt - 1)),
        )
        return self._uniform(0.0, upper_bound)
