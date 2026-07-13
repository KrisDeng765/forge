import math
import os
from types import TracebackType
from typing import NoReturn, Self

import httpx

from forge.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OverloadedError,
    RateLimitError,
)
from forge.models import (
    CreateMessageRequest,
    ErrorDetail,
    ErrorResponse,
    MessageResponse,
)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_TIMEOUT = httpx.Timeout(
    connect=5.0,
    read=300.0,
    write=30.0,
    pool=5.0,
)


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: httpx.Timeout | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_api_key = (
            api_key if api_key is not None else os.getenv("ANTHROPIC_API_KEY")
        )
        if resolved_api_key is None or not resolved_api_key.strip():
            raise ValueError(
                "An Anthropic API key is required; pass api_key or set "
                "ANTHROPIC_API_KEY."
            )

        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "x-api-key": resolved_api_key.strip(),
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            transport=transport,
        )

    def create_message(self, request: CreateMessageRequest) -> MessageResponse:
        try:
            response = self._client.post(
                "/v1/messages",
                json=request.model_dump(mode="json", exclude_none=True),
            )
        except httpx.ConnectTimeout as exc:
            raise APITimeoutError(str(exc), phase="connect") from exc
        except httpx.ReadTimeout as exc:
            raise APITimeoutError(str(exc), phase="read") from exc
        except httpx.WriteTimeout as exc:
            raise APITimeoutError(str(exc), phase="write") from exc
        except httpx.PoolTimeout as exc:
            raise APITimeoutError(str(exc), phase="pool") from exc
        except httpx.ConnectError as exc:
            raise APIConnectionError(str(exc)) from exc
        except httpx.RequestError as exc:
            # Once a connection exists, transport failures may follow request delivery.
            raise APIConnectionError(
                str(exc),
                request_may_have_completed=True,
            ) from exc

        if response.is_success:
            try:
                return MessageResponse.model_validate(response.json())
            except ValueError as exc:
                # A 2xx response may still follow a completed, billable POST, so a
                # retry must treat an unreadable body as an ambiguous completion.
                raise APIConnectionError(
                    "Could not parse a successful API response.",
                    request_may_have_completed=True,
                ) from exc

        _raise_status_error(response)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _raise_status_error(response: httpx.Response) -> NoReturn:
    error_response = _parse_error_response(response)
    request_id = response.headers.get("request-id")
    status_code = response.status_code

    if status_code in {401, 403}:
        raise AuthenticationError(status_code, error_response, request_id)
    if status_code == 429:
        raise RateLimitError(
            status_code,
            error_response,
            request_id,
            _parse_retry_after(response.headers.get("retry-after")),
        )
    if status_code == 529:
        raise OverloadedError(status_code, error_response, request_id)

    raise APIStatusError(status_code, error_response, request_id)


def _parse_error_response(response: httpx.Response) -> ErrorResponse:
    try:
        return ErrorResponse.model_validate(response.json())
    except ValueError:
        message = response.text.strip()
        if not message:
            message = f"HTTP {response.status_code} {response.reason_phrase}".strip()

        return ErrorResponse(
            type="error",
            error=ErrorDetail(type="unknown_error", message=message),
        )


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None

    try:
        retry_after = float(value)
    except ValueError:
        return None

    if retry_after < 0 or not math.isfinite(retry_after):
        return None
    return retry_after
