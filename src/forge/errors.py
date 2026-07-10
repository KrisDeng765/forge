from typing import Literal

from forge.models import ErrorResponse

type TimeoutPhase = Literal["connect", "read", "write", "pool"]


class ForgeError(Exception):
    """Base exception for all Forge runtime errors."""


class APIStatusError(ForgeError):
    def __init__(
        self,
        status_code: int,
        error_response: ErrorResponse,
        request_id: str | None,
    ) -> None:
        self.status_code = status_code
        self.error_response = error_response
        self.request_id = request_id

        super().__init__(
            f"API request failed with status {status_code}: "
            f"{error_response.error.message}"
        )


class AuthenticationError(APIStatusError):
    """Raised for authentication and permission failures (401/403)."""


class RateLimitError(APIStatusError):
    def __init__(
        self,
        status_code: int,
        error_response: ErrorResponse,
        request_id: str | None,
        retry_after: float | None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(status_code, error_response, request_id)


class OverloadedError(APIStatusError):
    """Raised when the provider is overloaded (529)."""


class APIConnectionError(ForgeError):
    def __init__(
        self,
        message: str,
        *,
        request_may_have_completed: bool = False,
    ) -> None:
        self.request_may_have_completed = request_may_have_completed
        super().__init__(message)


class APITimeoutError(APIConnectionError):
    def __init__(
        self,
        message: str,
        *,
        phase: TimeoutPhase,
    ) -> None:
        self.phase = phase

        # Read/write timeouts can occur after some or all request bytes were sent.
        super().__init__(
            message,
            request_may_have_completed=phase in {"read", "write"},
        )
