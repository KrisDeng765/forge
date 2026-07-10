import pytest

from forge.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OverloadedError,
    RateLimitError,
    TimeoutPhase,
)
from forge.models import ErrorResponse


def make_error_response() -> ErrorResponse:
    return ErrorResponse.model_validate(
        {
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded.",
            },
        }
    )


def test_error_response_matches_wire_shape() -> None:
    payload = {
        "type": "error",
        "error": {
            "type": "authentication_error",
            "message": "Invalid API key.",
        },
    }

    response = ErrorResponse.model_validate(payload)

    assert response.model_dump(mode="json") == payload


def test_api_status_error_preserves_response_metadata() -> None:
    error_response = make_error_response()

    error = AuthenticationError(401, error_response, "req_123")

    assert isinstance(error, APIStatusError)
    assert error.status_code == 401
    assert error.error_response is error_response
    assert error.request_id == "req_123"
    assert str(error) == "API request failed with status 401: Rate limit exceeded."


def test_rate_limit_error_preserves_retry_after() -> None:
    error = RateLimitError(429, make_error_response(), "req_123", 7.0)

    assert error.retry_after == 7.0
    assert error.status_code == 429


def test_overloaded_error_is_an_api_status_error() -> None:
    error = OverloadedError(529, make_error_response(), None)

    assert isinstance(error, APIStatusError)
    assert error.status_code == 529


@pytest.mark.parametrize(
    ("phase", "request_may_have_completed"),
    [
        ("connect", False),
        ("pool", False),
        ("read", True),
        ("write", True),
    ],
)
def test_timeout_phase_preserves_ambiguous_completion(
    phase: TimeoutPhase,
    request_may_have_completed: bool,
) -> None:
    error = APITimeoutError("API request timed out.", phase=phase)

    assert isinstance(error, APIConnectionError)
    assert error.phase == phase
    assert error.request_may_have_completed is request_may_have_completed
