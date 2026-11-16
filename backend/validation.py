"""Validation helpers for the Aurora Stays RAG API.

The API only accepts JSON request bodies. Validating them up front lets the
endpoints return a clear ``400`` response instead of raising an unhandled
exception (which Flask would otherwise turn into an HTML ``500`` page).
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any

# Defaults kept in one place so the API and its documentation stay aligned.
DEFAULT_K = 4
DEFAULT_MODEL_NAME = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 512

# Bounds that protect the service from wasteful or accidental requests.
MIN_K = 1
MAX_K = 20
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_P = 0.0
MAX_TOP_P = 1.0
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 16384
MAX_QUERY_LENGTH = 2000
MAX_MODEL_NAME_LENGTH = 100


class ValidationError(ValueError):
    """Raised when a request body contains missing or invalid fields."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.message = message
        self.field = field

    def to_dict(self) -> dict[str, dict[str, str]]:
        """Return the error in the JSON shape used by the API."""
        error = {"message": self.message}
        if self.field is not None:
            error["field"] = self.field
        return {"error": error}


@dataclass(frozen=True)
class RAGRequest:
    """Validated parameters for a retrieval or generation request."""

    query: str
    k: int = DEFAULT_K
    model_name: str = DEFAULT_MODEL_NAME
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS


def parse_rag_request(
    payload: Any,
    *,
    require_generation: bool = True,
) -> RAGRequest:
    """Validate a decoded JSON body and return normalised RAG parameters.

    ``require_generation`` should be ``False`` for endpoints that only retrieve
    chunks, because model settings are not used by those endpoints.
    """
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    query = _required_string(payload, "query", max_length=MAX_QUERY_LENGTH)
    k = _optional_int(payload, "k", DEFAULT_K, minimum=MIN_K, maximum=MAX_K)

    if not require_generation:
        return RAGRequest(query=query, k=k)

    return RAGRequest(
        query=query,
        k=k,
        model_name=_optional_string(
            payload,
            "model_name",
            DEFAULT_MODEL_NAME,
            max_length=MAX_MODEL_NAME_LENGTH,
        ),
        temperature=_optional_number(
            payload,
            "temperature",
            DEFAULT_TEMPERATURE,
            minimum=MIN_TEMPERATURE,
            maximum=MAX_TEMPERATURE,
        ),
        top_p=_optional_number(
            payload,
            "top_p",
            DEFAULT_TOP_P,
            minimum=MIN_TOP_P,
            maximum=MAX_TOP_P,
        ),
        max_tokens=_optional_int(
            payload,
            "max_tokens",
            DEFAULT_MAX_TOKENS,
            minimum=MIN_MAX_TOKENS,
            maximum=MAX_MAX_TOKENS,
        ),
    )


def _required_string(
    payload: dict[str, Any],
    field: str,
    *,
    max_length: int,
) -> str:
    if field not in payload or payload[field] is None:
        raise ValidationError(f"Field '{field}' is required.", field=field)

    value = payload[field]
    if not isinstance(value, str):
        raise ValidationError(f"Field '{field}' must be a string.", field=field)

    value = value.strip()
    if not value:
        raise ValidationError(f"Field '{field}' must not be empty.", field=field)
    if len(value) > max_length:
        raise ValidationError(
            f"Field '{field}' must be at most {max_length} characters.",
            field=field,
        )
    return value


def _optional_string(
    payload: dict[str, Any],
    field: str,
    default: str,
    *,
    max_length: int,
) -> str:
    if field not in payload:
        return default
    return _required_string(payload, field, max_length=max_length)


def _optional_int(
    payload: dict[str, Any],
    field: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if field not in payload:
        return default

    value = payload[field]

    # bool is a subclass of int, so check it explicitly.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"Field '{field}' must be an integer.", field=field)
    if not minimum <= value <= maximum:
        raise ValidationError(
            f"Field '{field}' must be between {minimum} and {maximum}.",
            field=field,
        )
    return value


def _optional_number(
    payload: dict[str, Any],
    field: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if field not in payload:
        return default

    value = payload[field]

    # bool is a subclass of int, so reject it as a numeric value.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"Field '{field}' must be a number.", field=field)
    if not isfinite(value):
        raise ValidationError(f"Field '{field}' must be a finite number.", field=field)

    value = float(value)
    if not minimum <= value <= maximum:
        raise ValidationError(
            f"Field '{field}' must be between {minimum} and {maximum}.",
            field=field,
        )
    return value
