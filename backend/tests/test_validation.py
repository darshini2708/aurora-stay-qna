"""Unit tests for request validation helpers."""

import pytest

from validation import (
    DEFAULT_K,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_K,
    MAX_MAX_TOKENS,
    MAX_MODEL_NAME_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_TEMPERATURE,
    ValidationError,
    parse_rag_request,
)


def test_parse_defaults():
    request = parse_rag_request({"query": "What is the pet policy?"})

    assert request.query == "What is the pet policy?"
    assert request.k == DEFAULT_K
    assert request.model_name == DEFAULT_MODEL_NAME
    assert request.temperature == DEFAULT_TEMPERATURE
    assert request.top_p == DEFAULT_TOP_P
    assert request.max_tokens == DEFAULT_MAX_TOKENS


def test_query_is_trimmed():
    request = parse_rag_request({"query": "  What is the pet policy?  "})

    assert request.query == "What is the pet policy?"


def test_retrieval_only_ignores_generation_fields():
    request = parse_rag_request(
        {
            "query": "What is the pet policy?",
            "k": 2,
            "model_name": "",
            "temperature": 99,
            "top_p": -1,
            "max_tokens": 0,
        },
        require_generation=False,
    )

    assert request.query == "What is the pet policy?"
    assert request.k == 2


def test_retrieval_only_still_validates_k():
    with pytest.raises(ValidationError) as error:
        parse_rag_request({"query": "q", "k": 0}, require_generation=False)

    assert error.value.field == "k"


@pytest.mark.parametrize("payload", [None, [], "query", 4, True])
def test_body_must_be_a_json_object(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field is None
    assert "JSON object" in error.value.message


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": None},
        {"query": ""},
        {"query": "   "},
        {"query": 123},
        {"query": ["What is the pet policy?"]},
    ],
)
def test_query_must_be_a_non_empty_string(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field == "query"


def test_query_length_is_limited():
    with pytest.raises(ValidationError) as error:
        parse_rag_request({"query": "a" * (MAX_QUERY_LENGTH + 1)})

    assert error.value.field == "query"
    assert str(MAX_QUERY_LENGTH) in error.value.message


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "k": 0},
        {"query": "q", "k": MAX_K + 1},
        {"query": "q", "k": 1.0},
        {"query": "q", "k": "4"},
        {"query": "q", "k": True},
        {"query": "q", "k": None},
    ],
)
def test_k_must_be_an_integer_in_range(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field == "k"


@pytest.mark.parametrize("k", [1, MAX_K])
def test_k_boundaries_are_accepted(k):
    assert parse_rag_request({"query": "q", "k": k}).k == k


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "model_name": ""},
        {"query": "q", "model_name": "   "},
        {"query": "q", "model_name": 123},
        {"query": "q", "model_name": "m" * (MAX_MODEL_NAME_LENGTH + 1)},
    ],
)
def test_model_name_must_be_a_non_empty_string(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field == "model_name"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "temperature": -0.1},
        {"query": "q", "temperature": MAX_TEMPERATURE + 0.1},
        {"query": "q", "temperature": "0.5"},
        {"query": "q", "temperature": True},
        {"query": "q", "temperature": float("nan")},
        {"query": "q", "temperature": float("inf")},
    ],
)
def test_temperature_must_be_a_finite_number_in_range(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field == "temperature"


@pytest.mark.parametrize("temperature", [0, MAX_TEMPERATURE, 0.7])
def test_temperature_boundaries_are_accepted(temperature):
    request = parse_rag_request({"query": "q", "temperature": temperature})

    assert request.temperature == float(temperature)


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "top_p": -0.1},
        {"query": "q", "top_p": 1.1},
        {"query": "q", "top_p": "1.0"},
        {"query": "q", "top_p": False},
        {"query": "q", "top_p": float("nan")},
    ],
)
def test_top_p_must_be_a_finite_number_in_range(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field == "top_p"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "q", "max_tokens": 0},
        {"query": "q", "max_tokens": MAX_MAX_TOKENS + 1},
        {"query": "q", "max_tokens": 10.0},
        {"query": "q", "max_tokens": "10"},
        {"query": "q", "max_tokens": True},
        {"query": "q", "max_tokens": None},
    ],
)
def test_max_tokens_must_be_an_integer_in_range(payload):
    with pytest.raises(ValidationError) as error:
        parse_rag_request(payload)

    assert error.value.field == "max_tokens"


@pytest.mark.parametrize("max_tokens", [1, MAX_MAX_TOKENS, 512])
def test_max_tokens_boundaries_are_accepted(max_tokens):
    request = parse_rag_request({"query": "q", "max_tokens": max_tokens})

    assert request.max_tokens == max_tokens


def test_validation_error_serialises_to_api_shape():
    error = ValidationError("Field 'query' is required.", field="query")

    assert error.to_dict() == {
        "error": {
            "message": "Field 'query' is required.",
            "field": "query",
        }
    }


def test_validation_error_without_field_serialises_to_api_shape():
    error = ValidationError("Request body must be a JSON object.")

    assert error.to_dict() == {
        "error": {"message": "Request body must be a JSON object."}
    }
