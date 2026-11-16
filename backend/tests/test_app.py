"""Endpoint tests for the Aurora Stays RAG API."""

from unittest.mock import Mock

import pytest

import app as app_module


@pytest.fixture
def mock_rag(monkeypatch):
    mock = Mock(return_value=("Generated answer", ["chunk one"]))
    monkeypatch.setattr(app_module, "rag", mock)
    return mock


@pytest.fixture
def mock_retrieve(monkeypatch):
    mock = Mock(return_value=["chunk one"])
    monkeypatch.setattr(app_module, "retrieve", mock)
    return mock


def test_home_returns_welcome_message(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "Welcome to the Aurora Stay RAG API!"


def test_answer_uses_default_parameters(client, mock_rag):
    response = client.post("/v1/answer", json={"query": "What is the pet fee?"})

    assert response.status_code == 200
    assert response.get_json() == {"answer": "Generated answer"}

    mock_rag.assert_called_once()
    kwargs = mock_rag.call_args.kwargs
    assert kwargs["query"] == "What is the pet fee?"
    assert kwargs["k"] == 4
    assert kwargs["model_name"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["max_tokens"] == 512
    assert kwargs["vectorstore"] is app_module.vectorstore


def test_answer_forwards_custom_parameters(client, mock_rag):
    response = client.post(
        "/v1/answer",
        json={
            "query": "  What is the pet fee?  ",
            "k": 2,
            "model_name": "gpt-4o",
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 200
    kwargs = mock_rag.call_args.kwargs
    assert kwargs["query"] == "What is the pet fee?"
    assert kwargs["k"] == 2
    assert kwargs["model_name"] == "gpt-4o"
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.8
    assert kwargs["max_tokens"] == 128


def test_answer_with_relevant_chunks_returns_answer_and_chunks(client, mock_rag):
    response = client.post(
        "/v1/answer_with_relevant_chunks",
        json={"query": "q", "k": 3, "temperature": 0.5},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "query": "q",
        "k": 3,
        "model_name": "gpt-4o-mini",
        "temperature": 0.5,
        "top_p": 1.0,
        "max_tokens": 512,
        "relevant_chunks": ["chunk one"],
        "answer": "Generated answer",
    }


def test_relevant_chunks_returns_chunks(client, mock_retrieve):
    response = client.post("/v1/relevant_chunks", json={"query": "q", "k": 2})

    assert response.status_code == 200
    assert response.get_json() == {
        "query": "q",
        "k": 2,
        "relevant_chunks": ["chunk one"],
    }
    mock_retrieve.assert_called_once_with(
        query="q",
        k=2,
        vectorstore=app_module.vectorstore,
    )


def test_relevant_chunks_ignores_generation_parameters(client, mock_retrieve):
    response = client.post(
        "/v1/relevant_chunks",
        json={
            "query": "q",
            "model_name": "",
            "temperature": 99,
            "top_p": -1,
            "max_tokens": 0,
        },
    )

    assert response.status_code == 200
    mock_retrieve.assert_called_once()


@pytest.mark.parametrize(
    ("path", "payload", "field"),
    [
        ("/v1/answer", {}, "query"),
        ("/v1/answer", {"query": ""}, "query"),
        ("/v1/answer", {"query": 123}, "query"),
        ("/v1/answer", {"query": "q", "k": 0}, "k"),
        ("/v1/answer", {"query": "q", "k": 21}, "k"),
        ("/v1/answer", {"query": "q", "model_name": ""}, "model_name"),
        ("/v1/answer", {"query": "q", "temperature": 3}, "temperature"),
        ("/v1/answer", {"query": "q", "top_p": 1.5}, "top_p"),
        ("/v1/answer", {"query": "q", "max_tokens": 0}, "max_tokens"),
        ("/v1/answer_with_relevant_chunks", {"query": "q", "k": 1.5}, "k"),
        ("/v1/relevant_chunks", {}, "query"),
        ("/v1/relevant_chunks", {"query": "q", "k": -1}, "k"),
    ],
)
def test_invalid_payloads_return_400(
    client, mock_rag, mock_retrieve, path, payload, field
):
    response = client.post(path, json=payload)

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"]["field"] == field
    mock_rag.assert_not_called()
    mock_retrieve.assert_not_called()


def test_missing_body_returns_400(client, mock_rag):
    response = client.post("/v1/answer")

    assert response.status_code == 400
    assert response.get_json() == {
        "error": {"message": "Request body must be a JSON object."}
    }
    mock_rag.assert_not_called()


@pytest.mark.parametrize(
    ("data", "content_type"),
    [
        ("{invalid json", "application/json"),
        ("not json", "text/plain"),
    ],
)
def test_malformed_body_returns_400(client, mock_rag, data, content_type):
    response = client.post("/v1/answer", data=data, content_type=content_type)

    assert response.status_code == 400
    assert response.get_json()["error"]["message"] == (
        "Request body must be a JSON object."
    )
    mock_rag.assert_not_called()


def test_unknown_route_returns_json_404(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.get_json() == {"error": {"message": "Not found."}}


def test_wrong_method_returns_json_405(client):
    response = client.get("/v1/answer")

    assert response.status_code == 405
    assert response.get_json() == {"error": {"message": "Method not allowed."}}


def test_answer_returns_json_when_generation_fails(client, mock_rag, monkeypatch):
    monkeypatch.setattr(app_module.rag_api.logger, "exception", Mock())
    mock_rag.side_effect = RuntimeError("LLM unavailable")

    response = client.post("/v1/answer", json={"query": "q"})

    assert response.status_code == 502
    assert response.get_json() == {
        "error": {"message": "Unable to generate an answer."}
    }


def test_relevant_chunks_returns_json_when_retrieval_fails(
    client, mock_retrieve, monkeypatch
):
    monkeypatch.setattr(app_module.rag_api.logger, "exception", Mock())
    mock_retrieve.side_effect = RuntimeError("vector store unavailable")

    response = client.post("/v1/relevant_chunks", json={"query": "q"})

    assert response.status_code == 502
    assert response.get_json() == {
        "error": {"message": "Unable to retrieve relevant chunks."}
    }
