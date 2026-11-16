"""Unit tests for the retrieval and generation helpers."""

from types import SimpleNamespace

import rag_utils
from rag_utils import generate, rag, retrieve


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FakeVectorStore:
    def __init__(self, documents: list[FakeDocument]):
        self.documents = documents
        self.calls = []

    def similarity_search(self, query: str, k: int):
        self.calls.append((query, k))
        return self.documents[:k]


class FakeModel:
    def __init__(self, response: str = "generated answer"):
        self.response = response
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


def test_retrieve_formats_content_and_metadata():
    vectorstore = FakeVectorStore(
        [FakeDocument("Pet fee is $50.", {"source": "room_info.pdf"})]
    )

    chunks = retrieve("What is the pet fee?", k=1, vectorstore=vectorstore)

    assert chunks == [
        "Content:\nPet fee is $50.\n\nMetadata:\n{'source': 'room_info.pdf'}"
    ]
    assert vectorstore.calls == [("What is the pet fee?", 1)]


def test_retrieve_uses_default_k():
    vectorstore = FakeVectorStore([])

    retrieve("q", vectorstore=vectorstore)

    assert vectorstore.calls == [("q", 2)]


def test_retrieve_handles_no_matches():
    vectorstore = FakeVectorStore([])

    assert retrieve("q", k=3, vectorstore=vectorstore) == []


def test_generate_includes_query_and_chunks_in_prompt():
    model = FakeModel(response="The pet fee is $50.")

    result = generate(
        query="What is the pet fee?",
        retrieved_chunks=["Pet fee is $50."],
        model=model,
    )

    assert result.content == "The pet fee is $50."
    assert len(model.prompts) == 1
    assert "What is the pet fee?" in model.prompts[0]
    assert "Pet fee is $50." in model.prompts[0]
    assert "Answer:" in model.prompts[0]


def test_rag_passes_parameters_and_returns_answer(monkeypatch):
    created_models = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.invoke_result = SimpleNamespace(content="generated answer")
            created_models.append(self)

        def invoke(self, prompt: str):
            return self.invoke_result

    monkeypatch.setattr(rag_utils, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    vectorstore = FakeVectorStore(
        [FakeDocument("Pet fee is $50.", {"source": "room_info.pdf"})]
    )

    answer, chunks = rag(
        query="What is the pet fee?",
        k=1,
        model_name="gpt-4o",
        temperature=0.2,
        top_p=0.9,
        max_tokens=128,
        vectorstore=vectorstore,
    )

    assert answer == "generated answer"
    assert chunks == [
        "Content:\nPet fee is $50.\n\nMetadata:\n{'source': 'room_info.pdf'}"
    ]
    assert vectorstore.calls == [("What is the pet fee?", 1)]
    assert created_models[0].kwargs == {
        "model": "gpt-4o",
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 128,
        "openai_api_key": "test-key",
    }


def test_rag_uses_documented_defaults(monkeypatch):
    created_models = []

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_models.append(self)

        def invoke(self, prompt: str):
            return SimpleNamespace(content="generated answer")

    monkeypatch.setattr(rag_utils, "ChatOpenAI", FakeChatOpenAI)
    vectorstore = FakeVectorStore([])

    rag("q", vectorstore=vectorstore)

    assert vectorstore.calls == [("q", 2)]
    assert created_models[0].kwargs["model"] == "gpt-4o-mini"
    assert created_models[0].kwargs["temperature"] == 0.0
    assert created_models[0].kwargs["top_p"] == 1.0
    assert created_models[0].kwargs["max_tokens"] == 512
