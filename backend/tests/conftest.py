"""Shared pytest configuration for the backend tests.

The real application loads a HuggingFace embedding model and the FAISS index
at import time. Those resources are slow and unnecessary for endpoint tests,
so lightweight stubs are installed before ``app`` is imported.
"""

import sys
import types
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class _FakeIndex:
    ntotal = 0


class _FakeVectorStore:
    index = _FakeIndex()

    def similarity_search(self, query: str, k: int):
        return []


class _FakeHuggingFaceEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeFAISS:
    @classmethod
    def load_local(cls, *args, **kwargs):
        return _FakeVectorStore()


def _install_langchain_stubs() -> None:
    huggingface = types.ModuleType("langchain_huggingface")
    huggingface.HuggingFaceEmbeddings = _FakeHuggingFaceEmbeddings
    sys.modules["langchain_huggingface"] = huggingface

    community = types.ModuleType("langchain_community")
    community.__path__ = []
    vectorstores = types.ModuleType("langchain_community.vectorstores")
    vectorstores.FAISS = _FakeFAISS
    sys.modules["langchain_community"] = community
    sys.modules["langchain_community.vectorstores"] = vectorstores


_install_langchain_stubs()


@pytest.fixture
def client():
    """Return a Flask test client for the RAG API."""
    import app as app_module

    app_module.rag_api.config.update(TESTING=True)
    with app_module.rag_api.test_client() as test_client:
        yield test_client
