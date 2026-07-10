"""Retrieval pipeline unit tests"""
from __future__ import annotations

import pytest

from app.retrieval import SearchResult


@pytest.mark.unit
class TestVectorSearch:
    """Tests for app.retrieval.vector.vector_search"""

    async def test_returns_search_results(self, monkeypatch):
        from app.retrieval.vector import vector_search, _COLLECTION_MAP

        async def mock_embed(text):
            return [0.1] * 1024

        monkeypatch.setattr("app.retrieval.vector.embed_text", mock_embed)

        class MockKnowledge:
            @staticmethod
            def search(client, query_vector, top_k, document_ids=None, min_score=0.0):
                return [
                    {"id": 1, "content": "chunk one", "similarity": 0.9,
                     "document_id": 10, "chunk_index": 0, "content_hash": "a", "metadata": {}},
                ]

        monkeypatch.setitem(_COLLECTION_MAP, "knowledge_chunks", MockKnowledge)

        results = await vector_search(
            client=None,
            query="test query",
            collection="knowledge_chunks",
            top_k=4,
        )

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == 1
        assert results[0].content == "chunk one"
        assert results[0].score == 0.9
        assert results[0].source == "vector"

    async def test_passes_filters_to_question_bank(self, monkeypatch):
        from app.retrieval.vector import vector_search, _COLLECTION_MAP

        async def mock_embed(text):
            return [0.1] * 1024

        monkeypatch.setattr("app.retrieval.vector.embed_text", mock_embed)

        captured = {}

        class MockQuestionBank:
            @staticmethod
            def search(client, query_vector, top_k, position_tag=None, difficulty=None, min_score=0.7):
                captured.update({"position_tag": position_tag, "difficulty": difficulty})
                return []

        monkeypatch.setitem(_COLLECTION_MAP, "question_bank", MockQuestionBank)

        await vector_search(
            client=None,
            query="Python",
            collection="question_bank",
            top_k=10,
            filters={"position_tag": "python_backend", "difficulty": "medium"},
        )

        assert captured["position_tag"] == "python_backend"
        assert captured["difficulty"] == "medium"

    async def test_empty_on_unknown_collection(self, monkeypatch):
        from app.retrieval.vector import vector_search

        async def mock_embed(text):
            return [0.1] * 1024

        monkeypatch.setattr("app.retrieval.vector.embed_text", mock_embed)

        results = await vector_search(
            client=None,
            query="test",
            collection="nonexistent",
            top_k=5,
        )

        assert results == []
