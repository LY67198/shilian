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


@pytest.mark.unit
class TestBM25Index:
    def test_build_and_search(self):
        from app.retrieval.bm25 import BM25Index
        idx = BM25Index("test")
        idx.build(["Python async programming guide", "Java concurrency patterns", "Python web framework Django"])
        results = idx.search("python async", top_k=2)
        assert len(results) == 2
        assert "Python async" in idx._corpus[results[0][0]]

    def test_empty_corpus_returns_empty(self):
        from app.retrieval.bm25 import BM25Index
        idx = BM25Index("empty")
        assert idx.search("query", top_k=5) == []

    def test_tokenize_bigrams(self):
        from app.retrieval.bm25 import BM25Index
        idx = BM25Index("test")
        tokens = idx._tokenize("hello world")
        assert len(tokens) > 0


@pytest.mark.unit
class TestRRF:
    def test_merges_and_ranks_by_rrf(self):
        from app.retrieval.rrf import rrf_fuse
        from app.retrieval import SearchResult
        vector = [
            SearchResult(id=1, content="A", score=0.9, source="vector"),
            SearchResult(id=2, content="B", score=0.7, source="vector"),
            SearchResult(id=3, content="C", score=0.5, source="vector"),
        ]
        bm25 = [
            SearchResult(id=2, content="B", score=0.8, source="bm25"),
            SearchResult(id=3, content="C", score=0.6, source="bm25"),
            SearchResult(id=4, content="D", score=0.4, source="bm25"),
        ]
        result = rrf_fuse(vector, bm25, k=60)
        assert result[0].id == 2  # appears in both → boosted → first
        assert result[0].source == "both"
        assert len(result) == 4
        assert {r.id for r in result} == {1, 2, 3, 4}

    def test_empty_bm25_returns_vector_only(self):
        from app.retrieval.rrf import rrf_fuse
        from app.retrieval import SearchResult
        vector = [SearchResult(id=1, content="A", score=0.9, source="vector")]
        result = rrf_fuse(vector, [], k=60)
        assert len(result) == 1 and result[0].id == 1

    def test_empty_vector_returns_bm25_only(self):
        from app.retrieval.rrf import rrf_fuse
        from app.retrieval import SearchResult
        bm25 = [SearchResult(id=5, content="E", score=0.9, source="bm25")]
        result = rrf_fuse([], bm25, k=60)
        assert len(result) == 1 and result[0].id == 5


@pytest.mark.unit
class TestRerank:
    """Tests for app.retrieval.rerank.cross_encoder_rerank"""

    async def test_reranks_by_api_results(self, monkeypatch):
        from app.retrieval.rerank import cross_encoder_rerank
        from app.retrieval import SearchResult
        candidates = [
            SearchResult(id=1, content="doc A", score=0.9, source="both"),
            SearchResult(id=2, content="doc B", score=0.8, source="both"),
            SearchResult(id=3, content="doc C", score=0.7, source="vector"),
        ]

        class MockResult:
            def __init__(self, index, relevance_score):
                self.index = index
                self.relevance_score = relevance_score

        class MockTextReRank:
            @staticmethod
            def call(**kwargs):
                resp = type("R", (), {})()
                resp.output = type("O", (), {})()
                resp.output.results = [MockResult(2, 0.98), MockResult(0, 0.85), MockResult(1, 0.40)]
                return resp

        monkeypatch.setattr("app.retrieval.rerank.TextReRank", MockTextReRank)

        result = await cross_encoder_rerank(query="test", candidates=candidates, top_k=3)
        assert len(result) == 3
        assert result[0].id == 3  # doc C now first
        assert result[0].content == "doc C"

    async def test_graceful_fallback_on_api_error(self, monkeypatch):
        from app.retrieval.rerank import cross_encoder_rerank
        from app.retrieval import SearchResult
        candidates = [SearchResult(id=1, content="only doc", score=0.9, source="both")]

        class BrokenReRank:
            @staticmethod
            def call(**kwargs):
                raise RuntimeError("API unavailable")

        monkeypatch.setattr("app.retrieval.rerank.TextReRank", BrokenReRank)
        result = await cross_encoder_rerank(query="test", candidates=candidates, top_k=1)
        assert len(result) == 1
        assert result[0].id == 1
