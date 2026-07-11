"""Self-check loop node unit tests"""
from __future__ import annotations

import pytest


@pytest.mark.unit
class TestCheckSufficiencyNode:
    """check_sufficiency_node — LLM sufficiency evaluation"""

    async def test_empty_results_marks_insufficient(self, monkeypatch):
        from app.workflows.retrieval_check.nodes.check_sufficiency import check_sufficiency_node

        state = {
            "query": "test query",
            "current_query": "test query",
            "retrieval_results": [],
            "retry_count": 0,
        }

        result = await check_sufficiency_node(state)
        assert result["is_sufficient"] is False
        assert result["retry_reason"] == "too_few"

    async def test_returns_llm_result(self, monkeypatch):
        from app.workflows.retrieval_check.nodes.check_sufficiency import check_sufficiency_node, SufficiencyResult

        class MockChain:
            async def ainvoke(self, *args, **kwargs):
                return SufficiencyResult(sufficient=True, reason="ok")

        mock_llm = type("MockLLM", (), {
            "with_structured_output": lambda self, model, **kwargs: MockChain(),
        })()

        monkeypatch.setattr(
            "app.workflows.retrieval_check.nodes.check_sufficiency.get_chat_llm",
            lambda **kw: mock_llm,
        )

        class MockPrompt:
            def __or__(self, other):
                return other

        monkeypatch.setattr(
            "app.workflows.retrieval_check.nodes.check_sufficiency.load_prompt",
            lambda name: MockPrompt(),
        )

        state = {
            "query": "test",
            "current_query": "test",
            "retrieval_results": [
                {"content": "relevant content here", "score": 0.9},
            ],
            "retry_count": 0,
        }

        result = await check_sufficiency_node(state)
        assert result["is_sufficient"] is True
        assert result["retry_reason"] == "ok"


@pytest.mark.unit
class TestRewriteQueryNode:
    """rewrite_query_node — LLM query rewriting"""

    async def test_increments_retry_count(self, monkeypatch):
        from app.workflows.retrieval_check.nodes.rewrite_query import rewrite_query_node

        class MockResponse:
            content = "rewritten query text"

        class MockChain:
            async def ainvoke(self, *args, **kwargs):
                return MockResponse()

        monkeypatch.setattr(
            "app.workflows.retrieval_check.nodes.rewrite_query.get_chat_llm",
            lambda **kw: MockChain(),
        )

        class MockPrompt:
            def __or__(self, other):
                return other

        monkeypatch.setattr(
            "app.workflows.retrieval_check.nodes.rewrite_query.load_prompt",
            lambda name: MockPrompt(),
        )

        state = {
            "query": "original query",
            "retry_count": 0,
            "retry_reason": "low_relevance",
            "retrieval_history": [],
        }

        result = await rewrite_query_node(state)
        assert result["retry_count"] == 1
        assert result["current_query"] == "rewritten query text"


@pytest.mark.unit
class TestFormatContextNode:
    """format_context_node — context compilation"""

    async def test_deduplicates_by_content_prefix(self):
        from app.workflows.retrieval_check.nodes.format_context import format_context_node

        state = {
            "query": "test",
            "retry_count": 1,
            "retrieval_history": [
                {
                    "round": 0,
                    "query": "test",
                    "results": [
                        {"content": "document A content", "score": 0.9},
                        {"content": "document B content", "score": 0.7},
                    ],
                },
                {
                    "round": 1,
                    "query": "test rewritten",
                    "results": [
                        {"content": "document A content", "score": 0.85},
                        {"content": "document C content", "score": 0.8},
                    ],
                },
            ],
        }

        result = await format_context_node(state)
        # Duplicate "document A" should only appear once
        contexts = result["final_context"]
        assert len(contexts) == 3
        assert "document A content" in contexts
        assert "document B content" in contexts
        assert "document C content" in contexts
        # debug info present
        assert result["debug_info"]["total_unique_results"] == 3
        assert result["debug_info"]["total_retrieval_rounds"] == 2


@pytest.mark.unit
class TestRetrieveNode:
    """retrieve_node — pipeline search"""

    async def test_appends_to_history(self, monkeypatch):
        from app.workflows.retrieval_check.nodes.retrieve import retrieve_node
        from app.retrieval import SearchResult

        class MockPipeline:
            async def search(self, query, filters=None):
                return [
                    SearchResult(id=1, content="result one", score=0.9, source="both"),
                ]

        state = {
            "query": "test query",
            "current_query": "test query",
            "retry_count": 0,
            "retrieval_history": [],
            "custom": {"pipeline": MockPipeline()},
        }

        result = await retrieve_node(state)
        assert len(result["retrieval_results"]) == 1
        assert result["retrieval_results"][0]["content"] == "result one"
        assert len(result["retrieval_history"]) == 1
        assert result["retrieval_history"][0]["round"] == 0
