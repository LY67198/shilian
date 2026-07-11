# Phase 3 RAG Pipeline Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade retrieval from pure Milvus vector search to hybrid vector+BM25+RRF+rerank pipeline, add LangGraph self-check loop for query rewriting, and build offline RAGAS evaluation framework.

**Architecture:** New `app/retrieval/` module with composable pipeline stages, new `app/workflows/retrieval_check/` LangGraph StateGraph for deep self-check (retrieve → check sufficiency → rewrite → re-retrieve, max 2 rounds), and `eval/` for golden set + RAGAS 5-metric evaluation. Existing `retrieve_knowledge_node` delegates to `RetrievalCheckService`; `interview_service._generate_questions_with_rag` uses `RetrievalPipeline` for question bank recall.

**Spec:** `docs/superpowers/specs/2026-07-10-phase-3-rag-upgrade-design.md`

---

### Task 1: Dependencies and configuration

**Files:**
- Modify: `ai-interview-backend/requirements.txt` (append)
- Modify: `ai-interview-backend/app/core/config.py` (append after line 120)

- [ ] **Step 1: Add deps to requirements.txt**

```bash
echo "" >> ai-interview-backend/requirements.txt
echo "# === Phase 3 RAG 管线升级 ===" >> ai-interview-backend/requirements.txt
echo "rank-bm25==0.2.2" >> ai-interview-backend/requirements.txt
echo "ragas==0.2.15" >> ai-interview-backend/requirements.txt
```

- [ ] **Step 2: Add config settings to app/core/config.py**

After `QUESTION_BANK_TOP_K: int = 20` (line 120), insert:

```python
    # ============= Phase 3: Hybrid Retrieval =============
    DASHSCOPE_RERANK_MODEL: str = "gte-rerank"
    RRF_K: int = 60
    BM25_TOP_K: int = 20
    VECTOR_TOP_K: int = 20
    RERANK_TOP_K: int = 10
    SELF_CHECK_MAX_RETRIES: int = 2
```

- [ ] **Step 3: Rebuild and install in container**

```bash
cd ai-interview-backend
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
docker exec shilian-app pip install rank-bm25==0.2.2 ragas==0.2.15
```

Expected: `Successfully installed rank-bm25-0.2.2 ragas-0.2.15 ...`

- [ ] **Step 4: Commit**

```bash
cd ai-interview-backend
git add requirements.txt app/core/config.py
git commit -m "chore: add rank-bm25 + ragas deps and Phase 3 config settings

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: SearchResult model and retrieval __init__

**Files:**
- Create: `ai-interview-backend/app/retrieval/__init__.py`

- [ ] **Step 1: Write module init**

```python
"""Hybrid retrieval pipeline — vector + BM25 + RRF + rerank

Usage:
    from app.retrieval import SearchResult
    from app.retrieval.pipeline import RetrievalPipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Normalized search result across all retrieval stages."""

    id: int
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "vector"  # "vector" | "bm25" | "both"
```

- [ ] **Step 2: Verify import**

```bash
docker exec shilian-app python -c "from app.retrieval import SearchResult; r = SearchResult(id=1, content='test', score=0.9); print(r)"
```

Expected: `SearchResult(id=1, content='test', score=0.9, metadata={}, source='vector')`

- [ ] **Step 3: Commit**

```bash
git add app/retrieval/__init__.py
git commit -m "feat: add SearchResult model and retrieval module init

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Vector recall wrapper

**Files:**
- Create: `ai-interview-backend/app/retrieval/vector.py`
- Create: `ai-interview-backend/tests/unit/test_retrieval.py`

- [ ] **Step 1: Write tests**

```python
"""Retrieval pipeline unit tests"""
from __future__ import annotations

import pytest

from app.retrieval import SearchResult


@pytest.mark.unit
class TestVectorSearch:
    """Tests for app.retrieval.vector.vector_search"""

    async def test_returns_search_results(self, monkeypatch):
        from app.retrieval.vector import vector_search

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

        monkeypatch.setattr("app.retrieval.vector.knowledge_vdb", MockKnowledge)

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
        from app.retrieval.vector import vector_search

        async def mock_embed(text):
            return [0.1] * 1024

        monkeypatch.setattr("app.retrieval.vector.embed_text", mock_embed)

        captured = {}

        class MockQuestionBank:
            @staticmethod
            def search(client, query_vector, top_k, position_tag=None, difficulty=None, min_score=0.7):
                captured.update({"position_tag": position_tag, "difficulty": difficulty})
                return []

        monkeypatch.setattr("app.retrieval.vector.question_bank_vdb", MockQuestionBank)

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestVectorSearch -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval.vector'`

- [ ] **Step 3: Write implementation**

```python
"""Vector recall — wraps existing Milvus collection search"""
from __future__ import annotations

import logging
from typing import Optional

from pymilvus import MilvusClient

from app.llm.embedding import embed_text
from app.retrieval import SearchResult
from app.vector_db.collections import knowledge as knowledge_vdb
from app.vector_db.collections import question_bank as question_bank_vdb

logger = logging.getLogger(__name__)

_COLLECTION_MAP = {
    "knowledge_chunks": knowledge_vdb,
    "question_bank": question_bank_vdb,
}


async def vector_search(
    client: MilvusClient,
    query: str,
    collection: str,
    top_k: int = 20,
    filters: Optional[dict] = None,
) -> list[SearchResult]:
    """Run vector similarity search against a Milvus collection.

    Args:
        client: MilvusClient instance.
        query: Raw text to embed and search.
        collection: "knowledge_chunks" or "question_bank".
        top_k: Number of results to return.
        filters: Collection-specific filters.
            knowledge_chunks: {"document_ids": [1, 2]}
            question_bank: {"position_tag": "python", "difficulty": "medium"}

    Returns:
        List of SearchResult sorted by similarity descending.
    """
    coll = _COLLECTION_MAP.get(collection)
    if coll is None:
        logger.warning(f"Unknown collection: {collection}")
        return []

    try:
        query_vec = await embed_text(query)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return []

    filters = filters or {}

    try:
        if collection == "knowledge_chunks":
            raw = coll.search(
                client=client,
                query_vector=query_vec,
                top_k=top_k,
                document_ids=filters.get("document_ids"),
                min_score=filters.get("min_score", 0.0),
            )
        else:
            raw = coll.search(
                client=client,
                query_vector=query_vec,
                top_k=top_k,
                position_tag=filters.get("position_tag"),
                difficulty=filters.get("difficulty"),
                min_score=filters.get("min_score", 0.7),
            )
    except Exception as e:
        logger.warning(f"Vector search failed for {collection}: {e}")
        return []

    return [
        SearchResult(
            id=r.get("id", 0),
            content=r.get("content") or r.get("question", ""),
            score=r.get("similarity", 0.0),
            metadata={k: v for k, v in r.items()
                      if k not in ("id", "content", "question", "similarity")},
            source="vector",
        )
        for r in raw
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestVectorSearch -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/vector.py tests/unit/test_retrieval.py
git commit -m "feat: add vector recall wrapper for retrieval pipeline

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: BM25 keyword index

**Files:**
- Create: `ai-interview-backend/app/retrieval/bm25.py`
- Append tests to: `ai-interview-backend/tests/unit/test_retrieval.py`

- [ ] **Step 1: Write tests**

Append to `tests/unit/test_retrieval.py`:

```python

@pytest.mark.unit
class TestBM25Index:
    """Tests for app.retrieval.bm25.BM25Index"""

    def test_build_and_search(self):
        from app.retrieval.bm25 import BM25Index

        idx = BM25Index("test_collection")
        idx.build([
            "Python async programming guide",
            "Java concurrency patterns",
            "Python web framework Django",
        ])

        results = idx.search("python async", top_k=2)

        assert len(results) == 2
        # First result should be about Python async
        assert "Python async" in idx._corpus[results[0][0]]

    def test_empty_corpus_returns_empty(self):
        from app.retrieval.bm25 import BM25Index

        idx = BM25Index("empty")
        results = idx.search("query", top_k=5)
        assert results == []

    def test_tokenize_bigrams(self):
        from app.retrieval.bm25 import BM25Index

        idx = BM25Index("test")
        tokens = idx._tokenize("hello world")
        # character bigrams for Chinese-friendly matching
        assert len(tokens) > 0
        assert "he" in tokens or "hello" in " ".join(tokens)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestBM25Index -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.retrieval.bm25'`

- [ ] **Step 3: Write BM25Index implementation**

```python
"""BM25 keyword recall using rank-bm25 library"""
from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """Per-collection BM25 keyword index.

    Built from corpus texts on startup; incrementally updated on CRUD.
    Rebuild is O(N) and <50ms for hundreds of docs.
    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self._index: BM25Okapi | None = None
        self._corpus: list[list[str]] = []
        self._texts: list[str] = []

    def build(self, texts: list[str]) -> None:
        """Tokenize corpus texts and build BM25Okapi index."""
        self._texts = list(texts)
        self._corpus = [self._tokenize(t) for t in texts]
        if self._corpus:
            self._index = BM25Okapi(self._corpus)
        else:
            self._index = None
        logger.info(
            "BM25 index built for %s: %d docs",
            self._collection_name,
            len(texts),
        )

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search the BM25 index.

        Args:
            query: Raw query text.
            top_k: Max results to return.

        Returns:
            List of (corpus_index, bm25_score), sorted by score descending.
        """
        if self._index is None or not self._corpus:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # Sort by score descending, take top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def get_text(self, idx: int) -> str:
        """Get original text by corpus index."""
        if 0 <= idx < len(self._texts):
            return self._texts[idx]
        return ""

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize into character bigrams for Chinese-friendly matching.

        Bigrams work surprisingly well for CJK text without extra dependencies.
        If golden set evaluation shows poor recall, switch to jieba.
        """
        chars = list(text)
        if len(chars) <= 1:
            return [text] if text else []
        return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestBM25Index -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/bm25.py tests/unit/test_retrieval.py
git commit -m "feat: add BM25 keyword index with character bigram tokenizer

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: RRF fusion

**Files:**
- Create: `ai-interview-backend/app/retrieval/rrf.py`
- Append tests to: `ai-interview-backend/tests/unit/test_retrieval.py`

- [ ] **Step 1: Write tests**

Append to `tests/unit/test_retrieval.py`:

```python

@pytest.mark.unit
class TestRRF:
    """Tests for app.retrieval.rrf.rrf_fuse"""

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

        # id=2 appears in both lists → boosted → ranked first
        assert result[0].id == 2
        assert result[0].source == "both"
        # Should have all 4 unique items
        assert len(result) == 4
        ids = [r.id for r in result]
        assert set(ids) == {1, 2, 3, 4}

    def test_empty_bm25_returns_vector_only(self):
        from app.retrieval.rrf import rrf_fuse
        from app.retrieval import SearchResult

        vector = [
            SearchResult(id=1, content="A", score=0.9, source="vector"),
        ]

        result = rrf_fuse(vector, [], k=60)
        assert len(result) == 1
        assert result[0].id == 1

    def test_empty_vector_returns_bm25_only(self):
        from app.retrieval.rrf import rrf_fuse
        from app.retrieval import SearchResult

        bm25 = [
            SearchResult(id=5, content="E", score=0.9, source="bm25"),
        ]

        result = rrf_fuse([], bm25, k=60)
        assert len(result) == 1
        assert result[0].id == 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestRRF -v
```

Expected: FAIL

- [ ] **Step 3: Write RRF implementation**

```python
"""Reciprocal Rank Fusion — merges vector and BM25 result rankings"""
from __future__ import annotations

from app.retrieval import SearchResult


def rrf_fuse(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Fuse vector and BM25 rankings using Reciprocal Rank Fusion.

    For each unique item: RRF_score = sum(1 / (k + rank_i)) across both lists.
    Items in both lists get natural boost. Items absent from one list get
    rank=inf (0 contribution from that list).

    Args:
        vector_results: Vector search results (ranked by similarity).
        bm25_results: BM25 search results (ranked by score).
        k: Smoothing constant (default 60, per paper).

    Returns:
        Merged list sorted by combined RRF score descending.
    """
    scores: dict[int, tuple[float, SearchResult]] = {}

    for rank, r in enumerate(vector_results):
        score = 1.0 / (k + rank + 1)
        if r.id in scores:
            existing_score, existing_r = scores[r.id]
            existing_r.source = "both"
            scores[r.id] = (existing_score + score, existing_r)
        else:
            scores[r.id] = (score, SearchResult(
                id=r.id, content=r.content, score=0.0,
                metadata=r.metadata, source="vector",
            ))

    for rank, r in enumerate(bm25_results):
        score = 1.0 / (k + rank + 1)
        if r.id in scores:
            existing_score, existing_r = scores[r.id]
            existing_r.source = "both"
            scores[r.id] = (existing_score + score, existing_r)
        else:
            scores[r.id] = (score, SearchResult(
                id=r.id, content=r.content, score=0.0,
                metadata=r.metadata, source="bm25",
            ))

    # Sort by combined RRF score descending, update result score
    sorted_items = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    results: list[SearchResult] = []
    for rrf_score, r in sorted_items:
        r.score = rrf_score
        results.append(r)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestRRF -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/rrf.py tests/unit/test_retrieval.py
git commit -m "feat: add RRF fusion for merging vector and BM25 rankings

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Cross-encoder rerank

**Files:**
- Create: `ai-interview-backend/app/retrieval/rerank.py`
- Append tests to: `ai-interview-backend/tests/unit/test_retrieval.py`

- [ ] **Step 1: Write tests**

Append to `tests/unit/test_retrieval.py`:

```python

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

        async def mock_rerank_call(model, query, documents, top_n, **kwargs):
            class MockResponse:
                class Output:
                    class Result:
                        def __init__(self, index, relevance_score):
                            self.index = index
                            self.relevance_score = relevance_score
                    results = [
                        Result(2, 0.98),  # doc C now #1
                        Result(0, 0.85),  # doc A now #2
                        Result(1, 0.40),  # doc B now #3
                    ]
                output = Output()
            return MockResponse()

        # dashscope.TextReRank is a synchronous call pattern
        # We mock the entire dashscope module's TextReRank
        class MockTextReRank:
            @staticmethod
            def call(**kwargs):
                return MockTextReRank._resp

        MockTextReRank._resp = await mock_rerank_call(None, None, None, None)

        monkeypatch.setattr(
            "app.retrieval.rerank.TextReRank",
            MockTextReRank,
        )

        result = await cross_encoder_rerank(
            query="test query",
            candidates=candidates,
            top_k=3,
        )

        assert len(result) == 3
        # doc C (id=3) should now be first
        assert result[0].id == 3
        assert result[0].content == "doc C"

    async def test_graceful_fallback_on_api_error(self, monkeypatch):
        from app.retrieval.rerank import cross_encoder_rerank
        from app.retrieval import SearchResult

        candidates = [
            SearchResult(id=1, content="only doc", score=0.9, source="both"),
        ]

        class BrokenReRank:
            @staticmethod
            def call(**kwargs):
                raise RuntimeError("API unavailable")

        monkeypatch.setattr(
            "app.retrieval.rerank.TextReRank",
            BrokenReRank,
        )

        result = await cross_encoder_rerank(
            query="test",
            candidates=candidates,
            top_k=1,
        )

        # Falls back to original order
        assert len(result) == 1
        assert result[0].id == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestRerank -v
```

Expected: FAIL

- [ ] **Step 3: Write rerank implementation**

```python
"""Cross-encoder reranking via DashScope qwen3-rerank API"""
from __future__ import annotations

import logging

from dashscope import TextReRank

from app.core.config import settings
from app.retrieval import SearchResult

logger = logging.getLogger(__name__)


async def cross_encoder_rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int,
    model: str | None = None,
) -> list[SearchResult]:
    """Rerank candidates using DashScope cross-encoder model.

    Args:
        query: Original query text.
        candidates: Results from RRF fusion (up to ~20 items).
        top_k: Number of top results to return after reranking.
        model: DashScope rerank model name (defaults to settings).

    Returns:
        Re-ranked list of top_k SearchResults. Falls back to original order
        on API failure.
    """
    if not candidates:
        return []

    model = model or settings.DASHSCOPE_RERANK_MODEL
    documents = [c.content for c in candidates]

    try:
        resp = TextReRank.call(
            model=model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(candidates)),
        )

        reordered: list[SearchResult] = []
        for r in resp.output.results:
            idx = r.index
            if 0 <= idx < len(candidates):
                c = candidates[idx]
                c.score = r.relevance_score
                reordered.append(c)

        # Append any non-reranked results that were skipped
        reranked_indices = {r.index for r in resp.output.results}
        for i, c in enumerate(candidates):
            if i not in reranked_indices:
                reordered.append(c)

        return reordered[:top_k]

    except Exception as e:
        logger.warning(f"Rerank API failed, returning un-reranked results: {e}")
        return candidates[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestRerank -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/rerank.py tests/unit/test_retrieval.py
git commit -m "feat: add DashScope qwen3-rerank cross-encoder with graceful fallback

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Pipeline composer

**Files:**
- Create: `ai-interview-backend/app/retrieval/pipeline.py`
- Append tests to: `ai-interview-backend/tests/unit/test_retrieval.py`

- [ ] **Step 1: Write tests**

Append to `tests/unit/test_retrieval.py`:

```python

@pytest.mark.unit
class TestRetrievalPipeline:
    """Tests for app.retrieval.pipeline.RetrievalPipeline"""

    async def test_pipeline_calls_stages_in_order(self, monkeypatch):
        from app.retrieval.pipeline import RetrievalPipeline
        from app.retrieval.bm25 import BM25Index
        from app.retrieval import SearchResult

        bm25 = BM25Index("test")
        bm25.build(["doc one", "doc two", "doc three"])

        pipeline = RetrievalPipeline(
            client=None,
            collection="knowledge_chunks",
            bm25_index=bm25,
            vector_top_k=3,
            bm25_top_k=3,
            final_top_k=2,
            enable_rerank=True,
        )

        call_order = []

        async def mock_vector(client, query, collection, top_k, filters=None):
            call_order.append("vector")
            return [
                SearchResult(id=1, content="vec result 1", score=0.9, source="vector"),
                SearchResult(id=2, content="vec result 2", score=0.7, source="vector"),
            ]

        monkeypatch.setattr(
            "app.retrieval.pipeline.vector_search", mock_vector
        )

        async def mock_rerank(query, candidates, top_k, model=None):
            call_order.append("rerank")
            return candidates[:top_k]

        monkeypatch.setattr(
            "app.retrieval.pipeline.cross_encoder_rerank", mock_rerank
        )

        results = await pipeline.search(query="test query")

        assert "vector" in call_order
        assert "rerank" in call_order
        assert len(results) == 2

    async def test_pipeline_without_rerank(self, monkeypatch):
        from app.retrieval.pipeline import RetrievalPipeline
        from app.retrieval.bm25 import BM25Index
        from app.retrieval import SearchResult

        bm25 = BM25Index("test")
        bm25.build(["doc one", "doc two"])

        pipeline = RetrievalPipeline(
            client=None,
            collection="knowledge_chunks",
            bm25_index=bm25,
            enable_rerank=False,
            final_top_k=4,
        )

        async def mock_vector(client, query, collection, top_k, filters=None):
            return [SearchResult(id=1, content="vec", score=0.9, source="vector")]

        monkeypatch.setattr(
            "app.retrieval.pipeline.vector_search", mock_vector
        )

        results = await pipeline.search(query="test")

        # With rerank disabled, RRF fused results are returned directly
        assert len(results) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestRetrievalPipeline -v
```

Expected: FAIL

- [ ] **Step 3: Write pipeline implementation**

```python
"""RetrievalPipeline — composes vector + BM25 + RRF + rerank stages"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pymilvus import MilvusClient

from app.core.config import settings
from app.retrieval import SearchResult
from app.retrieval.bm25 import BM25Index
from app.retrieval.rerank import cross_encoder_rerank
from app.retrieval.rrf import rrf_fuse
from app.retrieval.vector import vector_search

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Composable retrieval pipeline.

    Configured per use case (knowledge vs question_bank, defaults vs self-check).
    """

    def __init__(
        self,
        client: MilvusClient,
        collection: str,
        bm25_index: BM25Index,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 10,
        enable_rerank: bool = True,
    ):
        self.client = client
        self.collection = collection
        self.bm25_index = bm25_index
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.enable_rerank = enable_rerank

    async def search(
        self,
        query: str,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Run full pipeline: vector + BM25 → RRF → rerank.

        Args:
            query: Raw query text.
            filters: Collection-specific filters forwarded to vector_search.

        Returns:
            Ranked list of SearchResult.
        """
        # 1. Parallel vector + BM25
        vector_task = vector_search(
            client=self.client,
            query=query,
            collection=self.collection,
            top_k=self.vector_top_k,
            filters=filters,
        )

        # BM25.search is sync — run in thread
        bm25_results_raw = await asyncio.to_thread(
            self.bm25_index.search, query, self.bm25_top_k
        )

        vector_results = await vector_task

        # 2. Convert BM25 (idx, score) to SearchResult
        bm25_results: list[SearchResult] = []
        for idx, score in bm25_results_raw:
            text = self.bm25_index.get_text(idx)
            if text:
                bm25_results.append(SearchResult(
                    id=idx,
                    content=text,
                    score=float(score),
                    source="bm25",
                ))

        # 3. RRF fusion
        fused = rrf_fuse(vector_results, bm25_results, k=settings.RRF_K)

        # 4. Rerank (optional)
        if self.enable_rerank and len(fused) > 1:
            fused = await cross_encoder_rerank(
                query=query,
                candidates=fused,
                top_k=self.final_top_k,
            )
        else:
            fused = fused[:self.final_top_k]

        return fused
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval.py::TestRetrievalPipeline -v
```

Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/pipeline.py tests/unit/test_retrieval.py
git commit -m "feat: add RetrievalPipeline composer with parallel vector+BM25

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Self-check loop YAML prompts

**Files:**
- Create: `ai-interview-backend/app/prompts/retrieval_check_sufficiency.yaml`
- Create: `ai-interview-backend/app/prompts/retrieval_rewrite_query.yaml`

- [ ] **Step 1: Write check_sufficiency prompt**

```yaml
name: retrieval_check_sufficiency
version: 1
description: Evaluate whether retrieved context is sufficient to answer the query
system: |
  你是一个检索质量评估器。判断检索到的文档片段是否足够回答用户的问题。

  充分的标准：
  - 相关片段覆盖了问题的核心概念
  - 片段包含足够的细节给出有意义的回答
  - 不要求所有细节都完美覆盖

  不充分的标准：
  - 片段与问题主题无关
  - 片段太少，信息量不足
  - 片段内容偏题，讨论的是其他话题

  返回纯JSON：
  {
    "sufficient": true,
    "reason": "low_relevance / too_few / off_topic / ok"
  }
temperature: 0.3
response_format: json
user_template: |
  问题：{query}

  检索到的文档片段（共{result_count}条）：
  {retrieved_content}

  请判断这些片段是否足够回答上述问题。
```

- [ ] **Step 2: Write rewrite_query prompt**

```yaml
name: retrieval_rewrite_query
version: 1
description: Rewrite query to improve retrieval when results are insufficient
system: |
  你是一个查询改写器。原始查询检索效果不佳，请改写以提升命中率。

  改写策略：
  - 如果片段偏题：提取问题核心主题，去掉歧义词语
  - 如果片段太少：扩展关键词，加同义词和近义词
  - 如果片段不相关：换成更明确的表述，避免模糊词
  - 可以拆分长问题为短查询，也可以合并多个关键字

  只返回改写后的查询文本（纯文本，不含JSON，不含markdown）。
temperature: 0.7
response_format: text
user_template: |
  原始查询：{query}
  不充分原因：{reason}
  已检索到的内容：
  {retrieved_content}

  请改写查询以获取更好的检索结果。
```

- [ ] **Step 3: Verify prompts load**

```bash
docker exec shilian-app python -c "
from app.llm.prompts import load_prompt
p1 = load_prompt('retrieval_check_sufficiency')
p2 = load_prompt('retrieval_rewrite_query')
print('sufficiency prompt OK:', p1)
print('rewrite prompt OK:', p2)
"
```

Expected: both prompts load without error

- [ ] **Step 4: Commit**

```bash
git add app/prompts/retrieval_check_sufficiency.yaml app/prompts/retrieval_rewrite_query.yaml
git commit -m "feat: add YAML prompts for retrieval self-check loop

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 9: RetrievalCheckState

**Files:**
- Create: `ai-interview-backend/app/workflows/retrieval_check/__init__.py`
- Create: `ai-interview-backend/app/workflows/retrieval_check/state.py`

- [ ] **Step 1: Write __init__.py**

```python
"""Retrieval self-check loop — LangGraph StateGraph

Nodes: retrieve → check_sufficiency → rewrite_query → format_context
Max 2 retry cycles.
"""
```

- [ ] **Step 2: Write state.py**

```python
"""RetrievalCheckState — self-check loop state"""
from __future__ import annotations

from typing import Any

from app.workflows._shared.state_base import BaseWorkflowState


class RetrievalCheckState(BaseWorkflowState, total=False):
    """Self-check loop state for retrieval quality validation.

    Inherits thread_id / retry_count / errors from BaseWorkflowState.
    """

    # ── Input ──
    query: str                         # original query
    current_query: str                 # current rewrite (may differ from original)
    collection: str                    # which collection to search
    pipeline_config: dict              # {final_top_k, enable_rerank, ...}

    # ── Retrieve output ──
    retrieval_results: list[dict]      # latest round results
    retrieval_history: list[dict]      # all results across rounds

    # ── Check output ──
    is_sufficient: bool                # LLM check result
    retry_reason: str                  # "low_relevance" | "too_few" | "off_topic" | "ok"

    # ── Format output ──
    final_context: list[str]           # compiled context strings
    debug_info: dict[str, Any]         # trace data: {rounds, final_query, total_results}
```

- [ ] **Step 3: Verify import**

```bash
docker exec shilian-app python -c "from app.workflows.retrieval_check.state import RetrievalCheckState; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add app/workflows/retrieval_check/
git commit -m "feat: add RetrievalCheckState for self-check loop

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 10: Self-check loop nodes

**Files:**
- Create: `ai-interview-backend/app/workflows/retrieval_check/nodes/__init__.py`
- Create: `ai-interview-backend/app/workflows/retrieval_check/nodes/retrieve.py`
- Create: `ai-interview-backend/app/workflows/retrieval_check/nodes/check_sufficiency.py`
- Create: `ai-interview-backend/app/workflows/retrieval_check/nodes/rewrite_query.py`
- Create: `ai-interview-backend/app/workflows/retrieval_check/nodes/format_context.py`
- Create: `ai-interview-backend/tests/unit/test_retrieval_check_nodes.py`

- [ ] **Step 1: Write nodes __init__.py**

```python
from app.workflows.retrieval_check.nodes.retrieve import retrieve_node
from app.workflows.retrieval_check.nodes.check_sufficiency import check_sufficiency_node
from app.workflows.retrieval_check.nodes.rewrite_query import rewrite_query_node
from app.workflows.retrieval_check.nodes.format_context import format_context_node

__all__ = [
    "retrieve_node",
    "check_sufficiency_node",
    "rewrite_query_node",
    "format_context_node",
]
```

- [ ] **Step 2: Write retrieve node**

```python
"""retrieve node — call RetrievalPipeline.search()"""
from __future__ import annotations

import logging

from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval import SearchResult
from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


async def retrieve_node(state: RetrievalCheckState) -> dict:
    """Run retrieval pipeline with current query.

    Pipeline instance is passed via state.custom to avoid graph serialization issues.
    """
    pipeline: RetrievalPipeline = state.get("custom", {}).get("pipeline")
    if pipeline is None:
        logger.error("No pipeline in state.custom — cannot run retrieval")
        return {"retrieval_results": [], "retrieval_history": []}

    query = state.get("current_query") or state.get("query", "")
    if not query:
        return {"retrieval_results": [], "retrieval_history": []}

    try:
        results: list[SearchResult] = await pipeline.search(
            query=query,
            filters=state.get("custom", {}).get("retrieval_filters"),
        )
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        results = []

    result_dicts = [
        {"id": r.id, "content": r.content, "score": r.score,
         "source": r.source, "metadata": r.metadata}
        for r in results
    ]

    history: list[dict] = list(state.get("retrieval_history", []))
    history.append({
        "round": state.get("retry_count", 0),
        "query": query,
        "results": result_dicts,
    })

    return {
        "retrieval_results": result_dicts,
        "retrieval_history": history,
    }
```

- [ ] **Step 3: Write check_sufficiency node**

```python
"""check_sufficiency node — LLM evaluates retrieval quality"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.llm.client import get_chat_llm
from app.llm.prompts import load_prompt
from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


class SufficiencyResult(BaseModel):
    """Structured output for retrieval sufficiency check."""
    sufficient: bool = Field(description="Whether context is sufficient")
    reason: str = Field(description="low_relevance | too_few | off_topic | ok")


async def check_sufficiency_node(state: RetrievalCheckState) -> dict:
    """Evaluate whether retrieved context adequately covers the query."""
    query = state.get("current_query") or state.get("query", "")
    results = state.get("retrieval_results", [])

    if not results:
        return {
            "is_sufficient": False,
            "retry_reason": "too_few",
        }

    # Format results for LLM
    retrieved_content = "\n\n---\n".join(
        f"[{i + 1}] (score={r.get('score', 0):.2f}) {r.get('content', '')[:500]}"
        for i, r in enumerate(results[:10])
    )

    prompt = load_prompt("retrieval_check_sufficiency")
    llm = get_chat_llm(temperature=0.3).with_structured_output(SufficiencyResult)

    try:
        chain = prompt | llm
        result: SufficiencyResult = await chain.ainvoke({
            "query": query,
            "result_count": str(len(results)),
            "retrieved_content": retrieved_content,
        })
        return {
            "is_sufficient": result.sufficient,
            "retry_reason": result.reason,
        }
    except Exception as e:
        logger.warning(f"Sufficiency check failed: {e}")
        return {
            "is_sufficient": True,  # don't block on LLM error
            "retry_reason": "ok",
        }
```

- [ ] **Step 4: Write rewrite_query node**

```python
"""rewrite_query node — LLM rewrites query to improve retrieval"""
from __future__ import annotations

import logging

from app.llm.client import get_chat_llm
from app.llm.prompts import load_prompt
from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


async def rewrite_query_node(state: RetrievalCheckState) -> dict:
    """Rewrite query based on insufficiency reason and previous results.

    Increments retry_count (from BaseWorkflowState) for cycle tracking.
    """
    original_query = state.get("query", "")
    retry_reason = state.get("retry_reason", "low_relevance")
    history = state.get("retrieval_history", [])

    # Build content summary from all previous rounds
    retrieved_content = ""
    for h in history:
        for r in h.get("results", [])[:5]:
            retrieved_content += f"- {r.get('content', '')[:200]}\n"

    if not retrieved_content:
        retrieved_content = "(无) "

    prompt = load_prompt("retrieval_rewrite_query")
    llm = get_chat_llm(temperature=0.7)

    try:
        chain = prompt | llm
        response = await chain.ainvoke({
            "query": original_query,
            "reason": retry_reason,
            "retrieved_content": retrieved_content,
        })
        rewritten = response.content.strip() if hasattr(response, "content") else str(response).strip()
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}")
        rewritten = original_query  # keep original on failure

    new_retry_count = state.get("retry_count", 0) + 1

    return {
        "current_query": rewritten,
        "retry_count": new_retry_count,
    }
```

- [ ] **Step 5: Write format_context node**

```python
"""format_context node — compile final context from all retrieval rounds"""
from __future__ import annotations

import logging

from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


async def format_context_node(state: RetrievalCheckState) -> dict:
    """Compile deduplicated, relevance-sorted context from all rounds."""
    query = state.get("query", "")
    history = state.get("retrieval_history", [])
    retry_count = state.get("retry_count", 0)

    # Collect all results, deduplicate by content prefix
    seen: set[str] = set()
    all_results: list[dict] = []
    for h in history:
        for r in h.get("results", []):
            key = r.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                all_results.append(r)

    # Sort by score descending
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Compile context strings
    final_context = [r.get("content", "") for r in all_results if r.get("content")]

    # Build debug info
    debug_info = {
        "rounds": [
            {
                "query": h.get("query", ""),
                "result_count": len(h.get("results", [])),
                "top_scores": [r.get("score", 0) for r in h.get("results", [])[:3]],
            }
            for h in history
        ],
        "final_query": query,
        "total_retrieval_rounds": len(history),
        "total_unique_results": len(all_results),
        "retry_count": retry_count,
    }

    return {
        "final_context": final_context,
        "debug_info": debug_info,
    }
```

- [ ] **Step 6: Write node tests**

Create `tests/unit/test_retrieval_check_nodes.py`:

```python
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
            "with_structured_output": lambda self, model: MockChain(),
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
                        {"content": "document A content", "score": 0.85},  # duplicate
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
```

- [ ] **Step 7: Run node tests**

```bash
docker exec shilian-app pytest tests/unit/test_retrieval_check_nodes.py -v
```

Expected: all PASS (4 test classes)

- [ ] **Step 8: Commit**

```bash
git add app/workflows/retrieval_check/nodes/ tests/unit/test_retrieval_check_nodes.py
git commit -m "feat: add self-check loop nodes with unit tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 11: Self-check graph and service

**Files:**
- Create: `ai-interview-backend/app/workflows/retrieval_check/graph.py`
- Create: `ai-interview-backend/app/workflows/retrieval_check/service.py`

- [ ] **Step 1: Write graph.py**

```python
"""retrieval_check StateGraph — self-check loop with max 2 retries"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.graph import END, START, StateGraph

from app.workflows.retrieval_check.nodes import (
    check_sufficiency_node,
    format_context_node,
    retrieve_node,
    rewrite_query_node,
)
from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


def build_retrieval_check_graph() -> StateGraph:
    """Build the retrieval self-check StateGraph.

    Flow:
        START → retrieve → check_sufficiency
                              ├── sufficient → format_context → END
                              └── insufficient → rewrite_query → retrieve
                                                    (max 2 retries, then format_context)
    """
    builder = StateGraph(RetrievalCheckState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("check_sufficiency", check_sufficiency_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("format_context", format_context_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "check_sufficiency")

    builder.add_conditional_edges(
        "check_sufficiency",
        _route_after_check,
        {
            "sufficient": "format_context",
            "insufficient": "rewrite_query",
        },
    )

    builder.add_edge("rewrite_query", "retrieve")  # loop back
    builder.add_edge("format_context", END)

    return builder


def _route_after_check(state: RetrievalCheckState) -> str:
    """Route: sufficient → format, insufficient → rewrite (or format if max retries).

    Max retries controlled by state.custom pipeline_config or default 2.
    """
    if state.get("is_sufficient"):
        return "sufficient"

    max_retries = (state.get("custom") or {}).get("max_retries", 2)
    if state.get("retry_count", 0) >= max_retries:
        logger.info("Max retries reached, proceeding with current results")
        return "sufficient"  # force format_context path

    return "insufficient"
```

- [ ] **Step 2: Write service.py**

```python
"""RetrievalCheckService — wraps self-check graph for interview integration"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph

from app.retrieval.pipeline import RetrievalPipeline
from app.workflows.retrieval_check.graph import build_retrieval_check_graph

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCheckResult:
    """Output from self-check retrieval."""
    final_context: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)


class RetrievalCheckService:
    """Wraps the retrieval self-check graph.

    Usage:
        service = RetrievalCheckService(pipeline, max_retries=2)
        result = await service.check_and_retrieve("Python GIL explained")
        context_strings = result.final_context
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        max_retries: int = 2,
    ):
        self.pipeline = pipeline
        self.max_retries = max_retries
        self._graph: StateGraph | None = None

    def _get_graph(self) -> StateGraph:
        if self._graph is None:
            builder = build_retrieval_check_graph()
            self._graph = builder.compile()
        return self._graph

    async def check_and_retrieve(
        self,
        query: str,
        max_retries: int | None = None,
        retrieval_filters: dict | None = None,
    ) -> RetrievalCheckResult:
        """Run retrieval with self-check loop.

        Args:
            query: The search query (interview question text).
            max_retries: Override default max retries.
            retrieval_filters: Optional filters forwarded to the pipeline.

        Returns:
            RetrievalCheckResult with final_context and debug_info.
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        graph = self._get_graph()

        initial_state = {
            "query": query,
            "current_query": query,
            "retry_count": 0,
            "custom": {
                "pipeline": self.pipeline,
                "max_retries": max_retries,
                "retrieval_filters": retrieval_filters,
            },
        }

        try:
            result = await graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Self-check graph failed: {e}")
            return RetrievalCheckResult(
                final_context=[],
                debug_info={"error": str(e)},
            )

        return RetrievalCheckResult(
            final_context=result.get("final_context", []),
            debug_info=result.get("debug_info", {}),
        )
```

- [ ] **Step 3: Verify import**

```bash
docker exec shilian-app python -c "
from app.workflows.retrieval_check.service import RetrievalCheckService, RetrievalCheckResult
print('Service OK:', RetrievalCheckService)
print('Result OK:', RetrievalCheckResult)
"
```

Expected: prints without error

- [ ] **Step 4: Commit**

```bash
git add app/workflows/retrieval_check/graph.py app/workflows/retrieval_check/service.py
git commit -m "feat: add self-check StateGraph and RetrievalCheckService

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 12: Integrate into interview retrieve_knowledge node

**Files:**
- Modify: `ai-interview-backend/app/workflows/interview/nodes/retrieve_knowledge.py`

- [ ] **Step 1: Rewrite retrieve_knowledge_node**

Replace the entire file content:

```python
"""retrieve_knowledge node — delegates to RetrievalCheckService for hybrid RAG"""
from __future__ import annotations

import logging

from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def retrieve_knowledge_node(state: InterviewState) -> dict:
    """用当前题目检索知识库（hybrid pipeline + self-check loop）。

    Delegates to RetrievalCheckService which runs the full hybrid pipeline
    (vector + BM25 + RRF + rerank) with self-check query rewriting.

    The check service instance is stored in state.custom.retrieval_check_service
    (set up at interview start in InterviewGraphService).
    """
    current_question = state.get("current_question", "")

    if not current_question:
        return {"knowledge_context": []}

    check_service = (state.get("custom") or {}).get("retrieval_check_service")
    if check_service is None:
        logger.warning("RetrievalCheckService 不可用，回退到空 knowledge_context")
        return {"knowledge_context": []}

    try:
        result = await check_service.check_and_retrieve(
            query=current_question,
        )
        return {
            "knowledge_context": result.final_context,
        }
    except Exception as e:
        logger.warning(f"知识库 RAG 检索失败，跳过注入: {e}")
        return {"knowledge_context": []}
```

- [ ] **Step 2: Verify no regressions in existing interview flow**

```bash
docker exec shilian-app python -c "from app.workflows.interview.nodes.retrieve_knowledge import retrieve_knowledge_node; print('import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add app/workflows/interview/nodes/retrieve_knowledge.py
git commit -m "feat: integrate RetrievalCheckService into interview knowledge node

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 13: Wire BM25 index lifecycle and RetrievalCheckService into app startup

**Files:**
- Modify: `ai-interview-backend/app/workflows/interview/service.py` (wire RetrievalCheckService into state.custom)
- Create: `ai-interview-backend/app/retrieval/bm25_lifecycle.py`

The BM25 indices must be built on startup and stored for use by the pipeline. The `RetrievalCheckService` needs to be created with a pipeline + LLM and passed into the interview graph state.

- [ ] **Step 1: Write bm25_lifecycle.py**

```python
"""BM25 index lifecycle — build on startup, provide singletons"""
from __future__ import annotations

import logging

from app.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)

# Module-level singletons, built during startup
knowledge_bm25: BM25Index | None = None
question_bank_bm25: BM25Index | None = None


async def build_bm25_indices(db_session) -> None:
    """Build both BM25 indices from PostgreSQL data.

    Called during FastAPI startup event.
    """
    global knowledge_bm25, question_bank_bm25

    from sqlalchemy import select
    from app.models.knowledge_chunk import KnowledgeChunk
    from app.models.question_bank import QuestionBank

    # Build knowledge index
    result = await db_session.execute(select(KnowledgeChunk.content))
    knowledge_texts = [row[0] for row in result.fetchall() if row[0]]
    knowledge_bm25 = BM25Index("knowledge_chunks")
    knowledge_bm25.build(knowledge_texts)

    # Build question bank index
    result = await db_session.execute(
        select(QuestionBank.question, QuestionBank.reference_answer)
    )
    q_texts = []
    for question, answer in result.fetchall():
        text = f"{question or ''} {answer or ''}".strip()
        if text:
            q_texts.append(text)
    question_bank_bm25 = BM25Index("question_bank")
    question_bank_bm25.build(q_texts)

    logger.info(
        "BM25 indices built: knowledge=%d, question_bank=%d",
        knowledge_bm25.corpus_size,
        question_bank_bm25.corpus_size,
    )


def get_knowledge_bm25() -> BM25Index | None:
    return knowledge_bm25


def get_question_bank_bm25() -> BM25Index | None:
    return question_bank_bm25
```

- [ ] **Step 2: Modify InterviewGraphService to wire RetrievalCheckService**

In `app/workflows/interview/service.py`, modify `submit_answer` to create and pass `RetrievalCheckService`:

Add import at top:
```python
from app.retrieval.bm25_lifecycle import get_knowledge_bm25
from app.retrieval.pipeline import RetrievalPipeline
from app.workflows.retrieval_check.service import RetrievalCheckService
```

In `submit_answer`, before setting `state_data`, add:
```python
        # Build RetrievalCheckService if not already in state
        if is_first_call:
            knowledge_bm25 = get_knowledge_bm25()
            if knowledge_bm25 and milvus_client:
                knowledge_pipeline = RetrievalPipeline(
                    client=milvus_client,
                    collection="knowledge_chunks",
                    bm25_index=knowledge_bm25,
                    vector_top_k=settings.VECTOR_TOP_K,
                    bm25_top_k=settings.BM25_TOP_K,
                    final_top_k=settings.KNOWLEDGE_TOP_K,
                    enable_rerank=True,
                )
                from app.llm.client import get_chat_llm
                llm = get_chat_llm(temperature=0.3)
                retrieval_check_service = RetrievalCheckService(
                    pipeline=knowledge_pipeline,
                    max_retries=settings.SELF_CHECK_MAX_RETRIES,
                )
            else:
                retrieval_check_service = None
        else:
            retrieval_check_service = None
```

Then add `retrieval_check_service` to `state_data`:
```python
        state_data = {
            "answer": answer,
            "stream": stream,
            "custom": {
                "db": db,
                "milvus_client": milvus_client,
                "retrieval_check_service": retrieval_check_service,
            },
        }
```

The full modification is replacing the `state_data` dict and adding the pipeline setup. Show the exact diff:

In `submit_answer`, find:
```python
        state_data = {
            "answer": answer,
            "stream": stream,
            "custom": {"db": db, "milvus_client": milvus_client},
        }
```

Replace with:
```python
        # Wire up RetrievalCheckService for hybrid RAG (Phase 3)
        if is_first_call:
            knowledge_bm25 = get_knowledge_bm25()
            if knowledge_bm25 and milvus_client:
                knowledge_pipeline = RetrievalPipeline(
                    client=milvus_client,
                    collection="knowledge_chunks",
                    bm25_index=knowledge_bm25,
                    vector_top_k=settings.VECTOR_TOP_K,
                    bm25_top_k=settings.BM25_TOP_K,
                    final_top_k=settings.KNOWLEDGE_TOP_K,
                    enable_rerank=True,
                )
                from app.llm.client import get_chat_llm
                llm = get_chat_llm(temperature=0.3)
                retrieval_check_service = RetrievalCheckService(
                    pipeline=knowledge_pipeline,
                    max_retries=settings.SELF_CHECK_MAX_RETRIES,
                )
            else:
                retrieval_check_service = None
        else:
            retrieval_check_service = None

        state_data = {
            "answer": answer,
            "stream": stream,
            "custom": {
                "db": db,
                "milvus_client": milvus_client,
                "retrieval_check_service": retrieval_check_service,
            },
        }
```

- [ ] **Step 3: Commit**

```bash
git add app/retrieval/bm25_lifecycle.py app/workflows/interview/service.py
git commit -m "feat: wire BM25 lifecycle and RetrievalCheckService into interview graph

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 14: Integrate pipeline into interview_service question bank recall

**Files:**
- Modify: `ai-interview-backend/app/services/client/interview_service.py`

Replace the current `_generate_questions_with_rag` method to use `RetrievalPipeline` instead of direct `QuestionBankService.retrieve_questions()`.

- [ ] **Step 1: Write modified _generate_questions_with_rag**

Replace lines 33-115 of `interview_service.py` (the entire `_generate_questions_with_rag` method) with:

```python
    @staticmethod
    async def _generate_questions_with_rag(
        db: AsyncSession,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        total_questions: int,
    ) -> list:
        """
        Phase 3 RAG 出题：使用 RetrievalPipeline（vector + BM25 + RRF + rerank）。

        优先级：hybrid recall → AI select → AI seed → pure AI generate
        """
        from app.retrieval.bm25_lifecycle import get_question_bank_bm25
        from app.retrieval.pipeline import RetrievalPipeline
        from app.vector_db import get_milvus_client

        query = _build_retrieval_query(target_position, parsed_resume)
        recall_k = total_questions * settings.QUESTION_BANK_RECALL_FACTOR

        try:
            bm25_index = get_question_bank_bm25()
            milvus_client = get_milvus_client()

            if bm25_index and milvus_client:
                pipeline = RetrievalPipeline(
                    client=milvus_client,
                    collection="question_bank",
                    bm25_index=bm25_index,
                    vector_top_k=settings.VECTOR_TOP_K,
                    bm25_top_k=settings.BM25_TOP_K,
                    final_top_k=recall_k,
                    enable_rerank=True,
                )
                results = await pipeline.search(
                    query=query,
                    filters={
                        "position_tag": target_position,
                        "difficulty": difficulty,
                        "min_score": settings.QUESTION_BANK_MIN_SCORE,
                    },
                )
                # Convert SearchResult to the dict format expected by AI service
                candidates = []
                for r in results:
                    candidates.append({
                        "id": r.id,
                        "question": r.metadata.get("question", r.content),
                        "reference_answer": r.metadata.get("reference_answer", ""),
                        "key_points": r.metadata.get("key_points", []),
                        "difficulty": r.metadata.get("difficulty", difficulty),
                        "position_tag": r.metadata.get("position_tag", target_position),
                        "similarity": r.score,
                        "source": "from_bank",
                    })

                # Fallback: relax position_tag if insufficient
                if len(candidates) < total_questions:
                    relaxed_results = await pipeline.search(
                        query=query,
                        filters={
                            "difficulty": difficulty,
                            "min_score": settings.QUESTION_BANK_MIN_SCORE,
                        },
                    )
                    seen = {c["id"] for c in candidates}
                    for r in relaxed_results:
                        if r.id not in seen:
                            candidates.append({
                                "id": r.id,
                                "question": r.metadata.get("question", r.content),
                                "reference_answer": r.metadata.get("reference_answer", ""),
                                "key_points": r.metadata.get("key_points", []),
                                "difficulty": r.metadata.get("difficulty", difficulty),
                                "position_tag": r.metadata.get("position_tag", target_position),
                                "similarity": r.score,
                                "source": "from_bank",
                            })
            else:
                # Fallback: BM25 not available, use old vector-only path
                candidates = await QuestionBankService.retrieve_questions(
                    query=query,
                    db=db,
                    k=recall_k,
                    position_tag=target_position,
                    difficulty=difficulty,
                    min_score=settings.QUESTION_BANK_MIN_SCORE,
                )
                if len(candidates) < total_questions:
                    relaxed = await QuestionBankService.retrieve_questions(
                        query=query,
                        db=db,
                        k=recall_k,
                        position_tag=None,
                        difficulty=difficulty,
                        min_score=settings.QUESTION_BANK_MIN_SCORE,
                    )
                    seen = {c["id"] for c in candidates}
                    for c in relaxed:
                        if c["id"] not in seen:
                            candidates.append(c)
        except Exception as e:
            logger.error(f"[RAG出题] Hybrid retrieval failed, falling back to pure AI: {e}")
            candidates = []

        cnt = len(candidates)
        logger.info(f"[RAG出题] Hybrid recall: {cnt} questions, target: {total_questions}")

        if cnt >= total_questions:
            logger.info(f"[RAG出题] 走【题库充分】分支")
            questions = await AIService.select_and_adapt_questions(
                candidates=candidates,
                parsed_resume=parsed_resume,
                target_position=target_position,
                difficulty=difficulty,
                target_n=total_questions,
            )
        elif cnt > 0:
            logger.info(f"[RAG出题] 走【AI 兜底补全】分支（题库 {cnt} 题 + AI 补 {total_questions - cnt} 题）")
            questions = await AIService.generate_with_seeds(
                seed_questions=candidates,
                parsed_resume=parsed_resume,
                target_position=target_position,
                difficulty=difficulty,
                target_n=total_questions,
            )
        else:
            logger.warning(f"[RAG出题] 题库为空，走【纯 AI 生成】兜底分支")
            questions = await AIService.generate_questions(
                parsed_resume=parsed_resume,
                target_position=target_position,
                difficulty=difficulty,
                count=total_questions,
            )
            for q in questions:
                q.setdefault("source", "ai_fallback")
                q.setdefault("bank_id", None)

        return questions
```

- [ ] **Step 2: Commit**

```bash
git add app/services/client/interview_service.py
git commit -m "feat: integrate RetrievalPipeline into question bank recall flow

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 15: Golden set and RAGAS evaluation script

**Files:**
- Create: `ai-interview-backend/eval/__init__.py`
- Create: `ai-interview-backend/eval/golden_set.json`
- Create: `ai-interview-backend/eval/scripts/__init__.py`
- Create: `ai-interview-backend/eval/scripts/eval_ragas.py`

- [ ] **Step 1: Write eval __init__.py**

```python
"""RAG evaluation — golden set + RAGAS offline metrics"""
```

- [ ] **Step 2: Write scripts __init__.py**

```python
"""Evaluation scripts"""
```

- [ ] **Step 3: Write golden_set.json (starter set of 10 entries)**

```json
{
  "version": "1.0",
  "created": "2026-07-10",
  "description": "Phase 3 golden set — hybrid retrieval evaluation. Expand to 20-30 entries in Phase 4.",
  "entries": [
    {
      "id": "golden_001",
      "position_tag": "python_backend",
      "difficulty": "easy",
      "query": "请做一下自我介绍",
      "reference_answer": "介绍个人背景、技术栈、项目经验",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_002",
      "position_tag": "python_backend",
      "difficulty": "medium",
      "query": "Python 的 GIL 是什么？对多线程有什么影响？",
      "reference_answer": "GIL是全局解释器锁，保证同一时刻只有一个线程执行Python字节码。CPU密集型任务受限于GIL，IO密集型可以通过多线程提升性能。多进程可以绕过GIL。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_003",
      "position_tag": "python_backend",
      "difficulty": "medium",
      "query": "Python 装饰器的原理是什么？",
      "reference_answer": "装饰器本质是闭包/高阶函数，接收函数返回新函数，在函数执行前后添加额外逻辑。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_004",
      "position_tag": "python_backend",
      "difficulty": "hard",
      "query": "asyncio 事件循环是如何工作的？",
      "reference_answer": "事件循环是单线程协作式调度，通过epoll/kqueue监控IO就绪事件，使用coroutine和Task管理异步任务。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_005",
      "position_tag": "go_backend",
      "difficulty": "medium",
      "query": "Go 的 goroutine 和 channel 是如何配合的？",
      "reference_answer": "goroutine是轻量级协程，channel用于goroutine间通信。通过select可以多路复用channel。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_006",
      "position_tag": "java_backend",
      "difficulty": "medium",
      "query": "Spring Boot 依赖注入的原理是什么？",
      "reference_answer": "IoC容器管理bean生命周期，通过@Autowired或构造函数注入依赖。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_007",
      "position_tag": "vue_frontend",
      "difficulty": "medium",
      "query": "Vue 3 Composition API 和 Options API 有什么区别？",
      "reference_answer": "Composition API用setup()函数组织逻辑，更灵活可复用；Options API用data/methods/computed等选项组织。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_008",
      "position_tag": "python_backend",
      "difficulty": "medium",
      "query": "数据库索引的 B+ 树结构是如何工作的？",
      "reference_answer": "B+树是多路平衡搜索树，所有数据存在叶子节点，叶子节点通过链表连接支持范围查询。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_009",
      "position_tag": "python_backend",
      "difficulty": "hard",
      "query": "Redis 的持久化策略 RDB 和 AOF 各有什么优缺点？",
      "reference_answer": "RDB是快照持久化，恢复快但可能丢数据；AOF是追加日志，安全性高但文件大恢复慢。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    },
    {
      "id": "golden_010",
      "position_tag": "python_backend",
      "difficulty": "easy",
      "query": "HTTP GET 和 POST 有什么区别？",
      "reference_answer": "GET用于请求数据，参数在URL中，可缓存；POST用于提交数据，参数在请求体中，不可缓存。",
      "relevant_chunk_ids": [],
      "relevant_question_ids": [],
      "relevance_grades": {}
    }
  ]
}
```

- [ ] **Step 4: Write eval_ragas.py**

```python
"""RAGAS 5-metric evaluation for Phase 3 retrieval pipeline.

Usage:
    docker exec shilian-app python eval/scripts/eval_ragas.py

Output:
    eval/reports/baseline-YYYYMMDD.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.llm.client import get_chat_llm
from app.llm.embedding import embed_text
from app.vector_db import get_milvus_client
from app.vector_db.collections import knowledge as knowledge_vdb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent.parent
REPORTS_DIR = EVAL_DIR / "reports"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.json"


def load_golden_set(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_evaluation():
    golden = load_golden_set(GOLDEN_SET_PATH)
    entries = golden.get("entries", [])
    logger.info(f"Loaded {len(entries)} golden entries")

    client = get_milvus_client()
    llm = get_chat_llm(temperature=0.0)
    samples = []

    for entry in entries:
        query = entry["query"]
        reference = entry["reference_answer"]

        # Retrieve context using current pipeline (vector only for baseline)
        try:
            query_vec = await embed_text(query)
            chunks = knowledge_vdb.search(
                client=client,
                query_vector=query_vec,
                top_k=4,
            )
            contexts = [c.get("content", "") for c in chunks if c.get("content")]
        except Exception as e:
            logger.warning(f"Retrieval failed for {entry['id']}: {e}")
            contexts = []

        if not contexts:
            logger.warning(f"No contexts retrieved for {entry['id']}, skipping")
            continue

        # Generate answer using LLM with retrieved context
        ctx_text = "\n\n---\n".join(contexts[:4])
        answer_prompt = (
            f"基于以下参考资料回答问题：\n\n"
            f"参考资料：\n{ctx_text}\n\n"
            f"问题：{query}\n\n"
            f"请用中文回答。"
        )
        try:
            response = await llm.ainvoke(answer_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.warning(f"LLM generation failed for {entry['id']}: {e}")
            answer = ""

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference,
        )
        samples.append(sample)

    if not samples:
        logger.error("No valid samples for evaluation")
        return

    logger.info(f"Running RAGAS evaluation on {len(samples)} samples...")

    result = evaluate(
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
        dataset=samples,
    )

    # Build report
    report = {
        "date": datetime.now().isoformat(),
        "golden_set_version": golden.get("version"),
        "num_samples": len(samples),
        "metrics": {},
    }
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision",
                         "context_recall", "answer_correctness"]:
        score = result.get(metric_name, None)
        if score is not None:
            report["metrics"][metric_name] = float(score) if hasattr(score, "__float__") else score

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    report_path = REPORTS_DIR / f"baseline-{date_str}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Metrics: {json.dumps(report['metrics'], indent=2)}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_evaluation())
```

- [ ] **Step 5: Commit**

```bash
git add eval/
git commit -m "feat: add golden set and RAGAS 5-metric evaluation script

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 16: Final integration test and end-to-end smoke test

**Files:** None (run existing tests)

- [ ] **Step 1: Run all unit tests**

```bash
docker exec shilian-app pytest tests/unit/ -v
```

Expected: all unit tests pass (existing + new retrieval + new self-check nodes)

- [ ] **Step 2: Run retrieval pipeline smoke test**

```bash
docker exec shilian-app python -c "
import asyncio
from app.retrieval.bm25 import BM25Index
from app.retrieval.rrf import rrf_fuse
from app.retrieval import SearchResult

# Quick integration: BM25 + RRF
async def main():
    idx = BM25Index('smoke_test')
    idx.build(['Python async programming', 'Java concurrency', 'Django web framework'])
    results = idx.search('python web', top_k=2)
    print('BM25 results:', results)

    vec = [SearchResult(id=1, content='Django', score=0.9, source='vector')]
    bm = [SearchResult(id=1, content='Django', score=0.8, source='bm25')]
    fused = rrf_fuse(vec, bm)
    print('RRF fused:', fused)
    print('Smoke test PASSED')

asyncio.run(main())
"
```

Expected: prints "Smoke test PASSED"

- [ ] **Step 3: Run existing smoke tests**

```bash
docker exec shilian-app pytest -m "smoke" -v
```

Expected: existing smoke tests still pass

- [ ] **Step 4: Commit if any changes from test fixes**

```bash
git status
# If no changes: echo "All tests passing, no changes needed"
```

---

## Implementation Summary

**New files:** 20
**Modified files:** 4 (requirements.txt, config.py, retrieve_knowledge.py, interview_service.py)
**New dependencies:** rank-bm25, ragas
**Test files:** 2 new (test_retrieval.py, test_retrieval_check_nodes.py)
