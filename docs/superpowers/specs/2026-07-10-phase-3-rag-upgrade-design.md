# Phase 3 RAG Pipeline Upgrade — Design Spec

**Date:** 2026-07-10
**Status:** Approved
**Scope:** Hybrid retrieval (vector + BM25 + RRF + rerank) + self-check loop + RAGAS evaluation framework

## Motivation

Current retrieval is pure Milvus vector search (HNSW + L2). No keyword search, no hybrid fusion, no re-ranking, no retrieval quality checks, and no evaluation framework. Phase 3 upgrades the full retrieval pipeline with BM25 keyword search, RRF fusion, cross-encoder reranking, a LangGraph self-check loop for query rewriting, and an offline RAGAS evaluation framework with a golden set.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BM25 implementation | `rank-bm25` Python library | No new infra dependency; in-process, fast rebuild |
| Tokenization | Character bigrams (fallback: jieba) | No dependency for v1; jieba if bigrams underperform |
| Rerank model | DashScope `gte-rerank` API | Already use DashScope for embedding; no local model overhead |
| Golden set size | 20-30 entries initially | Validate pipeline first, expand in Phase 4 |
| Self-check strategy | Deep self-check: retrieve → LLM sufficiency → rewrite → re-retrieve | Traceable decisions per round; 2 retries max |
| RAGAS metrics | 5 metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness | Standard 4 + correctness for answer quality |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERVIEW WORKFLOW                       │
│  fetch_context ──→ [retrieve_knowledge] ──→ ask_question   │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 RETRIEVAL PIPELINE                          │
│                                                             │
│  query ──→ ┌──────────┐  ┌──────┐  ┌──────┐  ┌──────────┐ │
│            │ Vector    │  │ BM25 │  │ RRF  │  │ qwen3-   │ │
│            │ (Milvus)  │  │(rank-│  │Fusion│  │ rerank   │ │
│            │           │  │ bm25)│  │      │  │(DashScope│ │
│            └────┬──────┘  └──┬───┘  └──┬───┘  └────┬─────┘ │
│                 │            │         │            │       │
│                 ▼            ▼         ▼            ▼       │
│              top_k x2    top_k x2   merged     re-ranked     │
│               vectors     keywords   by RRF    top_k results │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               SELF-CHECK LOOP (LangGraph)                   │
│                                                             │
│  results ──→ [check_sufficiency] ──→ sufficient?            │
│                   │                      │                  │
│                   │ insufficient         yes                │
│                   ▼                      │                  │
│              [rewrite_query]             │                  │
│                   │                      │                  │
│                   └──→ back to pipeline  │                  │
│                        (max 2 retries)   │                  │
└──────────────────────────────────────────┼──────────────────┘
                                           ▼
                                   final context
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION (offline)                      │
│                                                             │
│  golden_set.json ──→ eval_ragas.py ──→ 5 RAGAS metrics     │
└─────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. `app/retrieval/` — Retrieval Pipeline Module

New module with composable stages. Each stage has a single responsibility and clear I/O contract.

#### 1.1 `vector.py` — Vector Recall

```python
async def vector_search(
    query: str,
    collection: str,           # "knowledge_chunks" | "question_bank"
    top_k: int = 20,
    filters: dict | None = None,
) -> list[SearchResult]:
```

Thin wrapper over existing `collections/*.py`. Embeds query via `app.llm.embedding`, calls `collection.search()`, returns normalized `SearchResult` list.

#### 1.2 `bm25.py` — BM25 Keyword Recall

```python
class BM25Index:
    def __init__(self, collection_name: str): ...
    def build(self, texts: list[str]) -> None: ...
    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]: ...
    def _tokenize(self, text: str) -> list[str]: ...  # character bigrams
```

Two instances: `knowledge_index` and `question_bank_index`. Built on startup from PG texts, incrementally updated on CRUD. Rebuild is O(N) and <50ms for hundreds of docs.

**Lifecycle:**
- **Startup:** Load all texts from PG, build index, store in app state
- **Create:** Append to corpus → rebuild
- **Update:** Replace in corpus → rebuild
- **Delete:** Remove from corpus → rebuild

**Tokenization:** Character bigrams (no dependency). If golden set evaluation shows poor recall, switch to jieba.

#### 1.3 `rrf.py` — Reciprocal Rank Fusion

```python
def rrf_fuse(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
```

For each unique item: `RRF_score = sum(1 / (k + rank_i))` across both lists. Items present in both lists get natural boost. Returns merged list sorted by combined score.

#### 1.4 `rerank.py` — Cross-Encoder Rerank

```python
async def cross_encoder_rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int,
    model: str = "gte-rerank",
) -> list[SearchResult]:
```

Calls DashScope `TextReRank` API. Input: ~20 candidates from RRF. Output: top_k re-ranked by cross-encoder relevance. Preserves all metadata through the rerank step.

#### 1.5 `pipeline.py` — Pipeline Composer

```python
class RetrievalPipeline:
    def __init__(
        self,
        collection: str,
        bm25_index: BM25Index,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 10,
        enable_rerank: bool = True,
    ): ...

    async def search(
        self, query: str, filters: dict | None = None
    ) -> list[SearchResult]:
        # 1. Parallel: vector_search() + bm25_index.search()
        # 2. rrf_fuse(vector_results, bm25_results)
        # 3. cross_encoder_rerank(query, fused, final_top_k) if enable_rerank
        # 4. Return final results
```

Configured per use case:

| Use Case | collection | final_top_k | enable_rerank |
|----------|-----------|-------------|---------------|
| Knowledge (default) | knowledge_chunks | 4 | True |
| Knowledge (self-check) | knowledge_chunks | 10 | False |
| Question bank (default) | question_bank | 20 | True |

### 2. `app/workflows/retrieval_check/` — Self-Check Loop

LangGraph StateGraph that wraps the retrieval pipeline with LLM-driven quality checks and query rewriting.

#### 2.1 State

```python
class RetrievalCheckState(GraphState):
    query: str                          # original query
    current_query: str                   # current rewrite
    retrieval_results: list[dict]        # latest results
    retrieval_history: list[dict]        # all results across rounds
    is_sufficient: bool                  # LLM check result
    retry_count: int                     # 0, 1, 2
    retry_reason: str                    # "low_relevance" | "too_few" | "off_topic"
    final_context: str                   # compiled context string
    debug_info: dict                     # trace data for LangSmith
```

#### 2.2 Graph

```
START → retrieve → check_sufficiency → [conditional]
                                         ├── sufficient → format_context → END
                                         └── insufficient → rewrite_query → retrieve (if retries < 2)
                                                                           └── format_context → END (if retries >= 2)
```

#### 2.3 Nodes

| Node | Responsibility |
|------|---------------|
| `retrieve` | Call `RetrievalPipeline.search()` with `current_query`, `enable_rerank=False`, `final_top_k=10`. Append to `retrieval_history`. |
| `check_sufficiency` | LLM with structured output: `{sufficient: bool, reason: str}`. Evaluates if retrieved context adequately covers the query. |
| `rewrite_query` | LLM rewrites query based on `retry_reason`: expand keywords, add synonyms, split multi-part queries. Increments `retry_count`. |
| `format_context` | Compile final context from best results across all rounds. Deduplicate, sort by relevance. Attach `debug_info`. |

#### 2.4 Service

```python
class RetrievalCheckService:
    def __init__(self, pipeline: RetrievalPipeline, llm): ...
    async def check_and_retrieve(
        self, query: str, max_retries: int = 2
    ) -> RetrievalCheckResult: ...
```

#### 2.5 Integration

The interview graph's `retrieve_knowledge_node` is modified to delegate to `RetrievalCheckService` instead of calling `knowledge_vdb.search()` directly. The old direct-Milvus path is removed.

### 3. Evaluation Framework

#### 3.1 Golden Set (`eval/golden_set.json`)

20-30 manually annotated entries covering all 8 position templates. Format:

```json
{
  "version": "1.0",
  "created": "2026-07-10",
  "entries": [
    {
      "id": "golden_001",
      "position_tag": "python_backend",
      "difficulty": "medium",
      "query": "Python memory management and garbage collection",
      "reference_answer": "Python uses reference counting + mark-and-sweep...",
      "relevant_chunk_ids": [12, 45, 78],
      "relevant_question_ids": [3, 7],
      "relevance_grades": {"12": 3, "45": 2, "78": 2}
    }
  ]
}
```

#### 3.2 RAGAS Script (`eval/scripts/eval_ragas.py`)

Runs 5 metrics: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `answer_correctness`.

For each golden entry:
1. Run retrieval pipeline
2. Generate answer via LLM with retrieved context
3. Build RAGAS `SingleTurnSample`
4. Evaluate all metrics
5. Save report to `eval/reports/baseline-{date}.json`

Run via: `docker exec shilian-app python eval/scripts/eval_ragas.py`

### 4. Configuration

New settings in `core/config.py`:

```python
DASHSCOPE_RERANK_MODEL: str = "gte-rerank"
RERANK_TOP_K: int = 10
BM25_TOP_K: int = 20
VECTOR_TOP_K: int = 20
RRF_K: int = 60
SELF_CHECK_MAX_RETRIES: int = 2
```

## File Manifest

| File | Type | Purpose |
|------|------|---------|
| `app/retrieval/__init__.py` | new | Module init |
| `app/retrieval/vector.py` | new | Milvus vector recall wrapper |
| `app/retrieval/bm25.py` | new | rank-bm25 keyword index + search |
| `app/retrieval/rrf.py` | new | RRF fusion (k=60) |
| `app/retrieval/rerank.py` | new | DashScope qwen3-rerank |
| `app/retrieval/pipeline.py` | new | Pipeline composer |
| `app/workflows/retrieval_check/__init__.py` | new | Module init |
| `app/workflows/retrieval_check/state.py` | new | RetrievalCheckState |
| `app/workflows/retrieval_check/graph.py` | new | build_retrieval_check_graph() |
| `app/workflows/retrieval_check/service.py` | new | RetrievalCheckService |
| `app/workflows/retrieval_check/nodes/__init__.py` | new | Nodes init |
| `app/workflows/retrieval_check/nodes/retrieve.py` | new | Pipeline search node |
| `app/workflows/retrieval_check/nodes/check_sufficiency.py` | new | LLM sufficiency check |
| `app/workflows/retrieval_check/nodes/rewrite_query.py` | new | LLM query rewrite |
| `app/workflows/retrieval_check/nodes/format_context.py` | new | Context compiler |
| `eval/golden_set.json` | new | 20-30 annotated golden queries |
| `eval/scripts/eval_ragas.py` | new | RAGAS 5-metric evaluation script |
| `app/workflows/interview/nodes/retrieve_knowledge.py` | modified | Delegate to RetrievalCheckService |
| `app/services/client/interview_service.py` | modified | Use RetrievalPipeline for question bank recall |
| `app/core/config.py` | modified | Add rerank/RRF/BM25 settings |

## Error Handling

- **BM25 index empty:** If corpus is empty (no knowledge/question data yet), BM25 returns empty results. Pipeline degrades gracefully — RRF with only vector results = vector-only ranking.
- **DashScope rerank API failure:** Log warning, return RRF-fused results without reranking. The `final_top_k` take from the fused list instead.
- **Self-check LLM failure:** If `check_sufficiency` or `rewrite_query` fails, use a fallback reason ("llm_error") and proceed with current results (don't block the interview).
- **Milvus unavailable:** Fail fast — the interview cannot proceed without retrieval. This matches current behavior.
- **Golden set evaluation with missing chunks:** If a `relevant_chunk_id` no longer exists in PG/Milvus, skip that entry in evaluation with a warning.

## Testing

- **Unit tests** for each `app/retrieval/` component:
  - `vector.py`: mock Milvus client, verify filter passthrough
  - `bm25.py`: known corpus → verify BM25 score ordering
  - `rrf.py`: known rank inputs → verify expected RRF output
  - `rerank.py`: mock DashScope response → verify re-ranking
  - `pipeline.py`: mock stages → verify composition order
- **Unit tests** for `retrieval_check/` nodes:
  - Each node as pure function, mock LLM + pipeline
  - Verify graph: sufficient → format_context; insufficient → rewrite → retrieve
  - Verify max retries limit (0, 1, 2, then forced exit)
- **Smoke test** for `eval_ragas.py` with a minimal golden set (3 entries)
- **Integration test** for full pipeline: real Milvus + real BM25 index → verify RRF + rerank output

## Out of Scope (Phase 4+)

- Parent-child chunk assembly (deferred — current flat chunking is sufficient)
- BM25 using jieba (deferred until golden set evaluation shows bigram weakness)
- Continuous RAGAS evaluation in CI (golden set expansion deferred to Phase 4)
- Repository completion for user/admin/question_bank/knowledge (Phase 4)
