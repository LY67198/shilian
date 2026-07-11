"""RetrievalPipeline — composes vector + BM25 + RRF + rerank into a single call."""
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
from app.workflows._shared.tracing import trace_span

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Hybrid retrieval pipeline: parallel vector + BM25, RRF fusion, cross-encoder rerank.

    Usage::

        pipeline = RetrievalPipeline(client, "knowledge_chunks", bm25_index)
        results = await pipeline.search("query text")
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
    ) -> None:
        """初始化混合检索管线。

        Args:
            client: MilvusClient 实例，用于向量检索。
            collection: Milvus collection 名称（"knowledge_chunks" 或 "question_bank"）。
            bm25_index: BM25 关键词索引实例。
            vector_top_k: 向量检索召回数量。
            bm25_top_k: BM25 检索召回数量。
            final_top_k: 融合后最终返回的结果数量。
            enable_rerank: 是否启用重排序。
        """
        self._client = client
        self._collection = collection
        self._bm25 = bm25_index
        self._vector_top_k = vector_top_k
        self._bm25_top_k = bm25_top_k
        self._final_top_k = final_top_k
        self._enable_rerank = enable_rerank

    async def search(
        self, query: str, filters: dict | None = None
    ) -> list[SearchResult]:
        """Execute hybrid search: parallel vector + BM25, fuse, optionally rerank.

        Args:
            query: Raw search query string.
            filters: Optional collection-specific filter dict passed to vector_search.

        Returns:
            Ranked list of SearchResult (length <= final_top_k).
        """
        # 1. Vector recall (Milvus HNSW)
        with trace_span("vector_recall", {"query": query, "top_k": self._vector_top_k}) as span:
            vector_results = await vector_search(
                client=self._client,
                query=query,
                collection=self._collection,
                top_k=self._vector_top_k,
                filters=filters,
            )
            if span:
                span.add_outputs({"count": len(vector_results)})

        # 2. BM25 keyword search
        with trace_span("bm25_search", {"query": query, "top_k": self._bm25_top_k}) as span:
            bm25_raw = await asyncio.to_thread(
                self._bm25.search, query, self._bm25_top_k
            )
            bm25_results = [
                SearchResult(
                    id=idx,
                    content=self._bm25.get_text(idx),
                    score=score,
                    source="bm25",
                )
                for idx, score in bm25_raw
            ]
            if span:
                span.add_outputs({"count": len(bm25_results)})

        # 3. RRF fusion
        with trace_span("rrf_fusion", {"k": settings.RRF_K}) as span:
            fused = rrf_fuse(vector_results, bm25_results, k=settings.RRF_K)
            if span:
                span.add_outputs({"count": len(fused)})

        # 4. Rerank (DashScope gte-rerank)
        if self._enable_rerank and len(fused) > 1:
            with trace_span("rerank", {"final_top_k": self._final_top_k}) as span:
                results = await cross_encoder_rerank(query, fused, self._final_top_k)
                if span:
                    span.add_outputs({"count": len(results)})
                return results

        return fused[:self._final_top_k]
