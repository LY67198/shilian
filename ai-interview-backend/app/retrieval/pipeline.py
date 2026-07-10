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
        # 1. Parallel: vector_search() + bm25_index.search() via asyncio.to_thread
        vector_task = vector_search(
            client=self._client,
            query=query,
            collection=self._collection,
            top_k=self._vector_top_k,
            filters=filters,
        )
        bm25_task = asyncio.to_thread(
            self._bm25.search, query, self._bm25_top_k
        )

        vector_results, bm25_raw = await asyncio.gather(vector_task, bm25_task)

        # 2. Convert BM25 (idx, score) tuples to SearchResult using bm25_index.get_text()
        bm25_results = [
            SearchResult(
                id=idx,
                content=self._bm25.get_text(idx),
                score=score,
                source="bm25",
            )
            for idx, score in bm25_raw
        ]

        # 3. rrf_fuse(vector_results, bm25_results, k=settings.RRF_K)
        fused = rrf_fuse(vector_results, bm25_results, k=settings.RRF_K)

        # 4. cross_encoder_rerank(query, fused, final_top_k) if enable_rerank
        #    Skip rerank if enable_rerank=False or len(fused) <= 1
        if self._enable_rerank and len(fused) > 1:
            return await cross_encoder_rerank(query, fused, self._final_top_k)

        return fused[:self._final_top_k]
