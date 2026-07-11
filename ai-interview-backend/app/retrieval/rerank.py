"""DashScope qwen3-rerank cross-encoder reranking with graceful fallback."""
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
    """Rerank candidates using a cross-encoder model via DashScope TextReRank.

    Args:
        query: The search query string.
        candidates: List of SearchResult items to rerank.
        top_k: Number of top results to return after reranking.
        model: DashScope rerank model name. Defaults to settings.DASHSCOPE_RERANK_MODEL.

    Returns:
        Reranked list of SearchResult (length <= top_k), preserving original
        metadata.  Falls back to candidates[:top_k] on any API error.
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
            top_n=top_k,
        )
    except Exception:
        logger.warning(
            "DashScope rerank API failed, falling back to original order",
            exc_info=True,
        )
        return candidates[:top_k]

    # Build index lookup into original candidates
    reranked: list[SearchResult] = []
    for result in resp.output.results:
        idx = result.index
        if 0 <= idx < len(candidates):
            original = candidates[idx]
            reranked.append(SearchResult(
                id=original.id,
                content=original.content,
                score=result.relevance_score,
                metadata=dict(original.metadata),
                source=original.source,
            ))

    if not reranked:
        logger.warning("Rerank API returned empty results, falling back to original order")
        return candidates[:top_k]

    return reranked
