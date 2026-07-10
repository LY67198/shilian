"""Reciprocal Rank Fusion (RRF) for merging vector and BM25 result lists.

Reference: Cormack, Clarke, Buettcher (SIGIR 2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual rank learning methods."
"""
from __future__ import annotations

from app.retrieval import SearchResult


def rrf_fuse(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Merge two ranked result lists with Reciprocal Rank Fusion.

    For each unique item id, RRF_score = sum(1 / (k + rank_i)) across both
    lists (1-indexed rank).  Items present in both lists naturally receive a
    higher combined score and are assigned source="both".

    Args:
        vector_results: Ranked list from vector search (best first).
        bm25_results:   Ranked list from BM25 lexical search (best first).
        k:              RRF constant (default 60, per original paper).

    Returns:
        Single merged list sorted by combined RRF score descending.
    """
    rrf_scores: dict[int, float] = {}
    content_map: dict[int, str] = {}
    sources: dict[int, list[str]] = {}

    for rank, item in enumerate(vector_results, start=1):
        score_contribution = 1.0 / (k + rank)
        rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + score_contribution
        content_map.setdefault(item.id, item.content)
        sources.setdefault(item.id, []).append("vector")

    for rank, item in enumerate(bm25_results, start=1):
        score_contribution = 1.0 / (k + rank)
        rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + score_contribution
        content_map.setdefault(item.id, item.content)
        sources.setdefault(item.id, []).append("bm25")

    merged: list[SearchResult] = []
    for item_id in rrf_scores:
        source_set = set(sources.get(item_id, []))
        label = "both" if source_set == {"vector", "bm25"} else next(iter(source_set))
        merged.append(SearchResult(
            id=item_id,
            content=content_map.get(item_id, ""),
            score=rrf_scores[item_id],
            source=label,
        ))

    merged.sort(key=lambda r: r.score, reverse=True)
    return merged
