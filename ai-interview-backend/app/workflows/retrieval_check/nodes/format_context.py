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
