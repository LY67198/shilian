"""retrieve node — call RetrievalPipeline.search()"""
from __future__ import annotations

import logging

from app.retrieval import SearchResult
from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


async def retrieve_node(state: RetrievalCheckState) -> dict:
    """Run retrieval pipeline with current query.

    Pipeline instance is passed via state.custom to avoid graph serialization issues.
    """
    custom = state.get("custom") or {}
    pipeline = custom.get("pipeline")
    if pipeline is None:
        logger.error("No pipeline in state.custom — cannot run retrieval")
        return {"retrieval_results": [], "retrieval_history": []}

    query = state.get("current_query") or state.get("query", "")
    if not query:
        return {"retrieval_results": [], "retrieval_history": []}

    try:
        results: list[SearchResult] = await pipeline.search(
            query=query,
            filters=custom.get("retrieval_filters"),
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
