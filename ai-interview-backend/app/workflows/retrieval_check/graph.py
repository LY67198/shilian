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
