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
