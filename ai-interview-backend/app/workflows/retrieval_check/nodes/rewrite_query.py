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
        retrieved_content = "(无)"

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
