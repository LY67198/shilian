"""check_sufficiency node — LLM evaluates retrieval quality"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from app.llm.client import get_chat_llm
from app.llm.prompts import load_prompt
from app.workflows.retrieval_check.state import RetrievalCheckState

logger = logging.getLogger(__name__)


class SufficiencyResult(BaseModel):
    """Structured output for retrieval sufficiency check."""
    sufficient: bool = Field(description="Whether context is sufficient")
    reason: str = Field(description="low_relevance | too_few | off_topic | ok")


async def check_sufficiency_node(state: RetrievalCheckState) -> dict:
    """Evaluate whether retrieved context adequately covers the query."""
    query = state.get("current_query") or state.get("query", "")
    results = state.get("retrieval_results", [])

    if not results:
        return {
            "is_sufficient": False,
            "retry_reason": "too_few",
        }

    # Format results for LLM
    retrieved_content = "\n\n---\n".join(
        f"[{i + 1}] (score={r.get('score', 0):.2f}) {r.get('content', '')[:500]}"
        for i, r in enumerate(results[:10])
    )

    prompt = load_prompt("retrieval_check_sufficiency")
    llm = get_chat_llm(temperature=0.3).with_structured_output(SufficiencyResult)

    try:
        chain = prompt | llm
        result: SufficiencyResult = await chain.ainvoke({
            "query": query,
            "result_count": str(len(results)),
            "retrieved_content": retrieved_content,
        })
        return {
            "is_sufficient": result.sufficient,
            "retry_reason": result.reason,
        }
    except Exception as e:
        logger.warning(f"Sufficiency check failed: {e}")
        return {
            "is_sufficient": True,  # don't block on LLM error
            "retry_reason": "ok",
        }
