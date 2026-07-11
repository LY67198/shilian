"""retrieve_knowledge node — delegates to RetrievalCheckService for hybrid RAG"""
from __future__ import annotations

import logging

from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def retrieve_knowledge_node(state: InterviewState) -> dict:
    """用当前题目检索知识库（hybrid pipeline + self-check loop）。

    Delegates to RetrievalCheckService which runs the full hybrid pipeline
    (vector + BM25 + RRF + rerank) with self-check query rewriting.

    The check service instance is stored in state.custom.retrieval_check_service
    (set up at interview start in InterviewGraphService).
    """
    current_question = state.get("current_question", "")

    if not current_question:
        return {"knowledge_context": []}

    check_service = (state.get("custom") or {}).get("retrieval_check_service")
    if check_service is None:
        logger.warning("RetrievalCheckService 不可用，回退到空 knowledge_context")
        return {"knowledge_context": []}

    try:
        result = await check_service.check_and_retrieve(
            query=current_question,
        )
        return {
            "knowledge_context": result.final_context,
        }
    except Exception as e:
        logger.warning(f"知识库 RAG 检索失败，跳过注入: {e}")
        return {"knowledge_context": []}
