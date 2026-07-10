"""retrieve_knowledge node — Milvus 知识库 RAG 检索"""
from __future__ import annotations

import logging

from app.llm.embedding import embed_text
from app.vector_db.collections import knowledge as knowledge_vdb
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def retrieve_knowledge_node(state: InterviewState) -> dict:
    """用当前题目检索知识库，返回相关片段作为评分依据"""
    current_question = state.get("current_question", "")
    knowledge_context: list[str] = []

    if not current_question:
        return {"knowledge_context": knowledge_context}

    custom = state.get("custom") or {}
    milvus_client = custom.get("milvus_client")

    if not milvus_client:
        logger.warning("Milvus client 不可用，跳过知识库检索")
        return {"knowledge_context": knowledge_context}

    try:
        query_vec = await embed_text(current_question)
        chunks = knowledge_vdb.search(
            client=milvus_client,
            query_vector=query_vec,
            top_k=3,
        )
        knowledge_context = [c.get("content", "") for c in chunks if c.get("content")]
    except Exception as e:
        logger.warning(f"知识库 RAG 检索失败，跳过注入: {e}")

    return {"knowledge_context": knowledge_context}
