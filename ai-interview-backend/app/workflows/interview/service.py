"""面试 Graph Service — submit_answer 单一入口（async generator）"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.evaluator_agent import EvaluatorAgent
from app.agents.report_agent import ReportAgent
from app.core.config import settings
from app.models.interview_message import InterviewMessage
from app.retrieval.bm25_lifecycle import get_knowledge_bm25
from app.retrieval.pipeline import RetrievalPipeline
from app.vector_db import get_milvus_client
from app.workflows._shared.sse import astream_to_sse
from app.workflows.interview.graph import get_compiled_graph
from app.workflows.retrieval_check.service import RetrievalCheckService

logger = logging.getLogger(__name__)


async def submit_answer(
    db: AsyncSession,
    user_id: int,
    interview_id: int,
    answer: str,
    stream: bool = False,
) -> AsyncIterator[dict | str]:
    """提交面试回答

    Args:
        db: 数据库会话
        user_id: 用户 ID
        interview_id: 面试 ID
        answer: 候选人回答文本
        stream: 是否流式推送

    Yields:
        stream=False: 单个 dict (score, feedback, is_finished, next_question, report)
        stream=True: 多个 SSE 事件字符串
    """
    candidate_msg = InterviewMessage(
        interview_id=interview_id,
        role="candidate",
        content=answer,
        question_index=-1,
    )
    db.add(candidate_msg)
    await db.commit()

    graph = await get_compiled_graph()
    milvus_client = get_milvus_client()
    config = {"configurable": {"thread_id": f"interview-{interview_id}"}}

    checkpointer = graph.checkpointer
    checkpoint = await checkpointer.aget(config)
    is_first_call = checkpoint is None

    state_data = {
        "answer": answer,
        "stream": stream,
        "custom": {
            "db": db,
            "milvus_client": milvus_client,
            "evaluator_agent": EvaluatorAgent(),
            "report_agent": ReportAgent(),
            "retrieval_check_service": _build_retrieval_check_service(
                milvus_client, is_first_call
            ),
        },
    }

    if is_first_call:
        initial_state = {
            "interview_id": interview_id,
            "user_id": user_id,
            **state_data,
        }
        if stream:
            async for sse_event in astream_to_sse(
                graph.astream_events(initial_state, config, version="v2")
            ):
                yield sse_event
        else:
            result = await graph.ainvoke(initial_state, config)
            yield _extract_response(result)
    else:
        if stream:
            async for sse_event in astream_to_sse(
                graph.astream_events(Command(resume=state_data), config, version="v2")
            ):
                yield sse_event
        else:
            result = await graph.ainvoke(Command(resume=state_data), config)
            yield _extract_response(result)


def _extract_response(state: dict) -> dict:
    """从 graph 最终 state 提取 API 响应"""
    is_finished = state.get("is_finished", False)
    response = {
        "score": state.get("score", 5.0),
        "feedback": state.get("feedback", ""),
        "question_index": state.get("current_index", 0),
        "is_finished": is_finished,
        "next_question": None,
    }
    if is_finished:
        response["overall_score"] = state.get("overall_score", 0)
        response["report"] = state.get("report", {})
    else:
        response["next_question"] = state.get("next_question")
    return response


def _build_retrieval_check_service(milvus_client, is_first_call: bool):
    """Wire up RetrievalCheckService for hybrid RAG (Phase 3).

    Only builds the service on the first call to avoid re-initializing
    the pipeline on every HITL resume round.
    """
    if not is_first_call:
        return None

    knowledge_bm25 = get_knowledge_bm25()
    if not knowledge_bm25 or not milvus_client:
        return None

    knowledge_pipeline = RetrievalPipeline(
        client=milvus_client,
        collection="knowledge_chunks",
        bm25_index=knowledge_bm25,
        vector_top_k=settings.VECTOR_TOP_K,
        bm25_top_k=settings.BM25_TOP_K,
        final_top_k=settings.KNOWLEDGE_TOP_K,
        enable_rerank=True,
    )
    return RetrievalCheckService(
        pipeline=knowledge_pipeline,
        max_retries=settings.SELF_CHECK_MAX_RETRIES,
    )
