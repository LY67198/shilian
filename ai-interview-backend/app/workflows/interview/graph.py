"""面试 StateGraph 编译

interrupt_after=["ask_question"] 实现人在回路：
每次 ask_question 后暂停，等待用户 POST /answer resume。
Resume 后从 ask_question → fetch_context 重新走完整流水线。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from app.workflows._shared.checkpointer import get_checkpointer
from app.workflows.interview.nodes import (
    ask_question_node,
    check_finished_node,
    evaluate_node,
    fetch_context_node,
    generate_report_node,
    retrieve_knowledge_node,
)
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)

_compiled_graph: Optional[StateGraph] = None
_lock = asyncio.Lock()


def build_interview_graph() -> StateGraph:
    """构建 Interview StateGraph（不编译，仅声明结构）"""
    builder = StateGraph(InterviewState)

    builder.add_node("fetch_context", fetch_context_node)
    builder.add_node("retrieve_knowledge", retrieve_knowledge_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("check_finished", check_finished_node)
    builder.add_node("ask_question", ask_question_node)
    builder.add_node("generate_report", generate_report_node)

    builder.add_edge(START, "fetch_context")
    builder.add_edge("fetch_context", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "evaluate")
    builder.add_edge("evaluate", "check_finished")

    builder.add_conditional_edges(
        "check_finished",
        _route_after_check,
        {"continue": "ask_question", "finish": "generate_report"},
    )

    # 关键：ask_question 后回到 fetch_context（resume 后重查 DB）
    builder.add_edge("ask_question", "fetch_context")
    builder.add_edge("generate_report", END)

    return builder


async def get_compiled_graph():
    """获取编译后的 graph 实例（单例，线程安全）

    Returns:
        CompiledStateGraph with checkpointer + interrupt_after=["ask_question"]
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    async with _lock:
        if _compiled_graph is not None:
            return _compiled_graph
        checkpointer: AsyncPostgresSaver = await get_checkpointer()
        builder = build_interview_graph()
        _compiled_graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_after=["ask_question"],
        )
        logger.info("Interview StateGraph 编译完成")
    return _compiled_graph


def _route_after_check(state: InterviewState) -> str:
    """判断面试是否结束，返回条件边路由目标。

    Args:
        state: 当前面试状态。

    Returns:
        "finish" 表示进入报告生成节点，"continue" 表示继续下一题。
    """
    if state.get("is_finished"):
        return "finish"
    return "continue"
