"""公共 Workflow State 字段

所有 LangGraph workflow 都应该继承这个基类，自动获得：
- thread_id（外部传，标识会话）
- retry_count（自检循环计数器）
- errors（节点异常收集）
"""
from __future__ import annotations

from typing import Annotated, Any, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class BaseWorkflowState(TypedDict, total=False):
    """所有 workflow 共享的 state 字段"""
    # 会话标识（外部传，对应 LangGraph thread_id）
    thread_id: str

    # 自检循环计数器（用于 should_continue 等条件边）
    retry_count: int

    # 节点异常收集（异常处理节点用）
    errors: List[str]

    # LangChain 消息历史（用 add_messages reducer 追加）
    messages: Annotated[List[BaseMessage], add_messages]

    # 业务自定义字段
    custom: Any