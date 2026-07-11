"""Workflow 级 LLM 工厂

与 app.llm.get_chat_llm 的区别：
- 自动应用 LangSmith tracing（如果启用）
- workflow 内可定制的 prompt prefix
- 统一的 temperature / max_tokens 默认
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.llm import get_chat_llm

logger = logging.getLogger(__name__)


def get_workflow_llm(
    temperature: float = 0.7,
    max_tokens: int = 2000,
    workflow_name: Optional[str] = None,
) -> ChatOpenAI:
    """
    获取 workflow 用的 LLM

    Args:
        temperature: 0-1
        max_tokens: 最大输出 token
        workflow_name: 用于 LangSmith trace 标识（可选）
    """
    llm = get_chat_llm(temperature=temperature, max_tokens=max_tokens)

    # Phase 1 阶段：trace 配置在 tracing.py，未来启用
    if workflow_name:
        logger.debug(f"LLM for workflow: {workflow_name}")

    return llm