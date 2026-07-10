"""LangSmith tracing 配置

默认禁用。开启方式：
1. .env 加 LANGSMITH_API_KEY=lsv2_xxx
2. .env 加 LANGSMITH_PROJECT=shilian-dev（可选）
3. 重启服务

所有 workflow 的 graph.astream() / graph.ainvoke() 自动 trace 到 LangSmith。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_tracing_enabled() -> bool:
    """检查 LangSmith tracing 是否启用"""
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def configure_langsmith() -> None:
    """初始化 LangSmith 环境变量（在 app 启动时调用一次）"""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        logger.debug("LangSmith tracing 未启用（缺 LANGSMITH_API_KEY）")
        return

    # LangChain 客户端自动读这些环境变量
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    project = os.environ.get("LANGSMITH_PROJECT", "shilian-dev")
    os.environ["LANGCHAIN_PROJECT"] = project

    logger.info(f"LangSmith tracing 启用，项目: {project}")