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


from contextlib import contextmanager
from typing import Optional
import time


@contextmanager
def trace_span(name: str, metadata: Optional[dict] = None):
    """创建 LangSmith 子 span，仅在 tracing 启用时生效

    Usage:
        with trace_span("vector_recall") as span:
            result = await vector_search(query)
            if span:
                span.add_outputs({"count": len(result)})
    """
    if not is_tracing_enabled():
        yield None
        return

    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:
        yield None
        return

    parent_run = get_current_run_tree()
    start = time.time()
    span = None
    try:
        span = parent_run.create_child(
            name=name,
            run_type="retriever",
            inputs=metadata or {},
        ) if parent_run else None
        yield span
        duration_ms = (time.time() - start) * 1000
        if span:
            span.add_outputs({"duration_ms": duration_ms, "status": "ok"})
            span.end()
    except Exception as exc:
        if span:
            span.add_outputs({"error": str(exc), "status": "error"})
            span.end()
        raise