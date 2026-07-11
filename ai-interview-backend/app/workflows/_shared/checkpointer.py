"""AsyncPostgresSaver 单例

所有 LangGraph workflow 共享一个 checkpointer 连接池。
thread_id 由调用方传入（推荐用 "{workflow_type}-{业务主键}" 格式，如 "interview-42"）。

langgraph-checkpoint-postgres 3.x: from_conn_string() 返回 async context manager，
需手动管理 __aenter__/__aexit__ 以维持单例生命周期。
"""
from __future__ import annotations

import logging
from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings

logger = logging.getLogger(__name__)

_checkpointer: Optional[AsyncPostgresSaver] = None
_checkpointer_cm: Optional[object] = None  # async context manager 引用，用于关闭


def _build_db_url() -> str:
    """构造 asyncpg 兼容的数据库 URL"""
    return (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


async def get_checkpointer() -> AsyncPostgresSaver:
    """获取 checkpointer 单例（懒加载 + 异步 setup）"""
    global _checkpointer, _checkpointer_cm
    if _checkpointer is None:
        db_url = _build_db_url()
        _checkpointer_cm = AsyncPostgresSaver.from_conn_string(db_url)
        _checkpointer = await _checkpointer_cm.__aenter__()
        await _checkpointer.setup()  # 确保 checkpoint_* 表已建
        logger.info("AsyncPostgresSaver 初始化完成")
    return _checkpointer


async def close_checkpointer() -> None:
    """关闭 checkpointer（应用关闭时调用）"""
    global _checkpointer, _checkpointer_cm
    if _checkpointer_cm is not None:
        try:
            await _checkpointer_cm.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"关闭 checkpointer 失败: {e}")
        _checkpointer = None
        _checkpointer_cm = None