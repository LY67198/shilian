"""Milvus 客户端单例（MilvusClient 新 API）

用 pymilvus 2.4+ 的 MilvusClient 替代老 API（connections.connect + Collection），原因：
1. MilvusClient 是官方推荐 API，自动管理连接池
2. 支持 ctx manager 用法，资源自动释放
3. 类型签名更友好
"""
from __future__ import annotations

import logging
from typing import Optional

from pymilvus import MilvusClient

from app.core.config import settings

logger = logging.getLogger(__name__)

_client: Optional[MilvusClient] = None


def _create_client() -> MilvusClient:
    """根据配置创建新的 MilvusClient 实例。

    Args:
        （无参数，从全局 settings 读取连接配置）

    Returns:
        已配置的 MilvusClient 实例。
    """
    kwargs = {
        "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
        "db_name": settings.MILVUS_DB_NAME,
    }
    if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
        kwargs["user"] = settings.MILVUS_USER
        kwargs["password"] = settings.MILVUS_PASSWORD
    return MilvusClient(**kwargs)


def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例（懒加载）"""
    global _client
    if _client is None:
        _client = _create_client()
        logger.info(
            f"Milvus 客户端初始化: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        )
    return _client


def health_check_milvus() -> bool:
    """Milvus 健康检查"""
    try:
        client = get_milvus_client()
        client.get_server_version()
        return True
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        return False


def _reset_milvus_client() -> None:
    """测试用：清空单例（强制下次重建连接）"""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None