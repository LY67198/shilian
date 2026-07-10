"""Milvus 客户端单例（MilvusClient 新 API）

参考 OnCall Agent 的 milvus_client.py 模式，但用 pymilvus 2.4+ 的 MilvusClient
替代老 API（connections.connect + Collection），原因：
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


class MilvusClientWrapper:
    """Milvus 客户端包装（单例）"""

    _instance: Optional[MilvusClient] = None

    @classmethod
    def get_client(cls) -> MilvusClient:
        """懒加载 + 单例"""
        if cls._instance is None:
            cls._instance = cls._create_client()
            logger.info(
                f"Milvus 客户端初始化: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
            )
        return cls._instance

    @classmethod
    def _create_client(cls) -> MilvusClient:
        kwargs = {
            "uri": f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}",
            "db_name": settings.MILVUS_DB_NAME,
        }
        if settings.MILVUS_USER and settings.MILVUS_PASSWORD:
            kwargs["user"] = settings.MILVUS_USER
            kwargs["password"] = settings.MILVUS_PASSWORD
        return MilvusClient(**kwargs)

    @classmethod
    def health_check(cls) -> bool:
        """健康检查（version() 是最轻量的远程调用）"""
        try:
            client = cls.get_client()
            client.get_server_version()
            return True
        except Exception as e:
            logger.warning(f"Milvus 健康检查失败: {e}")
            return False

    @classmethod
    def reset(cls) -> None:
        """测试用：清空单例（强制下次重建连接）"""
        if cls._instance is not None:
            try:
                cls._instance.close()
            except Exception:
                pass
        cls._instance = None


# 向后兼容的类实例（旧代码可能 import milvus_client.MilvusClientWrapper）
milvus_client = MilvusClientWrapper


# ─── 推荐用法：函数式 API ───

def get_milvus_client() -> MilvusClient:
    """获取 Milvus 客户端单例（推荐用法）

    用法：
        from app.vector_db import get_milvus_client
        client = get_milvus_client()
    """
    return MilvusClientWrapper.get_client()


def health_check_milvus() -> bool:
    """Milvus 健康检查（推荐用法）"""
    return MilvusClientWrapper.health_check()