"""Embedding 工厂（从 services/common/embedding.py 迁移过来）

业务元数据:
- DashScope text-embedding-v3 输出 1024 维向量
- 已归一化，可与 Milvus L2 距离度量配合（见 vector_db/index.py）
"""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

_embeddings: Optional[DashScopeEmbeddings] = None

BATCH_SIZE = 25  # DashScope 单次最多 25 条


def get_embeddings() -> DashScopeEmbeddings:
    """获取 DashScope Embeddings 单例实例。

    Returns:
        DashScopeEmbeddings 实例。
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeEmbeddings(
            model=settings.KNOWLEDGE_EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )
    return _embeddings


async def embed_text(text: str) -> List[float]:
    """单文本异步向量化"""
    try:
        return await get_embeddings().aembed_query(text)
    except Exception as e:
        logger.error(f"Embedding 失败: {e}")
        raise


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量异步向量化（自动分批，每批最多 25 条）"""
    if not texts:
        return []
    results: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        try:
            vecs = await get_embeddings().aembed_documents(batch)
            results.extend(vecs)
        except Exception as e:
            logger.error(f"批量 Embedding 第 {i // BATCH_SIZE + 1} 批失败: {e}")
            raise
    return results


def embed_text_sync(text: str) -> List[float]:
    """同步向量化（用于 Celery 任务）"""
    try:
        return get_embeddings().embed_query(text)
    except Exception as e:
        logger.error(f"同步 Embedding 失败: {e}")
        raise


def embed_texts_sync(texts: List[str]) -> List[List[float]]:
    """批量同步向量化（用于 Celery 任务）"""
    if not texts:
        return []
    results: List[List[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        try:
            vecs = get_embeddings().aembed_documents(batch)
            results.extend(vecs)
        except Exception as e:
            logger.error(f"批量同步 Embedding 第 {i // BATCH_SIZE + 1} 批失败: {e}")
            raise
    return results