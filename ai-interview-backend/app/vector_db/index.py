"""Milvus 索引配置

统一管理 collection 索引的创建与查询参数。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pymilvus import MilvusClient

from app.core.config import settings

logger = logging.getLogger(__name__)

# 向量字段名
EMBEDDING_FIELD = "embedding"
# 主键字段名（与 Postgres 自增 ID 对齐）
PK_FIELD = "id"

# HNSW 索引参数（dev 环境默认值，prod 可在 .env 调参）
# pymilvus MilvusClient.create_index 期望 list[{field_name, metric_type, index_type, params}]
# 选用 L2 而非 COSINE：Milvus 2.4.10 的 HNSW+COSINE 有 bug（L2 self-match 正常）
# DashScope text-embedding-v3 输出已归一化，L2 distance 对归一化向量等价于 cosine distance：
#   cos_sim = 1 - L2_distance² / 2
#   对完全匹配：distance=0, similarity=1；对正交向量：distance=√2, similarity=0
def build_index_params(
    field_name: str = EMBEDDING_FIELD,
    metric_type: str = "L2",
    index_type: str = "HNSW",
    params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """构造 create_index 期望的 index_params 列表"""
    if params is None:
        params = {"M": 16, "efConstruction": 200}
    return [{
        "field_name": field_name,
        "metric_type": metric_type,
        "index_type": index_type,
        "params": params,
    }]


# 向后兼容：DEFAULT_INDEX_PARAMS 作为单 dict 形式（搜索时用）
DEFAULT_SEARCH_PARAMS: Dict[str, Any] = {
    "metric_type": "L2",
    "params": {"ef": 64},
}


def l2_distance_to_similarity(distance: float) -> float:
    """L2 distance → 0-1 similarity（前提：embedding 已归一化）

    - distance=0   → similarity=1.0 （完全匹配）
    - distance=√2  → similarity=0.0 （正交）
    - distance=2   → similarity=-1.0（反向；不会发生于归一化向量）
    """
    import math
    sim = 1.0 - distance / math.sqrt(2)
    return round(sim, 4)


def ensure_index(
    client: MilvusClient,
    collection_name: str,
    field_name: str = EMBEDDING_FIELD,
) -> None:
    """确保 collection 上有指定索引（已存在则跳过）"""
    try:
        client.create_index(
            collection_name=collection_name,
            index_params=build_index_params(field_name=field_name),
        )
        logger.info(f"索引已创建: {collection_name}.{field_name} (HNSW, COSINE)")
    except Exception as e:
        # 索引已存在时 Milvus 抛异常，幂等处理
        if "already" in str(e).lower() or "exist" in str(e).lower():
            logger.debug(f"索引已存在: {collection_name}.{field_name}")
            return
        raise


def describe_index(client: MilvusClient, collection_name: str) -> List[Dict]:
    """查询 collection 的索引详情"""
    return client.describe_index(collection_name=collection_name)


def has_collection(client: MilvusClient, collection_name: str) -> bool:
    return client.has_collection(collection_name)


def drop_collection(client: MilvusClient, collection_name: str) -> None:
    """删除 collection（仅供测试 / 重置使用）"""
    client.drop_collection(collection_name)


def get_search_params() -> Dict[str, Any]:
    """运行时获取搜索参数（允许 .env 覆盖）"""
    return DEFAULT_SEARCH_PARAMS.copy()