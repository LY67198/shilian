"""向量数据库层（Milvus）

业务元数据存 PostgreSQL，embedding 存 Milvus。
"""
from app.vector_db.client import (
    MilvusClientWrapper,
    get_milvus_client,
    health_check_milvus,
)

__all__ = [
    "MilvusClientWrapper",  # 类（保留用于高级用法）
    "get_milvus_client",     # 函数式：推荐用法
    "health_check_milvus",   # 健康检查
]