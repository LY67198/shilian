"""向量数据库层（Milvus）

业务元数据存 PostgreSQL，embedding 存 Milvus。
"""
from app.vector_db.client import (
    get_milvus_client,
    health_check_milvus,
)

__all__ = [
    "get_milvus_client",
    "health_check_milvus",
]