"""knowledge_chunks collection schema + CRUD

字段设计原则：
- `id` 与 PostgreSQL `knowledge_chunks.id` 同源（auto-increment 同步）
- `embedding` 与 `KNOWLEDGE_EMBEDDING_DIM` 绑定（默认 1024）
- 其他元数据冗余存 Milvus 便于按 `document_id` / `chunk_index` 过滤
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType
from pydantic import BaseModel, Field

from app.core.config import settings
from app.vector_db.index import (
    DEFAULT_SEARCH_PARAMS,
    EMBEDDING_FIELD,
    PK_FIELD,
    ensure_index,
    l2_distance_to_similarity,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.MILVUS_COLLECTION_KNOWLEDGE  # "knowledge_chunks"


def get_schema(dim: int) -> CollectionSchema:
    """Collection schema 定义（pymilvus 2.4+ CollectionSchema 对象）"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="document_id", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(
        fields=fields,
        description="Knowledge base document chunks with embeddings",
        enable_dynamic_field=False,
    )


class KnowledgeChunkPayload(BaseModel):
    """插入 / 检索时的数据载荷"""
    id: int = Field(..., description="Postgres knowledge_chunks.id")
    document_id: int
    chunk_index: int
    content: str
    content_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: List[float]


def create_collection(client: MilvusClient, dim: int) -> None:
    """创建 collection + 索引（已存在跳过）"""
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Collection 已存在: {COLLECTION_NAME}")
        ensure_index(client, COLLECTION_NAME)
        try:
            client.load_collection(COLLECTION_NAME)
        except Exception:
            pass  # 已加载会报错，幂等
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=get_schema(dim),
        shards_num=2,
    )
    logger.info(f"Collection 已创建: {COLLECTION_NAME}")
    ensure_index(client, COLLECTION_NAME)
    client.load_collection(COLLECTION_NAME)
    logger.info(f"Collection 已加载到内存: {COLLECTION_NAME}")


def insert_chunks(client: MilvusClient, chunks: List[KnowledgeChunkPayload]) -> List[int]:
    """批量插入 chunks，返回 ids

    注意：Milvus 默认 insert 后数据在写缓冲，必须 flush() 才能被 search 命中。
    为简化调用方，每次 insert 后自动 flush（吞吐低但语义清晰）。
    生产高吞吐场景可改为批量 insert + 定时 flush。
    """
    if not chunks:
        return []
    data = [c.model_dump() for c in chunks]
    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(COLLECTION_NAME)  # 让数据立即可被 search
    ids = []
    if isinstance(result, dict):
        ids = result.get("primary_keys", []) or result.get("ids", []) or []
    else:
        ids = getattr(result, "primary_keys", None) or []
    logger.info(f"Milvus 插入并 flush {len(ids)} chunks 到 {COLLECTION_NAME}")
    return ids


def search(
    client: MilvusClient,
    query_vector: List[float],
    top_k: int = 4,
    document_ids: Optional[List[int]] = None,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """向量检索

    Returns:
        [{"id", "document_id", "chunk_index", "content", "content_hash",
          "metadata", "distance"}, ...]
    """
    filter_expr = ""
    if document_ids:
        ids_str = ", ".join(str(i) for i in document_ids)
        filter_expr = f"document_id in [{ids_str}]"

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field=EMBEDDING_FIELD,
        search_params=DEFAULT_SEARCH_PARAMS,
        limit=top_k,
        output_fields=["document_id", "chunk_index", "content", "content_hash", "metadata"],
        filter=filter_expr,
    )

    out: List[Dict[str, Any]] = []
    for hits in results:
        for hit in hits:
            score = l2_distance_to_similarity(float(hit["distance"]))
            if score < min_score:
                continue
            out.append({
                "id": hit["id"],
                "document_id": hit["entity"].get("document_id"),
                "chunk_index": hit["entity"].get("chunk_index"),
                "content": hit["entity"].get("content"),
                "content_hash": hit["entity"].get("content_hash"),
                "metadata": hit["entity"].get("metadata") or {},
                "similarity": score,
            })
    return out


def delete_by_document(client: MilvusClient, document_id: int) -> int:
    """按 document_id 删除所有 chunks，返回删除条数"""
    result = client.delete(
        collection_name=COLLECTION_NAME,
        filter=f"document_id == {document_id}",
    )
    count = result.get("delete_count", 0) if isinstance(result, dict) else getattr(result, "delete_count", 0)
    logger.info(f"Milvus 删除 document_id={document_id}, count={count}")
    return count


def get_by_ids(client: MilvusClient, ids: List[int]) -> List[Dict[str, Any]]:
    """按主键批量取（用于父子块组装 / 引用回溯）"""
    if not ids:
        return []
    ids_str = ", ".join(str(i) for i in ids)
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter=f"id in [{ids_str}]",
        output_fields=["document_id", "chunk_index", "content", "content_hash", "metadata"],
    )
    return list(results)