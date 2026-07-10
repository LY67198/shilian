"""question_bank collection schema + CRUD

字段设计：
- `id` 与 PostgreSQL `question_bank.id` 同源
- 业务字段（category / position_tag / difficulty）冗余存 Milvus 便于过滤
- 召回时除了向量相似度，还要按 position_tag / difficulty 过滤
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

COLLECTION_NAME = settings.MILVUS_COLLECTION_QUESTION_BANK  # "question_bank"


def get_schema(dim: int) -> CollectionSchema:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="position_tag", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="difficulty", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="reference_answer", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="key_points", dtype=DataType.JSON),
        FieldSchema(name="tags", dtype=DataType.JSON),
        FieldSchema(name="embedding_text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(
        fields=fields,
        description="Question bank items with embeddings",
        enable_dynamic_field=False,
    )


class QuestionPayload(BaseModel):
    """插入 / 检索时的数据载荷"""
    id: int = Field(..., description="Postgres question_bank.id")
    category: str
    position_tag: str
    difficulty: str
    question: str
    reference_answer: Optional[str] = None
    key_points: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    embedding_text: Optional[str] = None
    embedding: List[float]


def create_collection(client: MilvusClient, dim: int) -> None:
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Collection 已存在: {COLLECTION_NAME}")
        ensure_index(client, COLLECTION_NAME)
        try:
            client.load_collection(COLLECTION_NAME)
        except Exception:
            pass
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


def insert_questions(client: MilvusClient, questions: List[QuestionPayload]) -> List[int]:
    """批量插入 questions，返回 ids（自动 flush）"""
    if not questions:
        return []
    data = [q.model_dump() for q in questions]
    result = client.insert(collection_name=COLLECTION_NAME, data=data)
    client.flush(COLLECTION_NAME)
    ids = []
    if isinstance(result, dict):
        ids = result.get("primary_keys", []) or result.get("ids", []) or []
    else:
        ids = getattr(result, "primary_keys", None) or []
    logger.info(f"Milvus 插入并 flush {len(ids)} questions 到 {COLLECTION_NAME}")
    return ids


def search(
    client: MilvusClient,
    query_vector: List[float],
    top_k: int = 20,
    position_tag: Optional[str] = None,
    difficulty: Optional[str] = None,
    min_score: float = 0.7,
) -> List[Dict[str, Any]]:
    """向量检索（支持 position_tag / difficulty 过滤）"""
    filters = []
    if position_tag:
        filters.append(f'position_tag like "%{position_tag}%"')
    if difficulty:
        filters.append(f'difficulty == "{difficulty}"')
    filter_expr = " and ".join(filters) if filters else ""

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field=EMBEDDING_FIELD,
        search_params=DEFAULT_SEARCH_PARAMS,
        limit=top_k,
        output_fields=[
            "category", "position_tag", "difficulty",
            "question", "reference_answer", "key_points", "tags",
        ],
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
                "category": hit["entity"].get("category"),
                "position_tag": hit["entity"].get("position_tag"),
                "difficulty": hit["entity"].get("difficulty"),
                "question": hit["entity"].get("question"),
                "reference_answer": hit["entity"].get("reference_answer"),
                "key_points": hit["entity"].get("key_points"),
                "tags": hit["entity"].get("tags"),
                "similarity": score,
                "source": "from_bank",
            })
    return out


def delete_question(client: MilvusClient, question_id: int) -> int:
    result = client.delete(
        collection_name=COLLECTION_NAME,
        filter=f"id == {question_id}",
    )
    return result.get("delete_count", 0) if isinstance(result, dict) else getattr(result, "delete_count", 0)