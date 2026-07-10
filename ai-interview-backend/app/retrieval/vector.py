"""Vector recall — wraps existing Milvus collection search"""
from __future__ import annotations

import logging
from typing import Optional

from pymilvus import MilvusClient

from app.llm.embedding import embed_text
from app.retrieval import SearchResult
from app.vector_db.collections import knowledge as knowledge_vdb
from app.vector_db.collections import question_bank as question_bank_vdb

logger = logging.getLogger(__name__)

_COLLECTION_MAP = {
    "knowledge_chunks": knowledge_vdb,
    "question_bank": question_bank_vdb,
}


async def vector_search(
    client: MilvusClient,
    query: str,
    collection: str,
    top_k: int = 20,
    filters: Optional[dict] = None,
) -> list[SearchResult]:
    """Run vector similarity search against a Milvus collection.

    Args:
        client: MilvusClient instance.
        query: Raw text to embed and search.
        collection: "knowledge_chunks" or "question_bank".
        top_k: Number of results to return.
        filters: Collection-specific filters.
            knowledge_chunks: {"document_ids": [1, 2]}
            question_bank: {"position_tag": "python", "difficulty": "medium"}

    Returns:
        List of SearchResult sorted by similarity descending.
    """
    coll = _COLLECTION_MAP.get(collection)
    if coll is None:
        logger.warning(f"Unknown collection: {collection}")
        return []

    try:
        query_vec = await embed_text(query)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return []

    filters = filters or {}

    try:
        if collection == "knowledge_chunks":
            raw = coll.search(
                client=client,
                query_vector=query_vec,
                top_k=top_k,
                document_ids=filters.get("document_ids"),
                min_score=filters.get("min_score", 0.0),
            )
        else:
            raw = coll.search(
                client=client,
                query_vector=query_vec,
                top_k=top_k,
                position_tag=filters.get("position_tag"),
                difficulty=filters.get("difficulty"),
                min_score=filters.get("min_score", 0.7),
            )
    except Exception as e:
        logger.warning(f"Vector search failed for {collection}: {e}")
        return []

    return [
        SearchResult(
            id=r.get("id", 0),
            content=r.get("content") or r.get("question", ""),
            score=r.get("similarity", 0.0),
            metadata={k: v for k, v in r.items()
                      if k not in ("id", "content", "question", "similarity")},
            source="vector",
        )
        for r in raw
    ]
