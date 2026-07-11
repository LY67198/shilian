"""BM25 index lifecycle — build on startup, provide singletons"""
from __future__ import annotations

import logging

from app.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)

# Module-level singletons, built during startup
knowledge_bm25: BM25Index | None = None
question_bank_bm25: BM25Index | None = None


async def build_bm25_indices(db_session) -> None:
    """Build both BM25 indices from PostgreSQL data.

    Called during FastAPI startup event.
    """
    global knowledge_bm25, question_bank_bm25

    from sqlalchemy import select
    from app.models.knowledge import KnowledgeChunk
    from app.models.question_bank import QuestionBank

    # Build knowledge index
    result = await db_session.execute(select(KnowledgeChunk.content))
    knowledge_texts = [row[0] for row in result.fetchall() if row[0]]
    knowledge_bm25 = BM25Index("knowledge_chunks")
    knowledge_bm25.build(knowledge_texts)

    # Build question bank index
    result = await db_session.execute(
        select(QuestionBank.question, QuestionBank.reference_answer)
    )
    q_texts = []
    for question, answer in result.fetchall():
        text = f"{question or ''} {answer or ''}".strip()
        if text:
            q_texts.append(text)
    question_bank_bm25 = BM25Index("question_bank")
    question_bank_bm25.build(q_texts)

    logger.info(
        "BM25 indices built: knowledge=%d, question_bank=%d",
        knowledge_bm25.corpus_size,
        question_bank_bm25.corpus_size,
    )


def get_knowledge_bm25() -> BM25Index | None:
    """获取知识库 BM25 索引单例。

    Returns:
        已构建的知识库 BM25Index 实例，若尚未构建则返回 None。
    """
    return knowledge_bm25


def get_question_bank_bm25() -> BM25Index | None:
    """获取题库 BM25 索引单例。

    Returns:
        已构建的题库 BM25Index 实例，若尚未构建则返回 None。
    """
    return question_bank_bm25
