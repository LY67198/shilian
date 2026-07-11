"""Knowledge Repository — 知识库数据访问层 (PG 元数据)"""
from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import KnowledgeChunk
from app.repositories.base import BaseRepository


class KnowledgeRepository(BaseRepository[KnowledgeChunk]):
    model = KnowledgeChunk

    async def list_by_document(
        self,
        db: AsyncSession,
        document_id: int,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[KnowledgeChunk], int]:
        """分页列出某文档的所有 chunk"""
        base = select(KnowledgeChunk).where(
            KnowledgeChunk.document_id == document_id,
        )
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await db.execute(count_stmt)
        total_count = int(total.scalar_one())

        stmt = (
            base
            .order_by(KnowledgeChunk.chunk_index.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all()), total_count

    async def get_by_chunk_id(
        self, db: AsyncSession, chunk_id: int
    ) -> Optional[KnowledgeChunk]:
        return await self.get_by_id(db, chunk_id)

    async def full_text_search(
        self,
        db: AsyncSession,
        query: str,
        limit: int = 20,
    ) -> List[KnowledgeChunk]:
        """PG ILIKE 全文搜索（非向量检索，用于辅助查询）"""
        pattern = f"%{query}%"
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.content.ilike(pattern))
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


knowledge_repo = KnowledgeRepository()
