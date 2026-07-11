"""题库服务（Milvus 版）

业务字段在 PostgreSQL，embedding 在 Milvus。
- 写策略：PG insert → 拿 auto-increment id → Milvus 同 id 写入；失败回滚 PG
- 读策略：直接调 Milvus search，绕过 PG
"""
from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question_bank import QuestionBank
from app.llm.embedding import embed_text, embed_texts
from app.vector_db import get_milvus_client
from app.vector_db.collections import question_bank as question_vdb
from app.vector_db.collections.question_bank import QuestionPayload

logger = logging.getLogger(__name__)


def _build_embedding_text(question: str, reference_answer: Optional[str]) -> str:
    """构造用于向量化的文本：题面 + 参考答案"""
    parts = [question.strip()]
    if reference_answer and reference_answer.strip():
        parts.append(f"参考答案：{reference_answer.strip()}")
    return "\n\n".join(parts)


class QuestionBankService:

    # ── CRUD ────────────────────────────────────────────────────────────

    async def create(self, db: AsyncSession, data: dict) -> QuestionBank:
        """新增题目：PG 拿 id → Milvus 写 embedding；Milvus 失败回滚 PG"""
        emb_text = _build_embedding_text(data["question"], data.get("reference_answer"))

        # 1. PG insert（先不 embed，拿到 id 再 embed，避免无效 embedding 浪费）
        q = QuestionBank(
            category=data["category"],
            position_tag=data["position_tag"],
            difficulty=data["difficulty"],
            question=data["question"],
            reference_answer=data.get("reference_answer"),
            key_points=data.get("key_points"),
            tags=data.get("tags"),
            source=data.get("source", "manual"),
            created_by=data.get("created_by"),
            embedding_text=emb_text,
        )
        db.add(q)
        await db.commit()
        await db.refresh(q)

        # 2. Embed + Milvus insert
        try:
            embedding = await embed_text(emb_text)
            client = get_milvus_client()
            question_vdb.insert_questions(client, [
                QuestionPayload(
                    id=q.id,
                    category=q.category,
                    position_tag=q.position_tag,
                    difficulty=q.difficulty,
                    question=q.question,
                    reference_answer=q.reference_answer,
                    key_points=q.key_points,
                    tags=q.tags,
                    embedding_text=emb_text,
                    embedding=embedding,
                )
            ])
            logger.info(f"题目 {q.id} 已创建并向量化（PG + Milvus）")
        except Exception as e:
            # Milvus 失败 → 回滚 PG
            await db.delete(q)
            await db.commit()
            logger.error(f"题目创建 Milvus 失败，已回滚 PG: {e}")
            raise
        return q

    async def get_by_id(self, db: AsyncSession, question_id: int) -> Optional[QuestionBank]:
        return await db.get(QuestionBank, question_id)

    async def update(self, db: AsyncSession, question_id: int, data: dict) -> Optional[QuestionBank]:
        """更新题目；若改题面/答案则重新向量化（同步写 PG + Milvus）"""
        q = await db.get(QuestionBank, question_id)
        if not q:
            return None

        need_reembed = False
        for field in ("question", "reference_answer"):
            if field in data and data[field] != getattr(q, field):
                need_reembed = True

        for field, val in data.items():
            if hasattr(q, field) and field != "id":
                setattr(q, field, val)

        await db.commit()
        await db.refresh(q)

        if need_reembed:
            emb_text = _build_embedding_text(q.question, q.reference_answer)
            q.embedding_text = emb_text
            await db.commit()
            # Milvus：delete + reinsert（同 id）
            client = get_milvus_client()
            question_vdb.delete_question(client, q.id)
            embedding = await embed_text(emb_text)
            question_vdb.insert_questions(client, [
                QuestionPayload(
                    id=q.id,
                    category=q.category,
                    position_tag=q.position_tag,
                    difficulty=q.difficulty,
                    question=q.question,
                    reference_answer=q.reference_answer,
                    key_points=q.key_points,
                    tags=q.tags,
                    embedding_text=emb_text,
                    embedding=embedding,
                )
            ])
            logger.info(f"题目 {q.id} 已更新并重新向量化")
        return q

    async def delete(self, db: AsyncSession, question_id: int) -> bool:
        """删除：Milvus + PG 双删"""
        # 先 Milvus（避免 PG 删了但 Milvus 还有）
        try:
            client = get_milvus_client()
            question_vdb.delete_question(client, question_id)
        except Exception as e:
            logger.warning(f"Milvus 删除 question_id={question_id} 失败（继续删 PG）: {e}")

        result = await db.execute(
            delete(QuestionBank).where(QuestionBank.id == question_id)
        )
        await db.commit()
        return result.rowcount > 0

    async def toggle(self, db: AsyncSession, question_id: int, is_active: bool) -> bool:
        """启用/禁用（PG only，Milvus 端按 is_active 过滤由调用方决定）"""
        result = await db.execute(
            update(QuestionBank)
            .where(QuestionBank.id == question_id)
            .values(is_active=is_active)
        )
        await db.commit()
        return result.rowcount > 0

    async def get_list(
        self,
        db: AsyncSession,
        page: int = 1,
        size: int = 20,
        category: Optional[str] = None,
        position_tag: Optional[str] = None,
        difficulty: Optional[str] = None,
        is_active: Optional[bool] = None,
        search: Optional[str] = None,
    ) -> dict:
        """分页查询（仅 PG，不涉及向量）"""
        stmt = select(QuestionBank)
        if category:
            stmt = stmt.where(QuestionBank.category == category)
        if position_tag:
            stmt = stmt.where(QuestionBank.position_tag.contains(position_tag))
        if difficulty:
            stmt = stmt.where(QuestionBank.difficulty == difficulty)
        if is_active is not None:
            stmt = stmt.where(QuestionBank.is_active == is_active)
        if search:
            stmt = stmt.where(
                or_(
                    QuestionBank.question.ilike(f"%{search}%"),
                    QuestionBank.reference_answer.ilike(f"%{search}%"),
                )
            )

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await db.execute(count_stmt)).scalar_one()
        stmt = stmt.order_by(QuestionBank.created_at.desc()).offset((page - 1) * size).limit(size)
        items = (await db.execute(stmt)).scalars().all()
        return {"items": items, "total": total, "page": page, "size": size}

    # ── 向量化 ──────────────────────────────────────────────────────────

    async def embed_question(self, db: AsyncSession, question_id: int) -> bool:
        """单题重新向量化（PG embedding_text + Milvus embedding）"""
        q = await db.get(QuestionBank, question_id)
        if not q:
            return False
        emb_text = _build_embedding_text(q.question, q.reference_answer)
        q.embedding_text = emb_text
        await db.commit()

        embedding = await embed_text(emb_text)
        client = get_milvus_client()
        question_vdb.delete_question(client, q.id)
        question_vdb.insert_questions(client, [
            QuestionPayload(
                id=q.id,
                category=q.category,
                position_tag=q.position_tag,
                difficulty=q.difficulty,
                question=q.question,
                reference_answer=q.reference_answer,
                key_points=q.key_points,
                tags=q.tags,
                embedding_text=emb_text,
                embedding=embedding,
            )
        ])
        return True

    def batch_embed_sync(self, question_ids: List[int]) -> dict:
        """同步批量向量化（Celery 调用）"""
        import asyncio
        from app.db.base import get_session_local

        async def _run():
            async with get_session_local()() as db:
                stmt = select(QuestionBank).where(QuestionBank.id.in_(question_ids))
                rows = (await db.execute(stmt)).scalars().all()

                texts = [_build_embedding_text(q.question, q.reference_answer) for q in rows]
                embeddings = await embed_texts(texts)

                # 更新 PG embedding_text
                for q, txt in zip(rows, texts):
                    q.embedding_text = txt
                await db.commit()

                # Milvus 批量写入
                client = get_milvus_client()
                payloads = [
                    QuestionPayload(
                        id=q.id,
                        category=q.category,
                        position_tag=q.position_tag,
                        difficulty=q.difficulty,
                        question=q.question,
                        reference_answer=q.reference_answer,
                        key_points=q.key_points,
                        tags=q.tags,
                        embedding_text=txt,
                        embedding=emb,
                    )
                    for q, txt, emb in zip(rows, texts, embeddings)
                ]
                question_vdb.insert_questions(client, payloads)
                return len(rows)

        count = asyncio.run(_run())
        logger.info(f"批量向量化完成：{count} 题")
        return {"embedded": count}

    def reindex_all_sync(self) -> dict:
        """全量重建 embedding（升级模型时使用）"""
        import asyncio
        from app.db.base import get_session_local

        async def _run():
            async with get_session_local()() as db:
                stmt = select(QuestionBank)
                rows = (await db.execute(stmt)).scalars().all()
                ids = [q.id for q in rows]

            batch_size = 50
            total = 0
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i: i + batch_size]
                result = self.batch_embed_sync(batch_ids)
                total += result["embedded"]
            return total

        total = asyncio.run(_run())
        return {"total_reindexed": total}

    # ── 检索 ─────────────────────────────────────────────────────────────

    async def retrieve_questions(
        self,
        query: str,
        db: AsyncSession = None,  # noqa 保留参数仅为向后兼容（不再用）
        k: int = 20,
        position_tag: Optional[str] = None,
        difficulty: Optional[str] = None,
        min_score: float = 0.7,
    ) -> list[dict]:
        """语义检索（Milvus 端执行，支持 position_tag / difficulty 过滤）"""
        query_vec = await embed_text(query)
        client = get_milvus_client()
        return question_vdb.search(
            client,
            query_vec,
            top_k=k,
            position_tag=position_tag,
            difficulty=difficulty,
            min_score=min_score,
        )

    async def increment_use_count(self, db: AsyncSession, question_ids: List[int]) -> None:
        """面试选题后增加使用次数（PG only）"""
        if not question_ids:
            return
        await db.execute(
            update(QuestionBank)
            .where(QuestionBank.id.in_(question_ids))
            .values(use_count=QuestionBank.use_count + 1)
        )
        await db.commit()

    async def get_stats(self, db: AsyncSession) -> dict:
        """题库统计（PG only — Milvus 端所有有效记录都有 embedding）"""
        by_category = (
            await db.execute(
                select(QuestionBank.category, func.count().label("cnt"))
                .group_by(QuestionBank.category)
            )
        ).all()
        by_difficulty = (
            await db.execute(
                select(QuestionBank.difficulty, func.count().label("cnt"))
                .group_by(QuestionBank.difficulty)
            )
        ).all()
        top_used = (
            await db.execute(
                select(QuestionBank.id, QuestionBank.question, QuestionBank.use_count)
                .order_by(QuestionBank.use_count.desc())
                .limit(10)
            )
        ).all()
        total = (await db.execute(select(func.count()).select_from(QuestionBank))).scalar_one()

        return {
            "total": total,
            # 注：所有有效记录都在 Milvus 端，不再单独统计 embedded
            "embedded": total,
            "by_category": {row[0]: row[1] for row in by_category},
            "by_difficulty": {row[0]: row[1] for row in by_difficulty},
            "top_used": [{"id": r[0], "question": r[1][:50], "use_count": r[2]} for r in top_used],
        }


question_bank_service = QuestionBankService()
