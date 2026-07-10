"""Interview 数据访问层（Phase 3 完整迁移，Phase 1 仅建骨架）"""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.interview import Interview
from app.models.interview_message import InterviewMessage
from app.repositories.base import BaseRepository


class InterviewRepository(BaseRepository[Interview]):
    model = Interview

    async def list_by_user(
        self,
        db: AsyncSession,
        user_id: int,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Interview]:
        """按用户列出面试记录（可选按状态过滤）"""
        stmt = select(Interview).where(Interview.user_id == user_id)
        if status is not None:
            stmt = stmt.where(Interview.status == status)
        stmt = stmt.order_by(Interview.created_at.desc()).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_messages(
        self,
        db: AsyncSession,
        interview_id: int,
    ) -> List[InterviewMessage]:
        """获取面试的所有消息（按 id 升序）"""
        stmt = (
            select(InterviewMessage)
            .where(InterviewMessage.interview_id == interview_id)
            .order_by(InterviewMessage.id)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_active_question(
        self,
        db: AsyncSession,
        interview: Interview,
    ) -> Optional[dict]:
        """获取当前索引对应的题目（从 interview.questions_data JSON 字段）"""
        idx = interview.current_question_index
        questions = interview.questions_data or []
        if 0 <= idx < len(questions):
            return questions[idx]
        return None


# 默认单例
interview_repo = InterviewRepository()