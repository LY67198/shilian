"""Interview 数据访问层"""
from __future__ import annotations

import decimal
import json
from typing import Any, List, Optional

from sqlalchemy import select, update as sql_update
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

    async def get_by_id_for_user(
        self,
        db: AsyncSession,
        interview_id: int,
        user_id: int,
    ) -> Optional[Interview]:
        """按 id + user_id 查面试记录（含归属权校验）"""
        stmt = select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user_id,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

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

    async def get_scored_messages(
        self,
        db: AsyncSession,
        interview_id: int,
    ) -> List[InterviewMessage]:
        """获取已评分的候选人消息"""
        stmt = select(InterviewMessage).where(
            InterviewMessage.interview_id == interview_id,
            InterviewMessage.role == "candidate",
            InterviewMessage.score.isnot(None),
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

    async def update_question_index(
        self,
        db: AsyncSession,
        interview_id: int,
        next_index: int,
    ) -> None:
        """更新当前题目索引"""
        await db.execute(
            sql_update(Interview)
            .where(Interview.id == interview_id)
            .values(current_question_index=next_index)
        )

    async def update_result(
        self,
        db: AsyncSession,
        interview_id: int,
        overall_score: float,
        report: dict[str, Any],
    ) -> None:
        """写入面试最终结果"""
        await db.execute(
            sql_update(Interview)
            .where(Interview.id == interview_id)
            .values(
                status="completed",
                overall_score=decimal.Decimal(str(overall_score)),
                report=json.dumps(report, ensure_ascii=False),
            )
        )

    async def create_message(
        self,
        db: AsyncSession,
        interview_id: int,
        role: str,
        content: str,
        question_index: int = -1,
    ) -> InterviewMessage:
        """创建面试消息"""
        msg = InterviewMessage(
            interview_id=interview_id,
            role=role,
            content=content,
            question_index=question_index,
        )
        db.add(msg)
        return msg


# 默认单例
interview_repo = InterviewRepository()