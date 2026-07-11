"""Repository 基类 — 提供 get_by_id 模式，避免重复写 `await db.get(Model, id)`"""
from __future__ import annotations

from typing import Generic, Optional, Type, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """泛型 Repository 基类

    用法：
        class InterviewRepository(BaseRepository[Interview]):
            model = Interview

        repo = InterviewRepository()
        interview = await repo.get_by_id(db, 42)
    """

    model: Type[T]

    async def get_by_id(self, db: AsyncSession, id: int) -> Optional[T]:
        """按主键获取"""
        return await db.get(self.model, id)