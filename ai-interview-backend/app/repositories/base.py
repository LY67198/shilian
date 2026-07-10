"""Repository 基类

提供通用 CRUD 模式，子类只需指定 model 类即可获得：
- get_by_id
- list_all
- count

具体业务查询（带过滤、join、aggregate）仍在各 repo 里写。
"""
from __future__ import annotations

from typing import Generic, List, Optional, Type, TypeVar

from sqlalchemy import func, select
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

    model: Type[T]  # 子类覆盖

    async def get_by_id(self, db: AsyncSession, id: int) -> Optional[T]:
        """按主键获取"""
        return await db.get(self.model, id)

    async def list_all(
        self,
        db: AsyncSession,
        limit: int = 100,
        offset: int = 0,
    ) -> List[T]:
        """全表分页查询（不带过滤）"""
        stmt = select(self.model).limit(limit).offset(offset)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def count(self, db: AsyncSession) -> int:
        """全表行数"""
        stmt = select(func.count()).select_from(self.model)
        result = await db.execute(stmt)
        return int(result.scalar_one())

    async def create(self, db: AsyncSession, obj: T) -> T:
        """插入并 flush（不 commit）"""
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def delete(self, db: AsyncSession, id: int) -> bool:
        """按主键删除（不 commit）"""
        obj = await self.get_by_id(db, id)
        if obj is None:
            return False
        await db.delete(obj)
        return True