"""QuestionAgent — 面试出题 Agent"""
from __future__ import annotations

import json
import logging
from typing import List

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.repositories.question_bank_repo import QuestionBankRepository

logger = logging.getLogger(__name__)


class QuestionItem(BaseModel):
    question: str
    reference_answer: str = ""
    key_points: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    bank_id: int | None = None  # None 表示 LLM 生成的非题库题


class QuestionAgent(BaseAgent):
    """出题 Agent — 优先题库 RAG，不足时 LLM 兜底"""

    def __init__(self, question_bank_repo: QuestionBankRepository | None = None):
        super().__init__(prompt_name="question_agent", temperature=0.9)
        self._bank_repo = question_bank_repo

    async def generate(
        self,
        db,
        position: str,
        difficulty: str,
        count: int,
        resume: dict,
    ) -> List[dict]:
        """生成 N 道题：先选题库，不足 LLM 兜底。

        Returns:
            list[dict]: [{"question": ..., "reference_answer": ..., "key_points": [...], ...}]
        """
        bank_qs = await self._select_from_bank(db, position, difficulty, count)

        if len(bank_qs) >= count:
            logger.info(f"题库命中 {len(bank_qs)} 道，满足 {count} 题需求")
            return bank_qs[:count]

        shortage = count - len(bank_qs)
        logger.info(f"题库命中 {len(bank_qs)} 道，缺 {shortage} 道，LLM 兜底生成")

        generated = await self._fallback_generate(
            position=position,
            difficulty=difficulty,
            count=shortage,
            resume=resume,
            bank_questions=bank_qs,
        )

        return bank_qs + generated

    async def _select_from_bank(
        self, db, position: str, difficulty: str, count: int
    ) -> List[dict]:
        """从题库 RAG 检索 + 选择"""
        if self._bank_repo is None:
            return []

        try:
            questions = await self._bank_repo.search_by_position(
                db, position_tag=position, difficulty=difficulty, limit=count
            )
        except Exception as e:
            logger.warning(f"题库查询失败: {e}")
            return []

        result = []
        for q in questions:
            result.append({
                "question": q.question,
                "reference_answer": q.reference_answer or "",
                "key_points": q.key_points or [],
                "difficulty": q.difficulty or difficulty,
                "bank_id": q.id,
            })
        return result

    async def _fallback_generate(
        self,
        position: str,
        difficulty: str,
        count: int,
        resume: dict,
        bank_questions: list,
    ) -> List[dict]:
        """LLM 兜底生成新题"""
        try:
            result = await self.invoke_structured(
                variables={
                    "position": position,
                    "difficulty": difficulty,
                    "count": str(count),
                    "resume_json": json.dumps(resume, ensure_ascii=False),
                    "bank_questions": json.dumps(bank_questions, ensure_ascii=False),
                },
                schema=QuestionItem,
            )
            return [_item_to_dict(result)]
        except Exception as e:
            logger.error(f"LLM 兜底出题失败: {e}")
            # 返回一个基础题
            return [{
                "question": f"请介绍你在{position}岗位上的核心技术能力",
                "reference_answer": "",
                "key_points": [],
                "difficulty": difficulty,
                "bank_id": None,
            }]


def _item_to_dict(item: QuestionItem) -> dict:
    return {
        "question": item.question,
        "reference_answer": item.reference_answer,
        "key_points": item.key_points,
        "difficulty": item.difficulty,
        "bank_id": item.bank_id,
    }
