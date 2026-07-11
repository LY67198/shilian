"""ask_question node — 推进题目索引 + 存下一题消息到 DB"""
from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.interview_repo import interview_repo
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def ask_question_node(state: InterviewState, config: RunnableConfig) -> dict:
    """current_index += 1, 存下一题的 InterviewMessage"""
    db: AsyncSession = config["configurable"]["db"]

    interview_id = state["interview_id"]
    current_index = state["current_index"]
    questions = state["questions"]
    next_index = current_index + 1

    await interview_repo.update_question_index(db, interview_id, next_index)

    if next_index >= len(questions):
        raise RuntimeError(
            f"题目索引越界: next_index={next_index}, total={len(questions)}"
        )

    next_question = questions[next_index]["question"]
    interview_repo.create_message(
        db, interview_id, role="interviewer",
        content=next_question, question_index=next_index,
    )
    await db.commit()

    return {
        "current_index": next_index,
        "current_question": next_question,
        "next_question": next_question,
        "index": next_index,
    }
