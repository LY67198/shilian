"""fetch_context node — 从 DB 查询面试上下文"""
from __future__ import annotations

import json
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.resume import Resume
from app.repositories.interview_repo import interview_repo
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def fetch_context_node(state: InterviewState) -> dict:
    """查询面试/简历/对话历史/当前题，写入 state"""
    interview_id = state["interview_id"]
    user_id = state["user_id"]

    custom = state.get("custom") or {}
    db: AsyncSession = custom.get("db")
    if not db:
        raise RuntimeError("fetch_context_node requires db session in state.custom.db")

    # 查面试记录
    interview = await interview_repo.get_by_id_for_user(db, interview_id, user_id)
    if not interview:
        raise ValueError(f"面试记录不存在: id={interview_id}")

    # 查简历
    resume_query = select(Resume).where(Resume.id == interview.resume_id)
    resume_result = await db.execute(resume_query)
    resume = resume_result.scalar_one_or_none()
    try:
        parsed_resume = json.loads(resume.parsed_content) if resume and resume.parsed_content else {}
    except json.JSONDecodeError:
        logger.warning(f"简历 parsed_content 不是有效 JSON，使用空 dict: resume_id={interview.resume_id}")
        parsed_resume = {}

    # 查对话历史
    messages = await interview_repo.list_messages(db, interview_id)
    chat_history = [{"role": m.role, "content": m.content} for m in messages]

    # 当前题目
    current_index = interview.current_question_index
    questions = interview.questions_data or []
    if current_index >= len(questions):
        raise ValueError(f"题目索引越界: index={current_index}, total={len(questions)}")

    current_q = questions[current_index]
    current_question = current_q.get("question", "")
    reference_answer = current_q.get("reference_answer")
    key_points = current_q.get("key_points")

    return {
        "target_position": interview.target_position or "",
        "difficulty": interview.difficulty or "medium",
        "total_questions": interview.total_questions or len(questions),
        "current_index": current_index,
        "questions": questions,
        "current_question": current_question,
        "reference_answer": reference_answer,
        "key_points": key_points,
        "resume_context": parsed_resume,
        "chat_history": chat_history,
        "score": 0,
        "feedback": "",
        "knowledge_context": [],
        "is_finished": False,
    }
