"""generate_report node — 委托 ReportAgent 生成报告"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.report_agent import ReportAgent
from app.repositories.interview_repo import interview_repo
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def generate_report_node(state: InterviewState) -> dict:
    """汇总所有评分，委托 ReportAgent 生成报告"""
    custom = state.get("custom") or {}
    db: AsyncSession = custom.get("db")
    if not db:
        raise RuntimeError("generate_report_node requires db session in state.custom.db")

    agent: ReportAgent = custom.get("report_agent")
    if agent is None:
        logger.warning("report_agent not found in state.custom, using default")
        agent = ReportAgent()

    interview_id = state["interview_id"]
    questions = state["questions"]
    resume_context = state.get("resume_context", {})
    target_position = state.get("target_position", "")

    scored_msgs = await interview_repo.get_scored_messages(db, interview_id)

    qa_data = []
    all_scores = []
    msg_by_idx = {m.question_index: m for m in scored_msgs}

    for i, q in enumerate(questions):
        m = msg_by_idx.get(i)
        score_val = float(m.score) if m and m.score else state.get("score", 0)
        answer_text = (
            m.content
            if m
            else (state.get("answer", "") if i == state.get("current_index", 0) else "未回答")
        )
        qa_data.append({
            "question": q["question"],
            "answer": answer_text,
            "score": score_val,
        })
        all_scores.append(score_val)

    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    report = await agent.generate(
        resume_context=resume_context,
        target_position=target_position,
        qa_data=qa_data,
    )

    report["question_scores"] = [
        {"question": qa["question"], "score": qa["score"], "feedback": ""}
        for qa in qa_data
    ]

    await interview_repo.update_result(db, interview_id, overall, report)
    await db.commit()

    return {
        "overall_score": overall,
        "report": report,
        "all_scores": qa_data,
        "is_finished": True,
    }
