"""generate_report node — 汇总评分 + LLM 生成综合报告"""
from __future__ import annotations

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.json_utils import extract_json
from app.llm import get_chat_llm
from app.llm.prompts import load_prompt
from app.repositories.interview_repo import interview_repo
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def generate_report_node(state: InterviewState) -> dict:
    """汇总所有评分，计算总分，调用 LLM 生成报告"""
    custom = state.get("custom") or {}
    db: AsyncSession = custom.get("db")
    if not db:
        raise RuntimeError("generate_report_node requires db session in state.custom.db")

    interview_id = state["interview_id"]
    questions = state["questions"]
    resume_context = state.get("resume_context", {})
    target_position = state.get("target_position", "")

    scored_msgs = await interview_repo.get_scored_messages(db, interview_id)

    # 构造 qa_data
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

    # LLM 生成报告
    try:
        prompt = load_prompt("generate_report")
        llm = get_chat_llm(temperature=0.5)
        chain = prompt | llm
        result_text = await chain.ainvoke({
            "resume_json": json.dumps(resume_context, ensure_ascii=False),
            "target_position": target_position,
            "qa_text": _build_qa_text(qa_data),
        })
        content = result_text.content if hasattr(result_text, "content") else str(result_text)
        report = extract_json(content)
    except Exception as e:
        logger.error(f"报告生成失败: {e}")
        report = {"summary": "报告生成失败", "strengths": [], "weaknesses": [], "suggestions": []}

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


def _build_qa_text(qa_data: list[dict]) -> str:
    """构建问答文本供 LLM 报告生成"""
    text = ""
    for item in qa_data:
        text += (
            f"问题：{item['question']}\n"
            f"回答：{item.get('answer', '未回答')}\n"
            f"得分：{item.get('score', 'N/A')}\n\n"
        )
    return text
