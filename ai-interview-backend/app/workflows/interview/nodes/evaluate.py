"""evaluate node — LLM 评分（with_structured_output）"""
from __future__ import annotations

import json
import logging

from app.llm import get_chat_llm
from app.llm.prompts import load_prompt
from app.workflows.interview.state import InterviewState, ScoreResult

logger = logging.getLogger(__name__)


async def evaluate_node(state: InterviewState) -> dict:
    """评估候选人回答，统一走 with_structured_output 拿 ScoreResult"""
    question = state.get("current_question", "")
    answer = state.get("answer", "")
    resume_context = state.get("resume_context", {})
    chat_history = state.get("chat_history", [])
    reference_answer = state.get("reference_answer")
    key_points = state.get("key_points")
    knowledge_context = state.get("knowledge_context", [])

    # 构造对话历史文本
    history_text = ""
    for msg in chat_history[-6:]:
        role = "面试官" if msg["role"] == "interviewer" else "候选人"
        history_text += f"{role}: {msg['content']}\n"

    # 参考依据块
    ref_block = ""
    if reference_answer:
        ref_block += f"\n【参考答案要点（评分依据，不要直接读给候选人）】：\n{reference_answer}\n"
    if key_points:
        ref_block += f"\n【关键采分点】：{json.dumps(key_points, ensure_ascii=False)}\n"

    kb_block = ""
    if knowledge_context:
        kb_block = (
            "\n【相关知识库片段（评分参考，不要直接读给候选人）】：\n"
            + "\n---\n".join(knowledge_context)
            + "\n"
        )

    scoring_hint = (
        "评分时请对照【参考答案要点】与【相关知识库片段】，候选人答中要点越多分越高。\n"
        if (ref_block or kb_block)
        else ""
    )

    try:
        prompt = load_prompt("evaluate_answer")
        llm = get_chat_llm(temperature=0.5)
        structured_llm = llm.with_structured_output(ScoreResult)
        chain = prompt | structured_llm
        result: ScoreResult = await chain.ainvoke({
            "question": question,
            "answer": answer,
            "resume_json": json.dumps(resume_context, ensure_ascii=False),
            "history_text": history_text,
            "ref_block": ref_block,
            "kb_block": kb_block,
            "scoring_hint": scoring_hint,
        })
        return {
            "score": result.score,
            "feedback": result.feedback,
        }
    except Exception as e:
        logger.error(f"结构化评分失败，返回兜底: {e}")
        return {
            "score": 5.0,
            "feedback": f"评分异常，已记录: {str(e)[:100]}",
        }
