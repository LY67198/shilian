"""EvaluatorAgent — 答案评分 Agent"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from app.agents.base import BaseAgent

if TYPE_CHECKING:
    from app.workflows.interview.state import ScoreResult

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """评分 Agent — 结构化输出 ScoreResult"""

    def __init__(self):
        """初始化评估 Agent，使用 evaluator_agent prompt 和较低温度以保证评分一致性。"""
        super().__init__(prompt_name="evaluator_agent", temperature=0.3)

    async def evaluate(
        self,
        question: str,
        answer: str,
        resume_context: dict,
        chat_history: list,
        reference_answer: str | None = None,
        key_points: list | None = None,
        knowledge_context: list | None = None,
    ) -> "ScoreResult":
        """评估候选人的回答

        Returns:
            ScoreResult: score (0-10), feedback (str), follow_up (bool)
        """
        from app.workflows.interview.state import ScoreResult  # noqa: F811

        # 构造对话历史文本
        history_text = ""
        for msg in (chat_history or [])[-6:]:
            role = "面试官" if msg.get("role") == "interviewer" else "候选人"
            history_text += f"{role}: {msg.get('content', '')}\n"

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
            result = await self.invoke_structured(
                variables={
                    "question": question,
                    "answer": answer,
                    "resume_json": json.dumps(resume_context, ensure_ascii=False),
                    "history_text": history_text,
                    "ref_block": ref_block,
                    "kb_block": kb_block,
                    "scoring_hint": scoring_hint,
                },
                schema=ScoreResult,
            )
            return result
        except Exception as e:
            logger.error(f"结构化评分失败，返回兜底: {e}")
            return ScoreResult(
                score=5.0,
                feedback=f"评分异常，已记录: {str(e)[:100]}",
                follow_up=False,
            )
