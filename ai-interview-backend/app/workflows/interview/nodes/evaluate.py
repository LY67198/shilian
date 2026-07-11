"""evaluate node — 委托 EvaluatorAgent 评分"""
from __future__ import annotations

import logging

from app.agents.evaluator_agent import EvaluatorAgent
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def evaluate_node(state: InterviewState) -> dict:
    """评估候选人回答，委托 EvaluatorAgent"""
    custom = state.get("custom") or {}
    agent: EvaluatorAgent = custom.get("evaluator_agent")

    if agent is None:
        logger.warning("evaluator_agent not found in state.custom, using default")
        agent = EvaluatorAgent()

    result = await agent.evaluate(
        question=state.get("current_question", ""),
        answer=state.get("answer", ""),
        resume_context=state.get("resume_context", {}),
        chat_history=state.get("chat_history", []),
        reference_answer=state.get("reference_answer"),
        key_points=state.get("key_points"),
        knowledge_context=state.get("knowledge_context", []),
    )

    return {
        "score": result.score,
        "feedback": result.feedback,
    }
