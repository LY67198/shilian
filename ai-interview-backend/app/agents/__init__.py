"""Agent 类 — LLM 调用封装层

每个 Agent 封装一组 LLM 交互：prompt + tools + temperature + 结构化输出。
Agent 类位于 LangChain 层（app/agents/），被 workflow nodes 调用，
不直接依赖 LangGraph。
"""

from app.agents.base import BaseAgent
from app.agents.question_agent import QuestionAgent
from app.agents.evaluator_agent import EvaluatorAgent
from app.agents.report_agent import ReportAgent

__all__ = ["BaseAgent", "QuestionAgent", "EvaluatorAgent", "ReportAgent"]
