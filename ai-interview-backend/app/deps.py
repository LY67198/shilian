"""FastAPI Depends 依赖注入声明

集中管理所有 service 和 agent 的工厂函数，供 API 层和 workflow nodes 使用。
"""

from app.agents.question_agent import QuestionAgent
from app.agents.evaluator_agent import EvaluatorAgent
from app.agents.report_agent import ReportAgent


def get_question_agent() -> QuestionAgent:
    return QuestionAgent()


def get_evaluator_agent() -> EvaluatorAgent:
    return EvaluatorAgent()


def get_report_agent() -> ReportAgent:
    return ReportAgent()
