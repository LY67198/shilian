"""FastAPI Depends 依赖注入声明

集中管理所有 service 和 agent 的工厂函数，供 API 层和 workflow nodes 使用。
"""

from app.repositories.question_bank_repo import question_bank_repo


def get_question_agent():
    from app.agents.question_agent import QuestionAgent
    return QuestionAgent(question_bank_repo=question_bank_repo)


def get_evaluator_agent():
    from app.agents.evaluator_agent import EvaluatorAgent
    return EvaluatorAgent()


def get_report_agent():
    from app.agents.report_agent import ReportAgent
    return ReportAgent()
