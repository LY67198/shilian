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


# ── Service factories ──

def get_ai_service():
    from app.services.client.ai_service import AIService
    return AIService()

def get_client_auth_service():
    from app.services.client.auth import ClientAuthService
    return ClientAuthService()

def get_interview_service():
    from app.services.client.interview_service import InterviewService
    return InterviewService()

def get_position_agent_service():
    from app.services.client.position_agent_service import PositionAgentService
    return PositionAgentService()

def get_redis_verification_service():
    from app.services.client.redis_verification import RedisVerificationService
    return RedisVerificationService()

def get_resume_service():
    from app.services.client.resume_service import ResumeService
    return ResumeService()

def get_client_waiting_list_service():
    from app.services.client.waiting_list import WaitingListService
    return WaitingListService()

def get_email_template_service():
    from app.services.client.email_templates import ClientEmailService
    return ClientEmailService()

def get_admin_service():
    from app.services.backoffice.admin import AdminService
    return AdminService()

def get_backoffice_auth_service():
    from app.services.backoffice.auth import BackofficeAuthService
    return BackofficeAuthService()

def get_knowledge_service():
    from app.services.backoffice.knowledge_service import KnowledgeService
    return KnowledgeService()

def get_position_template_service():
    from app.services.backoffice.position_template_service import PositionTemplateService
    return PositionTemplateService()

def get_question_bank_service():
    from app.services.backoffice.question_bank_service import QuestionBankService
    return QuestionBankService()

def get_backoffice_waiting_list_service():
    from app.services.backoffice.waiting_list import WaitingListService
    return WaitingListService()
