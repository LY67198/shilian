import json
import logging
from typing import Dict, List

from sqlalchemy import select, delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.exceptions.http_exceptions import NotFoundError, ValidationError
from app.models.interview import Interview
from app.models.interview_message import InterviewMessage
from app.models.resume import Resume
from app.services.backoffice.question_bank_service import QuestionBankService
from app.services.client.ai_service import AIService

logger = logging.getLogger(__name__)


def _build_retrieval_query(target_position: str, parsed_resume: dict) -> str:
    """从岗位 + 简历技能构造题库检索 query"""
    skills = parsed_resume.get("skills") or []
    #如果是列表，只取前 8 个技能并拼接
    if isinstance(skills, list):
        top_skills = " ".join(str(s) for s in skills[:8])
    else:
        #如果不是列表，就截取前 200 个字符
        top_skills = str(skills)[:200]
    return f"{target_position} {top_skills}".strip()


class InterviewService:

    @staticmethod
    async def _generate_questions_with_rag(
        db: AsyncSession,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        total_questions: int,
    ) -> list:
        """
        RAG 出题核心：
        1. 用岗位 + 简历技能构造 query，向题库做向量检索
        2. 根据召回数量分支：
           - 充分（>= total）→ AI 选题 + 微调（select_and_adapt_questions）
           - 不足（0 < cnt < total）→ AI 兜底补全（generate_with_seeds）
           - 为空（=0）→ 完全 AI 生成（fallback to legacy generate_questions）
        """
        query = _build_retrieval_query(target_position, parsed_resume)
        recall_k = total_questions * settings.QUESTION_BANK_RECALL_FACTOR

        # 题库 RAG 检索包裹 try/except：DashScope embedding 服务或 pgvector 出错时
        # 降级到纯 AI 生成，避免面试启动直接崩
        try:
            candidates = await QuestionBankService.retrieve_questions(
                query=query,
                db=db,
                k=recall_k,
                position_tag=target_position,
                difficulty=difficulty,
                min_score=settings.QUESTION_BANK_MIN_SCORE,
            )
            # 召回失败时兜底：放宽 position_tag 限制再试一次（命中率更高）
            if len(candidates) < total_questions:
                relaxed = await QuestionBankService.retrieve_questions(
                    query=query,
                    db=db,
                    k=recall_k,
                    position_tag=None,
                    difficulty=difficulty,
                    min_score=settings.QUESTION_BANK_MIN_SCORE,
                )
                seen = {c["id"] for c in candidates}
                for c in relaxed:
                    if c["id"] not in seen:
                        candidates.append(c)
        except Exception as e:
            logger.error(f"[RAG出题] 题库检索异常，降级到纯 AI 生成: {e}")
            candidates = []

        cnt = len(candidates)
        logger.info(f"[RAG出题] 题库召回 {cnt} 题，目标 {total_questions} 题")

        if cnt >= total_questions:
            logger.info(f"[RAG出题] 走【题库充分】分支")
            questions = await AIService.select_and_adapt_questions(
                candidates=candidates,
                parsed_resume=parsed_resume,
                target_position=target_position,
                difficulty=difficulty,
                target_n=total_questions,
            )
        elif cnt > 0:
            logger.info(f"[RAG出题] 走【AI 兜底补全】分支（题库 {cnt} 题 + AI 补 {total_questions - cnt} 题）")
            questions = await AIService.generate_with_seeds(
                seed_questions=candidates,
                parsed_resume=parsed_resume,
                target_position=target_position,
                difficulty=difficulty,
                target_n=total_questions,
            )
        else:
            logger.warning(f"[RAG出题] 题库为空，走【纯 AI 生成】兜底分支")
            questions = await AIService.generate_questions(
                parsed_resume=parsed_resume,
                target_position=target_position,
                difficulty=difficulty,
                count=total_questions,
            )
            # 给所有题打上 source 标记
            for q in questions:
                q.setdefault("source", "ai_fallback")
                q.setdefault("bank_id", None)

        return questions

    @staticmethod
    async def start_interview(
        db: AsyncSession,
        user_id: int,
        resume_id: int,
        target_position: str,
        difficulty: str,
        total_questions: int
    ) -> Dict:
        """开始新的面试会话"""
        # 验证简历是否存在且已解析
        query = select(Resume).where(
            Resume.id == resume_id,
            Resume.user_id == user_id
        )
        result = await db.execute(query)
        resume = result.scalar_one_or_none()

        if not resume:
            raise NotFoundError(message="简历不存在")
        if resume.status != "completed":
            raise ValidationError(message="简历尚未解析完成")

        try:
            parsed_resume = json.loads(resume.parsed_content)
        except json.JSONDecodeError:
            logger.error(f"简历 parsed_content 不是有效 JSON: resume_id={resume.id}")
            raise ValidationError(message="简历数据异常，请重新上传")

        # ── RAG 出题流程：题库召回优先 + AI 兜底 ──────────────────────
        questions = await InterviewService._generate_questions_with_rag(
            db=db,
            parsed_resume=parsed_resume,
            target_position=target_position,
            difficulty=difficulty,
            total_questions=total_questions,
        )

        # 题库选中题目，累加 use_count
        bank_ids = [q.get("bank_id") for q in questions if q.get("bank_id")]
        if bank_ids:
            await QuestionBankService.increment_use_count(db, bank_ids)

        # 创建面试记录
        interview = Interview(
            user_id=user_id,
            resume_id=resume_id,
            target_position=target_position,
            difficulty=difficulty,
            total_questions=total_questions,
            current_question_index=0,
            questions_data=questions,
            status="in_progress"
        )
        db.add(interview)
        await db.commit()
        await db.refresh(interview)

        # 保存第一道题作为面试官消息
        first_question = questions[0]["question"]
        msg = InterviewMessage(
            interview_id=interview.id,
            role="interviewer",
            content=first_question,
            question_index=0
        )
        db.add(msg)
        await db.commit()

        return {
            "interview_id": interview.id,
            "first_question": first_question,
            "question_index": 0,
            "total_questions": total_questions
        }

    @staticmethod
    async def get_report(
        db: AsyncSession,
        user_id: int,
        interview_id: int
    ) -> Dict:
        """获取面试评估报告"""
        query = select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user_id
        )
        result = await db.execute(query)
        interview = result.scalar_one_or_none()

        if not interview:
            raise NotFoundError(message="面试记录不存在")
        if interview.status != "completed":
            raise ValidationError(message="面试尚未完成")

        report = {}
        if interview.report:
            try:
                report = json.loads(interview.report)
            except json.JSONDecodeError:
                report = {}

        return {
            "interview_id": interview.id,
            "overall_score": float(interview.overall_score) if interview.overall_score else 0,
            "total_questions": interview.total_questions,
            "report": report
        }

    @staticmethod
    async def get_interviews(
        db: AsyncSession,
        user_id: int
    ) -> Dict:
        """获取用户的所有面试记录"""
        query = select(Interview).where(
            Interview.user_id == user_id
        ).order_by(Interview.created_at.desc())
        result = await db.execute(query)
        interviews = result.scalars().all()

        items = [
            {
                "interview_id": i.id,
                "target_position": i.target_position,
                "difficulty": i.difficulty,
                "overall_score": float(i.overall_score) if i.overall_score else None,
                "total_questions": i.total_questions,
                "status": i.status,
                "created_at": i.created_at.isoformat() if i.created_at else None
            }
            for i in interviews
        ]

        return {
            "total": len(items),
            "items": items
        }

    @staticmethod
    async def get_interview_messages(
        db: AsyncSession,
        user_id: int,
        interview_id: int
    ) -> List[Dict]:
        """获取面试的所有对话消息"""
        # 验证面试归属权
        interview_query = select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user_id
        )
        interview_result = await db.execute(interview_query)
        interview = interview_result.scalar_one_or_none()
        if not interview:
            raise NotFoundError(message="面试记录不存在")

        query = select(InterviewMessage).where(
            InterviewMessage.interview_id == interview_id
        ).order_by(InterviewMessage.id)
        result = await db.execute(query)
        messages = result.scalars().all()

        return [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "question_index": m.question_index,
                "score": float(m.score) if m.score else None,
                "feedback": m.feedback,
                "created_at": m.created_at.isoformat() if m.created_at else None
            }
            for m in messages
        ]

    @staticmethod
    async def delete_interview(
        db: AsyncSession,
        user_id: int,
        interview_id: int
    ) -> Dict:
        """删除面试记录及其关联的对话消息"""
        query = select(Interview).where(
            Interview.id == interview_id,
            Interview.user_id == user_id
        )
        result = await db.execute(query)
        interview = result.scalar_one_or_none()

        if not interview:
            raise NotFoundError(message="面试记录不存在")

        # 先删除关联的对话消息
        await db.execute(
            sql_delete(InterviewMessage).where(InterviewMessage.interview_id == interview_id)
        )
        # 再删除面试记录
        await db.delete(interview)
        await db.commit()

        return {"message": "面试记录已删除"}

    @staticmethod
    async def delete_interview_admin(
        db: AsyncSession,
        interview_id: int
    ) -> Dict:
        """管理员删除面试记录（不校验用户归属）"""

        interview = await db.get(Interview, interview_id)
        if not interview:
            raise NotFoundError(message="面试记录不存在")

        await db.execute(
            sql_delete(InterviewMessage).where(InterviewMessage.interview_id == interview_id)
        )
        await db.delete(interview)
        await db.commit()

        return {"message": "面试记录已删除"}


interview_service = InterviewService()
