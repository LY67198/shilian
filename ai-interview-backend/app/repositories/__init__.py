"""数据访问层（Repository Pattern）

Phase 4 补全：question_bank_repo + knowledge_repo + interview_repo 补全。

## 规则
1. **纯 DB 操作**：repository 函数不调 LLM / Milvus / Redis
2. **不抛 HTTPException**：用普通 Python exceptions 或返回 None
3. **事务边界**：repository 函数不 commit()，由调用方管事务
4. **async 优先**：所有 DB 操作 async
5. **类型提示**：函数签名必须明确返回类型
"""
from app.repositories.base import BaseRepository
from app.repositories.interview_repo import InterviewRepository, interview_repo
from app.repositories.question_bank_repo import (
    QuestionBankRepository,
    question_bank_repo,
)
from app.repositories.knowledge_repo import KnowledgeRepository, knowledge_repo

__all__ = [
    "BaseRepository",
    "InterviewRepository",
    "QuestionBankRepository",
    "KnowledgeRepository",
    "interview_repo",
    "question_bank_repo",
    "knowledge_repo",
]
