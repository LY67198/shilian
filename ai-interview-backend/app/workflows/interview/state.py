"""Interview workflow state definition"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.workflows._shared.state_base import BaseWorkflowState


class ScoreResult(BaseModel):
    """评分结构化输出"""
    score: float = Field(default=5.0, ge=0, le=10, description="评分 0-10")
    feedback: str = Field(default="", description="简短评语，50字以内")
    follow_up: bool = Field(default=False, description="是否需要追问")


class InterviewState(BaseWorkflowState, total=False):
    """面试 workflow state

    继承 BaseWorkflowState 的 thread_id / retry_count / errors / messages 字段。
    answer 字段通过 Command(resume={...}) 注入，不持久化到 checkpoint。
    """

    # ── 上下文（fetch_context 从 DB 读取并写入）──
    interview_id: int
    user_id: int
    target_position: str
    difficulty: str
    total_questions: int
    current_index: int
    questions: list[dict]          # 全部题目 [{question, bank_id, reference_answer, key_points, ...}]
    current_question: str           # 当前题目文本
    reference_answer: Optional[str] # 当前题参考答案（题库携带）
    key_points: Optional[list]      # 当前题采分点
    resume_context: dict            # 解析后的简历 JSON
    chat_history: list[dict]        # [{role, content}, ...]，含最近消息

    # ── 用户输入（Command(resume={...}) 注入，不持久化）──
    answer: str
    stream: bool

    # ── 知识库（retrieve_knowledge 写入）──
    knowledge_context: list[str]

    # ── 评分（evaluate 写入）──
    score: float
    feedback: str

    # ── 报告（generate_report 写入）──
    all_scores: list[dict]          # [{question, answer, score, feedback}]
    overall_score: float
    report: Optional[dict]
    is_finished: bool

    # ── ask_question 产出（流式推送用）──
    next_question: Optional[str]
    index: Optional[int]
