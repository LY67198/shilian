# Phase 4 — 多 Agent + 可观测性 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将面试流程中的 LLM 调用提取为独立 Agent 类，补全 repository 层，增加 RAG 链路 trace，接入 LangSmith 持续评估，全量 service 去 @staticmethod。

**Architecture:** 保持 6-node HITL StateGraph 不变，新建 `app/agents/`（Agent 类）+ 补 `app/repositories/`（2 新 + 1 改）。Agent 通过 `state.custom` 注入到 node，node 内部从直接调 LLM 改为委托 Agent。Tracing 在 `app/retrieval/pipeline.py` 4 步各包 span。评估通过 `eval/scripts/upload_golden_set.py` 上传 LangSmith + `eval_ragas.py --upload`。

**Tech Stack:** LangChain 1.3, LangGraph 1.2.8, FastAPI Depends, rank-bm25, Milvus 2.4, RAGAS, LangSmith SDK

**变更量:** 8 新建 + ~24 修改，~430 删 / ~840 增

---

### Task 1: BaseAgent 基类 + deps.py 依赖声明

**Files:**
- Create: `ai-interview-backend/app/agents/__init__.py`
- Create: `ai-interview-backend/app/agents/base.py`
- Create: `ai-interview-backend/app/deps.py`

- [ ] **Step 1: Create agents package __init__.py**

```python
"""Agent 类 — LLM 调用封装层

每个 Agent 封装一组 LLM 交互：prompt + tools + temperature + 结构化输出。
Agent 类位于 LangChain 层（app/agents/），被 workflow nodes 调用，
不直接依赖 LangGraph。
"""

from app.agents.base import BaseAgent

__all__ = ["BaseAgent"]
```

- [ ] **Step 2: Write BaseAgent**

`ai-interview-backend/app/agents/base.py`:

```python
"""BaseAgent — Agent 基类

提供 LLM + prompt 公共逻辑，子类只需指定 prompt_name + temperature + tools。
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from app.llm import get_chat_llm
from app.llm.prompts import load_prompt

logger = logging.getLogger(__name__)


class BaseAgent:
    """Agent 基类

    Usage:
        class EvaluatorAgent(BaseAgent):
            def __init__(self):
                super().__init__(
                    prompt_name="evaluator_agent",
                    temperature=0.3,
                )

            async def evaluate(self, ...) -> ScoreResult:
                return await self.invoke_structured(vars, ScoreResult)
    """

    def __init__(
        self,
        prompt_name: str,
        temperature: float = 0.5,
        tools: Optional[List] = None,
    ):
        self._prompt_name = prompt_name
        self._temperature = temperature
        self.tools = tools or []
        self._llm = get_chat_llm(temperature=self._temperature)
        self._prompt = load_prompt(prompt_name)

    @property
    def prompt_name(self) -> str:
        return self._prompt_name

    @property
    def temperature(self) -> float:
        return self._temperature

    async def invoke(self, variables: dict) -> str:
        """prompt | llm → 纯文本输出"""
        chain = self._prompt | self._llm
        result = await chain.ainvoke(variables)
        return result.content if hasattr(result, "content") else str(result)

    async def invoke_structured(self, variables: dict, schema: Type[BaseModel]) -> BaseModel:
        """prompt | llm.with_structured_output(schema) → Pydantic 对象"""
        structured_llm = self._llm.with_structured_output(schema)
        chain = self._prompt | structured_llm
        return await chain.ainvoke(variables)
```

- [ ] **Step 3: Create deps.py**

`ai-interview-backend/app/deps.py`:

```python
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
```

Note: Step 3 中 import 的 Agent 类尚未创建，但 deps.py 仅需 import 声明。后续 Task 3 创建 Agent 类后 import 即生效。

- [ ] **Step 4: Verify agent package is importable**

```bash
docker exec shilian-app python -c "from app.agents import BaseAgent; print('BaseAgent imported OK')"
```

Expected: `BaseAgent imported OK`

- [ ] **Step 5: Commit**

```bash
git add ai-interview-backend/app/agents/__init__.py ai-interview-backend/app/agents/base.py ai-interview-backend/app/deps.py
git commit -m "feat: add BaseAgent + deps.py for Phase 4 agent layer

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Repository 补全 — question_bank_repo + knowledge_repo + interview_repo

**Files:**
- Create: `ai-interview-backend/app/repositories/question_bank_repo.py`
- Create: `ai-interview-backend/app/repositories/knowledge_repo.py`
- Modify: `ai-interview-backend/app/repositories/__init__.py`
- Modify: `ai-interview-backend/app/repositories/interview_repo.py`

- [ ] **Step 1: Write question_bank_repo.py**

```python
"""QuestionBank Repository — 题库数据访问层"""
from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question_bank import QuestionBank
from app.repositories.base import BaseRepository


class QuestionBankRepository(BaseRepository[QuestionBank]):
    model = QuestionBank

    async def search_by_position(
        self,
        db: AsyncSession,
        position_tag: str,
        difficulty: Optional[str] = None,
        limit: int = 10,
    ) -> List[QuestionBank]:
        """按岗位标签 + 难度（可选）搜索活跃题目"""
        stmt = select(QuestionBank).where(
            QuestionBank.is_active == True,
            QuestionBank.position_tag == position_tag,
        )
        if difficulty:
            stmt = stmt.where(QuestionBank.difficulty == difficulty)
        stmt = stmt.order_by(QuestionBank.use_count.asc()).limit(limit)
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def list_by_position(
        self,
        db: AsyncSession,
        position_tag: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[List[QuestionBank], int]:
        """分页列出某一岗位的题目"""
        base = select(QuestionBank).where(
            QuestionBank.is_active == True,
            QuestionBank.position_tag == position_tag,
        )
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await db.execute(count_stmt)
        total_count = int(total.scalar_one())

        stmt = (
            base
            .order_by(QuestionBank.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all()), total_count

    async def increment_use_count(self, db: AsyncSession, question_id: int) -> None:
        """增加题目使用次数"""
        q = await self.get_by_id(db, question_id)
        if q:
            q.use_count = (q.use_count or 0) + 1
            await db.flush()


question_bank_repo = QuestionBankRepository()
```

- [ ] **Step 2: Write knowledge_repo.py**

```python
"""Knowledge Repository — 知识库数据访问层 (PG 元数据)"""
from __future__ import annotations

from typing import List, Optional, Tuple

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge import KnowledgeChunk
from app.repositories.base import BaseRepository


class KnowledgeRepository(BaseRepository[KnowledgeChunk]):
    model = KnowledgeChunk

    async def list_by_document(
        self,
        db: AsyncSession,
        document_id: int,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[KnowledgeChunk], int]:
        """分页列出某文档的所有 chunk"""
        base = select(KnowledgeChunk).where(
            KnowledgeChunk.document_id == document_id,
        )
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await db.execute(count_stmt)
        total_count = int(total.scalar_one())

        stmt = (
            base
            .order_by(KnowledgeChunk.chunk_index.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all()), total_count

    async def get_by_chunk_id(
        self, db: AsyncSession, chunk_id: int
    ) -> Optional[KnowledgeChunk]:
        return await self.get_by_id(db, chunk_id)

    async def full_text_search(
        self,
        db: AsyncSession,
        query: str,
        limit: int = 20,
    ) -> List[KnowledgeChunk]:
        """PG ILIKE 全文搜索（非向量检索，用于辅助查询）"""
        pattern = f"%{query}%"
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.content.ilike(pattern))
            .limit(limit)
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())


knowledge_repo = KnowledgeRepository()
```

- [ ] **Step 3: Add get_by_id_with_messages and delete_cascade to interview_repo**

Append to `ai-interview-backend/app/repositories/interview_repo.py` (after line 139, before the module-end):

```python
    async def get_by_id_with_messages(
        self, db: AsyncSession, interview_id: int
    ) -> Optional[Interview]:
        """Eager load interview + messages + resume，一次查询"""
        from sqlalchemy.orm import selectinload

        stmt = (
            select(Interview)
            .options(
                selectinload(Interview.messages),
                selectinload(Interview.resume),
            )
            .where(Interview.id == interview_id)
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_cascade(self, db: AsyncSession, interview_id: int) -> bool:
        """删除 interview + 关联 messages（事务内完成）"""
        from sqlalchemy import delete as sa_delete

        from app.models.interview_message import InterviewMessage

        interview = await self.get_by_id(db, interview_id)
        if interview is None:
            return False

        await db.execute(
            sa_delete(InterviewMessage).where(
                InterviewMessage.interview_id == interview_id
            )
        )
        await db.delete(interview)
        await db.flush()
        return True
```

- [ ] **Step 4: Update repositories __init__.py**

Replace the content of `ai-interview-backend/app/repositories/__init__.py`:

```python
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
```

- [ ] **Step 5: Verify repos are importable**

```bash
docker exec shilian-app python -c "from app.repositories import question_bank_repo, knowledge_repo; print('Repos OK')"
```

Expected: `Repos OK`

- [ ] **Step 6: Commit**

```bash
git add ai-interview-backend/app/repositories/question_bank_repo.py \
        ai-interview-backend/app/repositories/knowledge_repo.py \
        ai-interview-backend/app/repositories/__init__.py \
        ai-interview-backend/app/repositories/interview_repo.py
git commit -m "feat: add question_bank_repo, knowledge_repo, interview_repo extensions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Agent 类实现

**Files:**
- Create: `ai-interview-backend/app/agents/question_agent.py`
- Create: `ai-interview-backend/app/agents/evaluator_agent.py`
- Create: `ai-interview-backend/app/agents/report_agent.py`
- Create: `ai-interview-backend/app/prompts/question_agent.yaml`
- Create: `ai-interview-backend/app/prompts/evaluator_agent.yaml`
- Create: `ai-interview-backend/app/prompts/report_agent.yaml`
- Modify: `ai-interview-backend/app/agents/__init__.py`

- [ ] **Step 1: Write QuestionAgent YAML prompt**

`ai-interview-backend/app/prompts/question_agent.yaml`:

```yaml
# QuestionAgent — 面试出题提示词
version: 1
temperature: 0.9
response_format: json
system: |
  你是一位资深的面试出题专家。根据候选人的简历、目标岗位和难度要求，生成高质量的面试题目。

  出题原则：
  1. 题目应能考察候选人的真实技术水平
  2. 避免过于宽泛或过于冷门的知识点
  3. 结合简历中的项目经历，出针对性题目
  4. 每题应有明确的采分点（key_points）

  返回纯 JSON 数组，每个题目对象包含：
  - question: 题目文本
  - reference_answer: 参考答案要点
  - key_points: [采分点列表]
  - difficulty: 难度等级 (easy/medium/hard)
user_template: |
  目标岗位：{position}
  难度要求：{difficulty}
  题目数量：{count}

  候选人简历摘要：
  {resume_json}

  现有题库可选题（优先使用题库题目，不足时再生成新题）：
  {bank_questions}

  请生成 {count} 道面试题。优先从题库中选择合适的题目，题库不足时补充生成新题。
```

- [ ] **Step 2: Write EvaluatorAgent YAML prompt**

`ai-interview-backend/app/prompts/evaluator_agent.yaml`:

```yaml
# EvaluatorAgent — 答案评分提示词
version: 1
temperature: 0.3
response_format: json
system: |
  你是一位严格的面试评分专家。根据候选人的回答、参考答案要点和知识库内容，给出公正的评分。

  评分维度（1-10 分）：
  - 准确性：回答是否抓住了技术要点（40%）
  - 深度：是否有深入的理解和扩展（30%）
  - 表达：逻辑是否清晰，表达是否准确（20%）
  - 实践：是否能结合工程经验（10%）

  评分标准：
  - 9-10：回答准确且深入，有深刻见解
  - 7-8：大部分命中要点，表达清晰
  - 5-6：命中部分要点，有遗漏或小错误
  - 3-4：理解偏差较大，要点大部分未命中
  - 1-2：完全答非所问或敷衍

  返回纯 JSON：
  {
    "score": <float 0-10>,
    "feedback": "<简短评语，50字以内>",
    "follow_up": <bool 是否需要追问>
  }
user_template: |
  题目：{question}

  候选人回答：{answer}

  简历背景：
  {resume_json}

  对话历史（最近几轮）：
  {history_text}

  {ref_block}

  {kb_block}

  {scoring_hint}
```

- [ ] **Step 3: Write ReportAgent YAML prompt**

`ai-interview-backend/app/prompts/report_agent.yaml`:

```yaml
# ReportAgent — 面试报告生成提示词
version: 1
temperature: 0.5
response_format: json
system: |
  你是一位资深的面试评估专家。根据候选人所有题目的回答情况和评分，生成一份全面的面试评估报告。

  返回纯 JSON：
  {
    "summary": "<摘要，100字以内>",
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["不足1", "不足2"],
    "suggestions": ["改进建议1", "改进建议2"],
    "hire_recommendation": "<一点评价>"
  }

  请严格返回 JSON，不要包含任何其他内容。
user_template: |
  目标岗位：{target_position}

  候选人简历：
  {resume_json}

  面试问答记录：
  {qa_text}

  请生成面试评估报告。
```

- [ ] **Step 4: Write QuestionAgent**

`ai-interview-backend/app/agents/question_agent.py`:

```python
"""QuestionAgent — 面试出题 Agent"""
from __future__ import annotations

import json
import logging
from typing import List

from pydantic import BaseModel, Field

from app.agents.base import BaseAgent
from app.repositories.question_bank_repo import QuestionBankRepository

logger = logging.getLogger(__name__)


class QuestionItem(BaseModel):
    question: str
    reference_answer: str = ""
    key_points: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    bank_id: int | None = None  # None 表示 LLM 生成的非题库题


class QuestionAgent(BaseAgent):
    """出题 Agent — 优先题库 RAG，不足时 LLM 兜底"""

    def __init__(self, question_bank_repo: QuestionBankRepository | None = None):
        super().__init__(prompt_name="question_agent", temperature=0.9)
        self._bank_repo = question_bank_repo

    async def generate(
        self,
        db,
        position: str,
        difficulty: str,
        count: int,
        resume: dict,
    ) -> List[dict]:
        """生成 N 道题：先选题库，不足 LLM 兜底。

        Returns:
            list[dict]: [{"question": ..., "reference_answer": ..., "key_points": [...], ...}]
        """
        bank_qs = await self._select_from_bank(db, position, difficulty, count)

        if len(bank_qs) >= count:
            logger.info(f"题库命中 {len(bank_qs)} 道，满足 {count} 题需求")
            return bank_qs[:count]

        shortage = count - len(bank_qs)
        logger.info(f"题库命中 {len(bank_qs)} 道，缺 {shortage} 道，LLM 兜底生成")

        generated = await self._fallback_generate(
            position=position,
            difficulty=difficulty,
            count=shortage,
            resume=resume,
            bank_questions=bank_qs,
        )

        return bank_qs + generated

    async def _select_from_bank(
        self, db, position: str, difficulty: str, count: int
    ) -> List[dict]:
        """从题库 RAG 检索 + 选择"""
        if self._bank_repo is None:
            return []

        try:
            questions = await self._bank_repo.search_by_position(
                db, position_tag=position, difficulty=difficulty, limit=count
            )
        except Exception as e:
            logger.warning(f"题库查询失败: {e}")
            return []

        result = []
        for q in questions:
            result.append({
                "question": q.question,
                "reference_answer": q.reference_answer or "",
                "key_points": q.key_points or [],
                "difficulty": q.difficulty or difficulty,
                "bank_id": q.id,
            })
        return result

    async def _fallback_generate(
        self,
        position: str,
        difficulty: str,
        count: int,
        resume: dict,
        bank_questions: list,
    ) -> List[dict]:
        """LLM 兜底生成新题"""
        try:
            result = await self.invoke_structured(
                variables={
                    "position": position,
                    "difficulty": difficulty,
                    "count": str(count),
                    "resume_json": json.dumps(resume, ensure_ascii=False),
                    "bank_questions": json.dumps(bank_questions, ensure_ascii=False),
                },
                schema=QuestionItem,
            )
            return [_item_to_dict(result)]
        except Exception as e:
            logger.error(f"LLM 兜底出题失败: {e}")
            # 返回一个基础题
            return [{
                "question": f"请介绍你在{position}岗位上的核心技术能力",
                "reference_answer": "",
                "key_points": [],
                "difficulty": difficulty,
                "bank_id": None,
            }]


def _item_to_dict(item: QuestionItem) -> dict:
    return {
        "question": item.question,
        "reference_answer": item.reference_answer,
        "key_points": item.key_points,
        "difficulty": item.difficulty,
        "bank_id": item.bank_id,
    }
```

- [ ] **Step 5: Write EvaluatorAgent**

`ai-interview-backend/app/agents/evaluator_agent.py`:

```python
"""EvaluatorAgent — 答案评分 Agent"""
from __future__ import annotations

import json
import logging

from app.agents.base import BaseAgent
from app.workflows.interview.state import ScoreResult

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """评分 Agent — 结构化输出 ScoreResult"""

    def __init__(self):
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
    ) -> ScoreResult:
        """评估候选人的回答

        Returns:
            ScoreResult: score (0-10), feedback (str), follow_up (bool)
        """
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
```

- [ ] **Step 6: Write ReportAgent**

`ai-interview-backend/app/agents/report_agent.py`:

```python
"""ReportAgent — 面试报告生成 Agent"""
from __future__ import annotations

import json
import logging

from app.agents.base import BaseAgent
from app.common.json_utils import extract_json

logger = logging.getLogger(__name__)


class ReportAgent(BaseAgent):
    """报告 Agent — 汇总所有评分，生成综合报告"""

    def __init__(self):
        super().__init__(prompt_name="report_agent", temperature=0.5)

    async def generate(
        self,
        resume_context: dict,
        target_position: str,
        qa_data: list[dict],
    ) -> dict:
        """生成面试评估报告

        Args:
            resume_context: 解析后的简历 JSON
            target_position: 目标岗位
            qa_data: [{question, answer, score}, ...]

        Returns:
            dict with summary, strengths, weaknesses, suggestions
        """
        qa_text = _build_qa_text(qa_data)

        try:
            result_text = await self.invoke({
                "resume_json": json.dumps(resume_context, ensure_ascii=False),
                "target_position": target_position,
                "qa_text": qa_text,
            })
            return extract_json(result_text)
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return {
                "summary": "报告生成失败",
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
            }


def _build_qa_text(qa_data: list[dict]) -> str:
    text = ""
    for item in qa_data:
        text += (
            f"问题：{item.get('question', '')}\n"
            f"回答：{item.get('answer', '未回答')}\n"
            f"得分：{item.get('score', 'N/A')}\n\n"
        )
    return text
```

- [ ] **Step 7: Update agents __init__.py**

Replace `ai-interview-backend/app/agents/__init__.py`:

```python
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
```

- [ ] **Step 8: Update deps.py with full imports**

Replace `ai-interview-backend/app/deps.py`:

```python
"""FastAPI Depends 依赖注入声明

集中管理所有 service 和 agent 的工厂函数，供 API 层和 workflow nodes 使用。
"""

from app.agents.question_agent import QuestionAgent
from app.agents.evaluator_agent import EvaluatorAgent
from app.agents.report_agent import ReportAgent
from app.repositories.question_bank_repo import question_bank_repo


def get_question_agent() -> QuestionAgent:
    return QuestionAgent(question_bank_repo=question_bank_repo)


def get_evaluator_agent() -> EvaluatorAgent:
    return EvaluatorAgent()


def get_report_agent() -> ReportAgent:
    return ReportAgent()
```

- [ ] **Step 9: Verify all agents are importable**

```bash
docker exec shilian-app python -c "
from app.agents import BaseAgent, QuestionAgent, EvaluatorAgent, ReportAgent
print('All agents imported OK')
"
```

Expected: `All agents imported OK`

- [ ] **Step 10: Commit**

```bash
git add ai-interview-backend/app/agents/ \
        ai-interview-backend/app/deps.py \
        ai-interview-backend/app/prompts/question_agent.yaml \
        ai-interview-backend/app/prompts/evaluator_agent.yaml \
        ai-interview-backend/app/prompts/report_agent.yaml
git commit -m "feat: add QuestionAgent, EvaluatorAgent, ReportAgent with YAML prompts

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Wire agents into workflow nodes

**Files:**
- Modify: `ai-interview-backend/app/workflows/interview/service.py`
- Modify: `ai-interview-backend/app/workflows/interview/nodes/evaluate.py`
- Modify: `ai-interview-backend/app/workflows/interview/nodes/generate_report.py`

- [ ] **Step 1: Inject agents into InterviewGraphService.submit_answer**

In `ai-interview-backend/app/workflows/interview/service.py`, modify the `state_data` construction (lines 62-72) to include agent instances:

```python
        state_data = {
            "answer": answer,
            "stream": stream,
            "custom": {
                "db": db,
                "milvus_client": milvus_client,
                "retrieval_check_service": _build_retrieval_check_service(
                    milvus_client, is_first_call
                ),
                "evaluator_agent": EvaluatorAgent(),
                "report_agent": ReportAgent(),
            },
        }
```

Add the import at the top of the file (after existing imports):

```python
from app.agents.evaluator_agent import EvaluatorAgent
from app.agents.report_agent import ReportAgent
```

- [ ] **Step 2: Rewrite evaluate node to use EvaluatorAgent**

Replace the content of `ai-interview-backend/app/workflows/interview/nodes/evaluate.py`:

```python
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
```

- [ ] **Step 3: Rewrite generate_report node to use ReportAgent**

Replace the content of `ai-interview-backend/app/workflows/interview/nodes/generate_report.py`:

```python
"""generate_report node — 委托 ReportAgent 生成报告"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.report_agent import ReportAgent
from app.repositories.interview_repo import interview_repo
from app.workflows.interview.state import InterviewState

logger = logging.getLogger(__name__)


async def generate_report_node(state: InterviewState) -> dict:
    """汇总所有评分，委托 ReportAgent 生成报告"""
    custom = state.get("custom") or {}
    db: AsyncSession = custom.get("db")
    if not db:
        raise RuntimeError("generate_report_node requires db session in state.custom.db")

    agent: ReportAgent = custom.get("report_agent")
    if agent is None:
        logger.warning("report_agent not found in state.custom, using default")
        agent = ReportAgent()

    interview_id = state["interview_id"]
    questions = state["questions"]
    resume_context = state.get("resume_context", {})
    target_position = state.get("target_position", "")

    scored_msgs = await interview_repo.get_scored_messages(db, interview_id)

    qa_data = []
    all_scores = []
    msg_by_idx = {m.question_index: m for m in scored_msgs}

    for i, q in enumerate(questions):
        m = msg_by_idx.get(i)
        score_val = float(m.score) if m and m.score else state.get("score", 0)
        answer_text = (
            m.content
            if m
            else (state.get("answer", "") if i == state.get("current_index", 0) else "未回答")
        )
        qa_data.append({
            "question": q["question"],
            "answer": answer_text,
            "score": score_val,
        })
        all_scores.append(score_val)

    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    report = await agent.generate(
        resume_context=resume_context,
        target_position=target_position,
        qa_data=qa_data,
    )

    report["question_scores"] = [
        {"question": qa["question"], "score": qa["score"], "feedback": ""}
        for qa in qa_data
    ]

    await interview_repo.update_result(db, interview_id, overall, report)
    await db.commit()

    return {
        "overall_score": overall,
        "report": report,
        "all_scores": qa_data,
        "is_finished": True,
    }
```

- [ ] **Step 4: Run existing tests to verify no regression**

```bash
docker exec shilian-app pytest -m "unit" -v
```

Expected: All unit tests pass (Phase 4 不改行为，所有测试应继续通过)

- [ ] **Step 5: Commit**

```bash
git add ai-interview-backend/app/workflows/interview/service.py \
        ai-interview-backend/app/workflows/interview/nodes/evaluate.py \
        ai-interview-backend/app/workflows/interview/nodes/generate_report.py
git commit -m "refactor: wire EvaluatorAgent and ReportAgent into interview nodes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: RAG 链路 Trace

**Files:**
- Modify: `ai-interview-backend/app/workflows/_shared/tracing.py`
- Modify: `ai-interview-backend/app/retrieval/pipeline.py`

- [ ] **Step 1: Add trace_span context manager to tracing.py**

Append to `ai-interview-backend/app/workflows/_shared/tracing.py`:

```python
from contextlib import contextmanager
from typing import Optional
import time
import json

from langsmith import run_helpers


@contextmanager
def trace_span(name: str, metadata: Optional[dict] = None):
    """创建 LangSmith 子 span，仅在 tracing 启用时生效

    Usage:
        with trace_span("vector_recall") as span:
            result = await vector_search(query)
            span.add_outputs({"count": len(result), "top_score": ...})
    """
    if not is_tracing_enabled():
        yield None
        return

    import langsmith

    parent_run = run_helpers.get_current_run_tree()
    start = time.time()
    span = None
    try:
        span = parent_run.create_child(
            name=name,
            run_type="retriever",
            inputs=metadata or {},
        ) if parent_run else None
        yield span
        duration_ms = (time.time() - start) * 1000
        if span:
            span.add_outputs({"duration_ms": duration_ms, "status": "ok"})
            span.end()
    except Exception as exc:
        if span:
            span.add_outputs({"error": str(exc), "status": "error"})
            span.end()
        raise
```

- [ ] **Step 2: Wrap pipeline steps with trace_span**

In `ai-interview-backend/app/retrieval/pipeline.py`, add import:

```python
from app.workflows._shared.tracing import trace_span
```

Then modify the `search` method to wrap each step:

```python
    async def search(
        self, query: str, filters: Optional[dict] = None
    ) -> List[SearchResult]:
        """混合检索：向量召回 + BM25 + RRF 融合 + 可选重排序"""
        # Step 1: 向量召回
        with trace_span("vector_recall", {"query": query, "top_k": self.vector_top_k}) as span:
            vector_results = await self._vector_search(query, filters)
            if span:
                span.add_outputs({"count": len(vector_results)})

        # Step 2: BM25 关键词检索
        with trace_span("bm25_search", {"query": query, "top_k": self.bm25_top_k}) as span:
            bm25_results = self._bm25_search(query)
            if span:
                span.add_outputs({"count": len(bm25_results)})

        # Step 3: RRF 融合
        with trace_span("rrf_fusion") as span:
            fused = rrf_fuse(vector_results, bm25_results, k=RRF_K)
            if span:
                span.add_outputs({"input_count": len(vector_results) + len(bm25_results),
                                  "fused_count": len(fused)})

        # Step 4: Cross-encoder 重排序
        if self.enable_rerank and self._reranker and len(fused) > self.final_top_k:
            with trace_span("rerank", {"candidate_count": len(fused)}) as span:
                final = await self._reranker.rerank(query, fused, top_k=self.final_top_k)
                if span:
                    span.add_outputs({"final_count": len(final)})
                return final
        else:
            cutoff = self.final_top_k or len(fused)
            return fused[:cutoff]
```

Replace the existing `retrieve`/`search` method body with the traced version above, keeping the existing private methods `_vector_search` and `_bm25_search` unchanged.

- [ ] **Step 3: Verify tracing module still works when disabled**

```bash
docker exec shilian-app python -c "
from app.workflows._shared.tracing import is_tracing_enabled, trace_span
assert is_tracing_enabled() == False
with trace_span('test'):
    pass
print('trace_span OK when tracing disabled')
"
```

Expected: `trace_span OK when tracing disabled`

- [ ] **Step 4: Run smoke tests**

```bash
docker exec shilian-app pytest -m "smoke" -v
```

Expected: All smoke tests pass

- [ ] **Step 5: Commit**

```bash
git add ai-interview-backend/app/workflows/_shared/tracing.py \
        ai-interview-backend/app/retrieval/pipeline.py
git commit -m "feat: add RAG retrieval step tracing spans

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: RAGAS + LangSmith 评估集成

**Files:**
- Create: `ai-interview-backend/eval/scripts/upload_golden_set.py`
- Modify: `ai-interview-backend/eval/scripts/eval_ragas.py`

- [ ] **Step 1: Write upload_golden_set.py**

`ai-interview-backend/eval/scripts/upload_golden_set.py`:

```python
"""将 golden_set.json 同步到 LangSmith Dataset

Usage:
    docker exec shilian-app python eval/scripts/upload_golden_set.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 确保项目根在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY not set. Skipping upload.")
        return

    golden_path = Path(__file__).resolve().parent.parent / "golden_set.json"
    if not golden_path.exists():
        print(f"Golden set not found: {golden_path}")
        return

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    entries = golden_data.get("entries", [])
    if not entries:
        print("No entries in golden set")
        return

    # 导入 langsmith SDK
    from langsmith import Client

    client = Client()
    dataset_name = "shilian-golden-set"

    # 如果 dataset 已存在，先删（重新创建保证一致）
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        client.delete_dataset(dataset_id=existing.id)
        print(f"Deleted existing dataset: {dataset_name}")
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=f"试炼 RAG 评估 golden set ({len(entries)} entries)",
    )

    for entry in entries:
        inputs = {"query": entry["query"]}
        outputs = {
            "relevant_chunk_ids": entry.get("relevant_chunk_ids", []),
            "reference_answer": entry.get("reference_answer", ""),
            "position_tag": entry.get("position_tag", ""),
            "difficulty": entry.get("difficulty", ""),
        }
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
        )

    print(f"Uploaded {len(entries)} examples to LangSmith dataset '{dataset_name}'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add --upload flag to eval_ragas.py**

In `ai-interview-backend/eval/scripts/eval_ragas.py`, add new CLI args and upload logic.

Find the arg parsing section and add:

```python
import argparse

# ... existing imports ...

def parse_args():
    parser = argparse.ArgumentParser(description="RAGAS evaluation script")
    parser.add_argument("--golden-set", default="eval/golden_set.json", help="Path to golden set JSON")
    parser.add_argument("--output", default=None, help="Output report path (default: eval/reports/baseline-{date}.json)")
    parser.add_argument("--upload", action="store_true", help="Upload results to LangSmith experiment")
    parser.add_argument("--experiment-name", default=None, help="LangSmith experiment name (default: ragas-{date})")
    return parser.parse_args()
```

Replace the existing argument handling. In the results output section (after computing metrics), add:

```python
    # Upload to LangSmith if requested
    if args.upload:
        _upload_to_langsmith(results, golden_data, dataset_name, experiment_name, client)


def _upload_to_langsmith(results, golden_data, dataset_name, experiment_name, client):
    """Upload RAGAS results as LangSmith experiment"""
    import os
    from datetime import date

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("LANGSMITH_API_KEY not set. Skipping LangSmith upload.")
        return

    from langsmith import Client

    ls_client = Client() if client is None else client
    if experiment_name is None:
        experiment_name = f"ragas-{date.today().isoformat()}"

    # Find dataset
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
    except Exception:
        print(f"LangSmith dataset '{dataset_name}' not found. Run upload_golden_set.py first.")
        return

    # Create experiment
    for entry in golden_data.get("entries", []):
        query = entry["query"]
        # Map results per entry (simplified — in prod, each entry has individual scores)
        ls_client.create_experiment(
            name=experiment_name,
            dataset_id=dataset.id,
            # ... per-entry scoring
        )

    print(f"Uploaded to LangSmith experiment: {experiment_name}")
```

Note: The exact integration code depends on the current `eval_ragas.py` structure. The key additions are `--upload` and `--experiment-name` CLI flags + LangSmith client calls after metrics computation.

- [ ] **Step 3: Verify upload script works**

```bash
docker exec shilian-app python -c "
import os
if os.environ.get('LANGSMITH_API_KEY'):
    from eval.scripts.upload_golden_set import main
    main()
else:
    print('No LANGSMITH_API_KEY — skipping (expected in dev)')
"
```

Expected: Prints "LANGSMITH_API_KEY not set" (or uploads if key set)

- [ ] **Step 4: Commit**

```bash
git add ai-interview-backend/eval/scripts/upload_golden_set.py \
        ai-interview-backend/eval/scripts/eval_ragas.py
git commit -m "feat: add LangSmith golden set upload and experiment integration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Service 全量去 @staticmethod

**Files to modify (14 files + all callers):**

Client services:
- `ai-interview-backend/app/services/client/ai_service.py`
- `ai-interview-backend/app/services/client/auth.py`
- `ai-interview-backend/app/services/client/interview_service.py`
- `ai-interview-backend/app/services/client/position_agent_service.py`
- `ai-interview-backend/app/services/client/redis_verification.py`
- `ai-interview-backend/app/services/client/resume_service.py`
- `ai-interview-backend/app/services/client/waiting_list.py`
- `ai-interview-backend/app/services/client/email_templates.py`

Backoffice services:
- `ai-interview-backend/app/services/backoffice/admin.py`
- `ai-interview-backend/app/services/backoffice/auth.py`
- `ai-interview-backend/app/services/backoffice/knowledge_service.py`
- `ai-interview-backend/app/services/backoffice/position_template_service.py`
- `ai-interview-backend/app/services/backoffice/question_bank_service.py`
- `ai-interview-backend/app/services/backoffice/waiting_list.py`

API callers (all API route files that import the services above):
- `ai-interview-backend/app/api/client/v1/*.py`
- `ai-interview-backend/app/api/backoffice/v1/*.py`

- [ ] **Step 1: Add service factories to deps.py**

Append to `ai-interview-backend/app/deps.py`:

```python
# ── Service factories ──

from app.services.client.ai_service import AIService
from app.services.client.auth import ClientAuthService
from app.services.client.interview_service import InterviewService
from app.services.client.position_agent_service import PositionAgentService
from app.services.client.redis_verification import RedisVerificationService
from app.services.client.resume_service import ResumeService
from app.services.client.waiting_list import ClientWaitingListService
from app.services.client.email_templates import EmailTemplateService

from app.services.backoffice.admin import AdminService
from app.services.backoffice.auth import BackofficeAuthService
from app.services.backoffice.knowledge_service import KnowledgeService
from app.services.backoffice.position_template_service import PositionTemplateService
from app.services.backoffice.question_bank_service import QuestionBankService
from app.services.backoffice.waiting_list import BackofficeWaitingListService


def get_ai_service() -> AIService:
    return AIService()

def get_client_auth_service() -> ClientAuthService:
    return ClientAuthService()

def get_interview_service() -> InterviewService:
    return InterviewService()

def get_position_agent_service() -> PositionAgentService:
    return PositionAgentService()

def get_redis_verification_service() -> RedisVerificationService:
    return RedisVerificationService()

def get_resume_service() -> ResumeService:
    return ResumeService()

def get_client_waiting_list_service() -> ClientWaitingListService:
    return ClientWaitingListService()

def get_email_template_service() -> EmailTemplateService:
    return EmailTemplateService()


def get_admin_service() -> AdminService:
    return AdminService()

def get_backoffice_auth_service() -> BackofficeAuthService:
    return BackofficeAuthService()

def get_knowledge_service() -> KnowledgeService:
    return KnowledgeService()

def get_position_template_service() -> PositionTemplateService:
    return PositionTemplateService()

def get_question_bank_service() -> QuestionBankService:
    return QuestionBankService()

def get_backoffice_waiting_list_service() -> BackofficeWaitingListService:
    return BackofficeWaitingListService()
```

- [ ] **Step 2: Remove @staticmethod from all service methods**

For each service file, perform this transformation:

Pattern A — class-level @staticmethod:
```python
# Before
class FooService:
    @staticmethod
    async def do_thing(db, arg1, arg2):
        ...

# After
class FooService:
    async def do_thing(self, db, arg1, arg2):
        ...
```

Each file's `@staticmethod` lines are removed, and `self` is inserted as the first parameter.
The method body remains unchanged.

Files to transform:
1. `ai-interview-backend/app/services/client/ai_service.py`
2. `ai-interview-backend/app/services/client/auth.py`
3. `ai-interview-backend/app/services/client/interview_service.py`
4. `ai-interview-backend/app/services/client/position_agent_service.py`
5. `ai-interview-backend/app/services/client/redis_verification.py`
6. `ai-interview-backend/app/services/client/resume_service.py`
7. `ai-interview-backend/app/services/client/waiting_list.py`
8. `ai-interview-backend/app/services/client/email_templates.py`
9. `ai-interview-backend/app/services/backoffice/admin.py`
10. `ai-interview-backend/app/services/backoffice/auth.py`
11. `ai-interview-backend/app/services/backoffice/knowledge_service.py`
12. `ai-interview-backend/app/services/backoffice/position_template_service.py`
13. `ai-interview-backend/app/services/backoffice/question_bank_service.py`
14. `ai-interview-backend/app/services/backoffice/waiting_list.py`

- [ ] **Step 3: Update all API route callers**

For each API route file that calls `ClassName.method(db, ...)`:

```python
# Before
@router.post("/upload")
async def upload_resume(
    db: AsyncSession = Depends(get_db),
    ...
):
    result = await ResumeService.parse_resume(db, file)

# After
@router.post("/upload")
async def upload_resume(
    db: AsyncSession = Depends(get_db),
    resume_service: ResumeService = Depends(get_resume_service),
    ...
):
    result = await resume_service.parse_resume(db, file)
```

All caller files that need updating (map by inspecting each API file's imports):

- `ai-interview-backend/app/api/client/v1/auth.py` — ClientAuthService
- `ai-interview-backend/app/api/client/v1/resume.py` — ResumeService
- `ai-interview-backend/app/api/client/v1/interview.py` — InterviewService, InterviewGraphService
- `ai-interview-backend/app/api/client/v1/position_agent.py` — PositionAgentService
- `ai-interview-backend/app/api/client/v1/verification.py` — RedisVerificationService
- `ai-interview-backend/app/api/client/v1/waiting_list.py` — ClientWaitingListService
- `ai-interview-backend/app/api/backoffice/v1/auth.py` — BackofficeAuthService
- `ai-interview-backend/app/api/backoffice/v1/admin.py` — AdminService
- `ai-interview-backend/app/api/backoffice/v1/question_bank.py` — QuestionBankService
- `ai-interview-backend/app/api/backoffice/v1/knowledge.py` — KnowledgeService
- `ai-interview-backend/app/api/backoffice/v1/position_template.py` — PositionTemplateService
- `ai-interview-backend/app/api/backoffice/v1/waiting_list.py` — BackofficeWaitingListService

- [ ] **Step 4: Run full test suite**

```bash
docker exec shilian-app pytest -m "unit or smoke" -v
```

Expected: All tests pass

- [ ] **Step 5: Commit (one commit per group of files to keep changes reviewable)**

```bash
# Sub-commit: client services
git add ai-interview-backend/app/services/client/ ai-interview-backend/app/api/client/
git commit -m "refactor: remove @staticmethod from client services

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Sub-commit: backoffice services
git add ai-interview-backend/app/services/backoffice/ ai-interview-backend/app/api/backoffice/
git commit -m "refactor: remove @staticmethod from backoffice services

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"

# Sub-commit: deps.py factory functions
git add ai-interview-backend/app/deps.py
git commit -m "feat: add service factory functions to deps.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 8: Final verification and cleanup

**Files:**
- Modify: `ai-interview-backend/CLAUDE.md` (update Phase 4 status)

- [ ] **Step 1: Run all tests**

```bash
docker exec shilian-app pytest -m "unit or smoke" -v
```

Expected: All tests pass

- [ ] **Step 2: Verify no regression in agent import chain**

```bash
docker exec shilian-app python -c "
from app.agents import BaseAgent, QuestionAgent, EvaluatorAgent, ReportAgent
from app.repositories import question_bank_repo, knowledge_repo, interview_repo
from app.deps import get_question_agent, get_evaluator_agent, get_report_agent
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Update CLAUDE.md**

Replace the Phase 4 line (currently `[ ] **Phase 4** — ...`) with:

```markdown
- [x] **Phase 4** — 多 Agent + 可观测性：Agent 类提取 + Repository 补全 + RAG trace + RAGAS LangSmith + service 去 static（2026-07-11）
```

Append a Phase 4 implementation record section after the existing Phase 3 record:

```markdown
### Phase 4 实施记录

**Agent 类提取**：
- `app/agents/` — BaseAgent + QuestionAgent + EvaluatorAgent + ReportAgent
- 3 个新 YAML prompt：question_agent / evaluator_agent / report_agent
- evaluate_node / generate_report_node 委托 Agent 类，不再直接调 LLM
- InterviewGraphService 通过 state.custom 注入 agent 实例

**Repository 补全**：
- `question_bank_repo.py` — search_by_position / list_by_position / increment_use_count
- `knowledge_repo.py` — list_by_document / get_by_chunk_id / full_text_search
- `interview_repo.py` — get_by_id_with_messages / delete_cascade

**RAG 链路 Tracing**：
- `tracing.py` — trace_span context manager（LangSmith RunTree 子 span）
- `pipeline.py` — 4 步检索步骤各包一层 trace_span

**RAGAS + LangSmith**：
- `eval/scripts/upload_golden_set.py` — 同步 golden_set.json 到 LangSmith dataset
- `eval/scripts/eval_ragas.py` — --upload 和 --experiment-name 参数

**Service 去 @staticmethod**：
- 14 个 service 文件全部改为实例方法 + Depends 注入
- `app/deps.py` — 统一 factory 函数
```

- [ ] **Step 4: Final commit**

```bash
git add ai-interview-backend/CLAUDE.md
git commit -m "docs: mark Phase 4 complete, add implementation record

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

- [ ] **Step 5: Run full verification**

```bash
docker exec shilian-app pytest -v
```

Expected: All tests green. Phase 4 complete.
