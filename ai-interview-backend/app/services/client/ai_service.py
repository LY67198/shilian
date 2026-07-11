import json
import logging

from app.llm import chat_completion, get_chat_llm
from app.llm.prompts import load_prompt

logger = logging.getLogger(__name__)


class AIService:
    """DeepSeek AI 服务 - 面试模拟核心

    Phase 2 重构后：
    - 出题方法（parse/analyze/generate_questions/select_and_adapt/generate_with_seeds）切 YAML
    - 评分/报告方法已迁移到 app/workflows/interview/nodes/
    - _chat / _chat_stream 保留兼容旧调用
    - _extract_json 解析失败返回兜底 dict
    """

    # ── 底层 LLM 调用 ──────────────────────────────────────────────

    async def _chat(self, messages: list, temperature: float = 0.7) -> str:
        """基础对话补全（委托给 app.llm.chat_completion）"""
        return await chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=2000,
            stream=False,
        )

    async def _chat_stream(self, messages: list, temperature: float = 0.7):
        """流式对话补全（委托给 app.llm.chat_completion）"""
        gen = await chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=2000,
            stream=True,
        )
        async for chunk in gen:
            yield chunk

    # ── JSON 解析（委托到 shared utility）─────────────────────────────

    def _extract_json(self, text: str) -> dict:
        """从 AI 响应中提取 JSON（委托到 app.common.json_utils.extract_json）"""
        from app.common.json_utils import extract_json
        return extract_json(text)

    # ── 出题方法（YAML prompt）─────────────────────────────────────

    async def parse_resume(self, resume_text: str) -> dict:
        """解析简历文本，提取结构化信息（姓名、学历、技能、经历等）"""
        prompt = load_prompt("resume_parse")
        llm = get_chat_llm(temperature=0.3)
        chain = prompt | llm
        result = await chain.ainvoke({"resume_text": resume_text})
        content = result.content if hasattr(result, "content") else str(result)
        return self._extract_json(content)

    async def analyze_resume(self, parsed_resume: dict, target_position: str) -> dict:
        """分析简历质量，给出评分、优劣势和改进建议"""
        is_intern = any(kw in target_position.lower() for kw in ["实习", "intern"])
        if is_intern:
            level_hint = (
                "【重要】目标岗位是实习岗位，候选人是在校学生，请严格按实习生标准评价。\n"
                "禁止事项：\n"
                "- 禁止在weaknesses中提及'缺少实习经验'、'缺少工作经验'、'没有实际工作经验'等类似表述\n"
                "- 禁止因为项目是个人项目或校内项目而扣分，这对实习生来说是正常的\n"
                "- 禁止要求候选人具备线上生产环境经验\n"
                "评价重点：技术基础扎实度、项目完成度和技术深度、学习能力、编码能力。\n"
                "个人项目和校内项目同样能体现技术能力，请公正评价项目质量本身。\n"
            )
        else:
            level_hint = (
                "目标岗位是正式岗位，请按社招标准评价。\n"
                "重点关注：工作经验、项目深度、技术广度、解决复杂问题的能力。\n"
            )
        prompt = load_prompt("resume_analyze")
        llm = get_chat_llm(temperature=0.4)
        chain = prompt | llm
        result = await chain.ainvoke({
            "target_position": target_position,
            "level_hint": level_hint,
            "resume_json": json.dumps(parsed_resume, ensure_ascii=False),
        })
        content = result.content if hasattr(result, "content") else str(result)
        return self._extract_json(content)

    async def generate_questions(
        self,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        count: int,
    ) -> list:
        """根据简历和目标岗位生成面试题目"""
        difficulty_map = {
            "easy": "初级，侧重基础知识和简单项目经验",
            "medium": "中级，涵盖技术深度和项目设计思路",
            "hard": "高级，深入系统设计、性能优化和技术原理",
        }
        difficulty_desc = difficulty_map.get(difficulty, difficulty_map["medium"])

        is_intern = any(kw in target_position for kw in ["实习", "intern", "Intern"])
        position_hint = (
            "实习岗位，候选人可能是在校学生，请适当降低难度"
            if is_intern
            else "正式岗位，请按正常标准出题"
        )

        prompt = load_prompt("question_generate")
        llm = get_chat_llm(temperature=0.7)
        chain = prompt | llm
        result = await chain.ainvoke({
            "target_position": target_position,
            "difficulty_desc": difficulty_desc,
            "position_hint": position_hint,
            "resume_json": json.dumps(parsed_resume, ensure_ascii=False),
            "count": count,
        })
        content = result.content if hasattr(result, "content") else str(result)
        return self._extract_json(content)

    async def select_and_adapt_questions(
        self,
        candidates: list,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        target_n: int,
    ) -> list:
        """题库召回充分时调用：从候选题中挑 N 题"""
        is_intern = any(kw in target_position for kw in ["实习", "intern", "Intern"])
        intern_hint = (
            "候选人为实习岗位，优先选择基础类、项目类问题。"
            if is_intern
            else ""
        )

        prompt = load_prompt("question_select")
        llm = get_chat_llm(temperature=0.3)
        chain = prompt | llm
        result = await chain.ainvoke({
            "target_position": target_position,
            "difficulty": difficulty,
            "intern_hint": intern_hint,
            "candidate_count": len(candidates),
            "target_n": target_n,
            "resume_json": json.dumps(parsed_resume, ensure_ascii=False),
            "candidates_json": json.dumps(candidates, ensure_ascii=False),
        })
        content = result.content if hasattr(result, "content") else str(result)
        return self._extract_json(content)

    async def generate_with_seeds(
        self,
        seed_questions: list,
        parsed_resume: dict,
        target_position: str,
        difficulty: str,
        target_n: int,
    ) -> list:
        """题库召回不足时调用：以题库题作为种子，AI 兜底生成剩余题"""
        seed_count = len(seed_questions)
        fallback_count = target_n - seed_count

        is_intern = any(kw in target_position for kw in ["实习", "intern", "Intern"])
        intern_hint = "（实习岗位，候选人为在校学生，难度偏基础）" if is_intern else ""

        prompt = load_prompt("question_seed")
        llm = get_chat_llm(temperature=0.5)
        chain = prompt | llm
        result = await chain.ainvoke({
            "target_position": target_position,
            "intern_hint": intern_hint,
            "difficulty": difficulty,
            "seed_count": seed_count,
            "fallback_count": fallback_count,
            "target_n": target_n,
            "resume_json": json.dumps(parsed_resume, ensure_ascii=False),
            "seed_json": json.dumps(seed_questions, ensure_ascii=False),
        })
        content = result.content if hasattr(result, "content") else str(result)
        return self._extract_json(content)


ai_service = AIService()
