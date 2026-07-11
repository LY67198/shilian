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
        """初始化报告生成 Agent，使用 report_agent prompt。"""
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
    """将问答数据构建为格式化文本块，用于注入 prompt。

    Args:
        qa_data: 问答数据列表，每项包含 question / answer / score 字段。

    Returns:
        格式化的问答文本。
    """
    text = ""
    for item in qa_data:
        text += (
            f"问题：{item.get('question', '')}\n"
            f"回答：{item.get('answer', '未回答')}\n"
            f"得分：{item.get('score', 'N/A')}\n\n"
        )
    return text
