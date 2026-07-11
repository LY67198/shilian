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
