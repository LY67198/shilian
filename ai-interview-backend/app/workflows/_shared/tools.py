"""默认工具集注册中心

所有 LangGraph workflow 共用的工具（@tool 装饰）放在这里。
新增工具：在 DEFAULT_TOOLS 元组中追加。

参考 OnCall Agent 的 tools/__init__.py 模式 — 集中管理。
"""
from __future__ import annotations

from typing import Iterable

from langchain_core.tools import BaseTool

# 占位：后续 Phase 2-3 接入具体工具
# 当前 workflow（interview/question_gen）尚未实现 graph，所以工具集为空。
# 示例：等 Phase 3 实现 interview graph 时，加 retrieve_knowledge / evaluate_answer / submit_answer 工具
DEFAULT_TOOLS: tuple[BaseTool, ...] = ()


def get_default_tools() -> Iterable[BaseTool]:
    """获取默认工具集（用于 workflow 初始化）"""
    return list(DEFAULT_TOOLS)