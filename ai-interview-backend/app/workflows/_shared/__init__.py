"""LangGraph 工作流共享基础设施

Phase 1 引入，所有 workflow（interview / question_gen / evaluation）共用：
- `llm`           — 统一 LLM 工厂（包装 app.llm，加 workflow 级 trace）
- `checkpointer`   — AsyncPostgresSaver 单例 + schema 初始化
- `state_base`     — 公共 TypedDict 字段（thread_id / retry_count / errors）
- `tracing`        — LangSmith 配置（默认 off）
- `format_exception` — ExceptionGroup 解包
- `tools`          — 默认工具集注册中心

工作流目录约定：
    workflows/
    ├── _shared/          ← 本包
    ├── interview/        ← 多轮面试 StateGraph
    ├── question_gen/     ← 题目召回自检循环
    └── evaluation/       ← 多 Agent 评估
"""
from app.workflows._shared.llm import get_workflow_llm
from app.workflows._shared.checkpointer import get_checkpointer, close_checkpointer
from app.workflows._shared.state_base import BaseWorkflowState
from app.workflows._shared.format_exception import flatten_exceptions, format_exception_chain
from app.workflows._shared.tracing import configure_langsmith, is_tracing_enabled
from app.workflows._shared.tools import get_default_tools

__all__ = [
    "get_workflow_llm",
    "get_checkpointer",
    "close_checkpointer",
    "BaseWorkflowState",
    "format_exception_chain",
    "flatten_exceptions",
    "configure_langsmith",
    "is_tracing_enabled",
    "get_default_tools",
]