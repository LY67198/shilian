"""LLM 原子能力层（LangChain 域）

本包只做"原子能力"封装，不做任何业务编排：
- `client` — ChatOpenAI 工厂（包装 DeepSeek OpenAI 兼容模式）
- `embedding` — DashScopeEmbeddings 工厂 + async/sync 批处理
- `prompts` — YAML PromptTemplate 加载器

业务编排（多步骤流程、Agent 决策、条件分支）由 `app/workflows/` 下的 LangGraph 接管。
"""
from app.llm.client import chat_completion, get_chat_llm, reset_cache
from app.llm.embedding import (
    embed_text,
    embed_texts,
    embed_text_sync,
    embed_texts_sync,
)
from app.llm.prompts import get_prompt_metadata, list_prompts, load_prompt

__all__ = [
    "get_chat_llm",
    "chat_completion",          # 带重试的便捷调用（兼容 ai_service 旧风格）
    "reset_cache",
    "embed_text",
    "embed_texts",
    "embed_text_sync",
    "embed_texts_sync",
    "load_prompt",
    "list_prompts",
    "get_prompt_metadata",
]