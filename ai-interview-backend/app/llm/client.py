"""LLM 客户端工厂（统一入口）

设计原则：
- 全应用只有这一个 ChatOpenAI 工厂，禁止别处直接 import langchain_openai
- 用 LangChain ChatOpenAI 包装 DeepSeek（OpenAI 兼容 SDK）
- 集中管理重试、超时、温度默认值
- 单例懒加载，避免重复创建连接
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_llm_cache: dict[tuple, ChatOpenAI] = {}


def get_chat_llm(
    temperature: float = 0.7,
    streaming: bool = False,
    max_tokens: int = 2000,
    model: Optional[str] = None,
) -> ChatOpenAI:
    """
    获取 LLM 客户端（按参数缓存）

    Args:
        temperature: 0-1，越低越确定
        streaming: 是否流式（仅影响 invoke 调用方式）
        max_tokens: 最大输出 token
        model: 覆盖默认模型（默认用 settings.DEEPSEEK_MODEL）

    Returns:
        ChatOpenAI 实例
    """
    key = (
        temperature,
        streaming,
        max_tokens,
        model or settings.DEEPSEEK_MODEL,
    )
    if key not in _llm_cache:
        _llm_cache[key] = ChatOpenAI(
            model=key[3],
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            timeout=120,
        )
        logger.debug(
            f"LLM 客户端创建: model={key[3]}, temp={temperature}, "
            f"streaming={streaming}, max_tokens={max_tokens}"
        )
    return _llm_cache[key]


def reset_cache() -> None:
    """清空 LLM 缓存（测试用）"""
    _llm_cache.clear()


# ─── 带重试的便捷调用（向后兼容 ai_service 旧 _chat / _chat_stream 风格）───
import asyncio
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    asyncio.TimeoutError,
)
_MAX_RETRIES = 3

_raw_client: Optional[AsyncOpenAI] = None


def _get_raw_client() -> AsyncOpenAI:
    """裸 AsyncOpenAI（用于 ai_service 旧风格调用）"""
    global _raw_client
    if _raw_client is None:
        _raw_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return _raw_client


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(_MAX_RETRIES + 1),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _create_completion(client: AsyncOpenAI, messages: list, temperature: float, max_tokens: int, stream: bool):
    """执行单次 LLM 调用（带 tenacity 自动重试）"""
    return await client.chat.completions.create(
        model=settings.DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )


async def chat_completion(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    stream: bool = False,
):
    """带指数退避的 LLM 调用（兼容旧 ai_service 风格）

    Args:
        messages: OpenAI 风格 messages 列表
        temperature: 0-1
        max_tokens: 最大输出 token
        stream: True 返回异步生成器；False 返回字符串

    Returns:
        字符串 or 异步生成器
    """
    client = _get_raw_client()
    try:
        response = await _create_completion(client, messages, temperature, max_tokens, stream)
        if stream:
            return _stream_chunks(response, client)
        return response.choices[0].message.content.strip()
    except _RETRYABLE_ERRORS as e:
        logger.error(f"LLM 调用彻底失败（已重试 {_MAX_RETRIES} 次）: {e}")
        raise


async def _stream_chunks(stream, client):
    """流式输出：逐 chunk yield 文本"""
    try:
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"LLM 流式输出中断: {e}")
        raise