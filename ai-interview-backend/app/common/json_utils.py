"""JSON 解析工具 — 从 AI 响应中提取 JSON，兼容各种格式"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """从 AI 响应中提取 JSON，兼容 markdown 包裹、前后解释文字等情况。
    解析失败时返回兜底 dict，不再抛异常。"""
    if isinstance(text, (dict, list)):
        return text if isinstance(text, dict) else {"items": text}

    raw_text = (text or "").strip()
    if not raw_text:
        logger.warning("AI 返回内容为空，返回兜底结果")
        return {"score": 5.0, "feedback": "", "parse_failed": True}

    decoder = json.JSONDecoder()

    # 1. 最理想情况：整段就是 JSON
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # 2. 常见情况：```json ... ``` 代码块
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw_text, flags=re.IGNORECASE)
    for block in fenced_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            pass

    # 3. 宽松情况：文本中夹带一段 JSON，从第一个 { 或 [ 开始 raw_decode
    start_positions = [i for i in (raw_text.find("{"), raw_text.find("[")) if i != -1]
    for start in sorted(start_positions):
        snippet = raw_text[start:].strip()
        try:
            obj, _ = decoder.raw_decode(snippet)
            return obj
        except json.JSONDecodeError:
            continue

    logger.warning(f"JSON解析失败，返回兜底结果。原文前200字符: {raw_text[:200]}")
    return {
        "score": 5.0,
        "feedback": raw_text[:200],
        "parse_failed": True,
    }
