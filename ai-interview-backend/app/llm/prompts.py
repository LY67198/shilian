"""PromptTemplate 加载器（YAML 格式）

prompt 文件位于 `app/prompts/*.yaml`：
- 文件名 = prompt 名（不含扩展名）
- 必须含 system + user_template 字段
- 可选：version / temperature / response_format

使用：
    prompt = load_prompt("resume_parse")
    chain = prompt | llm | JsonOutputParser()
    result = await chain.ainvoke({"resume_text": "..."})
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=32)
def load_prompt(name: str) -> ChatPromptTemplate:
    """加载 YAML prompt 文件，返回 LangChain ChatPromptTemplate

    Args:
        name: prompt 名（不含 .yaml 后缀）

    Raises:
        FileNotFoundError: prompt 文件不存在
    """
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt 文件不存在: {path}\n"
            f"现有 prompts: {list_prompts()}"
        )

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    system = data.get("system", "")
    user_template = data.get("user_template", "")

    if user_template:
        return ChatPromptTemplate.from_messages([
            ("system", system),
            ("user", user_template),
        ])
    return ChatPromptTemplate.from_messages([
        ("system", system),
    ])


def list_prompts() -> list[str]:
    """列出所有可用的 prompt 名（按文件名）"""
    if not _PROMPTS_DIR.exists():
        return []
    return sorted(p.stem for p in _PROMPTS_DIR.glob("*.yaml"))


def get_prompt_metadata(name: str) -> Optional[dict]:
    """读取 prompt 的元数据（version / temperature / response_format）"""
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        "name": data.get("name", name),
        "version": data.get("version", 1),
        "temperature": data.get("temperature", 0.7),
        "response_format": data.get("response_format", "text"),
    }