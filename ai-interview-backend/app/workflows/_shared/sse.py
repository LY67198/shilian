"""LangGraph astream_events → SSE 事件封装

用法:
    from app.workflows._shared.sse import astream_to_sse

    async for sse_event in astream_to_sse(graph.astream_events(...)):
        yield sse_event  # 可直接给 StreamingResponse
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)

# 节点到中文消息的映射
_NODE_LABELS: dict[str, str] = {
    "fetch_context": "正在加载面试上下文...",
    "retrieve_knowledge": "正在检索知识库...",
    "evaluate": "正在评估回答...",
    "ask_question": "正在准备下一题...",
    "generate_report": "正在生成评估报告...",
}

# evaluate / ask_question / generate_report 等节点的完成事件名
_NODE_OUTPUT_EVENTS: set[str] = {"evaluate", "ask_question", "generate_report"}


async def astream_to_sse(
    event_stream,
    debug: bool = False,
) -> AsyncIterator[str]:
    """将 LangGraph astream_events() 输出转换为 SSE 事件字符串

    映射规则：
    - on_chat_model_stream → event: chunk
    - on_chain_start (已知节点名) → event: status
    - on_chain_end (evaluate with dict output) → event: score
    - on_chain_end (ask_question with next_question) → event: next_question
    - on_chain_end (generate_report) → event: report
    - on_tool_start / on_tool_end → event: tool (debug 模式)
    - 异常 → event: error
    """
    async for event in event_stream:
        kind = event.get("event", "")
        name = event.get("name", "")

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield _sse("chunk", {"content": chunk.content})

        elif kind == "on_chain_start":
            if name in _NODE_LABELS:
                yield _sse("status", {"node": name, "message": _NODE_LABELS[name]})

        elif kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if name == "evaluate" and isinstance(output, dict) and "score" in output:
                yield _sse("score", {
                    "score": output.get("score", 5.0),
                    "feedback": output.get("feedback", ""),
                    "follow_up": output.get("follow_up", False),
                })
            elif name == "ask_question" and isinstance(output, dict) and "next_question" in output:
                yield _sse("next_question", {
                    "question": output["next_question"],
                    "index": output.get("index", 0),
                })
            elif name == "generate_report" and isinstance(output, dict):
                yield _sse("report", {
                    "overall_score": output.get("overall_score", 0),
                    "report": output.get("report", {}),
                })

        elif kind in ("on_tool_start", "on_tool_end") and debug:
            yield _sse("tool", {"event": kind, "name": name})

        elif kind in ("on_chain_error", "error"):
            err = event.get("data", {})
            yield _sse("error", {
                "code": "GRAPH_ERROR",
                "message": str(err.get("error", err)),
            })

    yield _sse("done", {"message": "ok"})


def _sse(event: str, data: dict) -> str:
    """生成 SSE 格式字符串"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
