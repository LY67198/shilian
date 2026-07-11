"""
岗位匹配 Agent 编排层

设计要点：
- 用 LangChain `create_agent`（底层 LangGraph ReAct 循环）把 5 个工具串起来
- LLM 用 ChatOpenAI 包装 DeepSeek（DeepSeek API 兼容 OpenAI Function Calling）
- 系统 Prompt 严格规定工作流和最终输出格式
- 最终输出 JSON 字符串，由 service 层解析后返给前端
"""
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage, AIMessage

from app.common.json_utils import extract_json
from app.core.config import settings
from app.llm import get_chat_llm
from app.llm.prompts import load_prompt
from app.services.client.position_agent_tools import POSITION_AGENT_TOOLS

logger = logging.getLogger(__name__)


# ── LLM 配置（委托 app.llm.get_chat_llm 统一管理）────────────────

def get_llm() -> ChatOpenAI:
    """ChatOpenAI 实例（包装 DeepSeek）

    Phase 1 重构后委托给 app.llm.get_chat_llm 统一管理，
    旧签名保留只为不破坏外部 import。
    """
    return get_chat_llm(temperature=0.3)


# ── 系统 Prompt（从 YAML 加载）───────────────────────────────────────────

def _get_system_prompt() -> str:
    """从 position_agent_system.yaml 加载系统 Prompt"""
    prompt = load_prompt("position_agent_system")
    return prompt.messages[0].prompt.template


# ── 全局 Agent 实例（懒加载）────────────────────────────────────────────

_agent = None

def get_agent():
    """单例 Agent（LangGraph CompiledStateGraph）"""
    global _agent
    if _agent is None:
        _agent = create_agent(
            model=get_llm(),
            tools=POSITION_AGENT_TOOLS,
            system_prompt=_get_system_prompt(),
        )
    return _agent


# ── 对外服务 ────────────────────────────────────────────────────────────

class PositionAgentService:

    async def run_agent(
        self,
        resume_id: int,
        target_direction: str | None = None,
    ) -> dict:
        """
        运行岗位匹配 Agent，返回结构化推荐结果。

        Args:
            resume_id: 简历 ID
            target_direction: 用户期望方向（可选），如 "Python 后端"

        Returns:
            {
                "result": {...},          # Agent 最终输出 JSON
                "intermediate_steps": [...]  # 调用过的工具步骤摘要
            }
        """
        user_input = f"我的简历 ID 是 {resume_id}，请帮我做岗位匹配。"
        if target_direction:
            user_input += f"我想找的方向是：{target_direction}。"

        logger.info(f"[PositionAgent] 开始运行，resume_id={resume_id}")

        config = {"recursion_limit": 25}

        try:
            agent = get_agent()
            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
        except Exception as e:
            logger.error(f"[PositionAgent] Agent 异常: {e}")
            return {
                "result": {"error": f"Agent 执行失败: {str(e)}"},
                "intermediate_steps": [],
            }

        # 提取最终输出（最后一条消息的 content）
        messages = response.get("messages", [])
        if not messages:
            logger.error("[PositionAgent] Agent 返回空消息列表")
            return {
                "result": {"error": "Agent 未返回任何消息"},
                "intermediate_steps": [],
            }
        final_message = messages[-1]
        raw_output = final_message.content if hasattr(final_message, "content") else str(final_message)
        logger.info(f"[PositionAgent] 完成，raw_output 长度: {len(raw_output)}")

        # 解析最终 JSON
        try:
            result = extract_json(raw_output)
        except Exception as e:
            logger.error(f"[PositionAgent] 输出 JSON 解析失败: {e}, raw: {raw_output[:300]}")
            result = {
                "error": "Agent 最终输出格式异常",
                "raw_output": raw_output[:1000],
            }

        # 从消息列表中提取中间步骤摘要
        # 先构建 tool_call_id → {name, args} 映射（来自 AIMessage.tool_calls）
        tool_calls_map = {}
        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_map[tc["id"]] = {
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    }

        # 从 ToolMessage 提取执行结果
        steps_summary = []
        for msg in response["messages"]:
            if isinstance(msg, ToolMessage):
                tc_info = tool_calls_map.get(msg.tool_call_id, {})
                steps_summary.append({
                    "tool": tc_info.get("name", msg.name or "unknown"),
                    "input_preview": str(tc_info.get("args", {}))[:200],
                    "output_preview": str(msg.content)[:200],
                })

        return {
            "result": result,
            "intermediate_steps": steps_summary,
        }


position_agent_service = PositionAgentService()
