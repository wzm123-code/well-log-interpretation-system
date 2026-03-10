"""
智能体构建工具 - 根据 JSON 配置创建 LangChain ReAct Agent

【用途】DataAgent、ExpertAgent、SupervisorAgent 均由此构建。
从 config_path 读取 model、temperature、sp 等，实例化 ChatDeepSeek，用 create_react_agent 封装。
"""
import json
import os
from typing import List, Optional, Type
from langchain_core.tools import BaseTool
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

class BaseAgentState(MessagesState):
    """基础智能体状态，可扩展"""
    pass


def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def build_agent(
    config_path: str,
    tools: List[BaseTool],
    model_provider: str = "deepseek",
    checkpointer: Optional[MemorySaver] = None,
    state_schema: Type = BaseAgentState,
    ctx=None
):
    """
    构建智能体

    参数:
        config_path: 配置文件路径
        tools: 工具列表
        model_provider: 模型提供商（当前仅支持 deepseek）
        checkpointer: 检查点存储器（用于对话记忆）
        state_schema: 状态模式
        ctx: 上下文（未使用，保留兼容性）

    返回:
        Agent 执行器（可调用）
    """
    config = load_config(config_path)

    # 提取模型配置
    model_config = config.get("config", {})
    model_name = model_config.get("model", "deepseek-chat")
    temperature = model_config.get("temperature", 0.7)
    top_p = model_config.get("top_p", 0.9)
    max_tokens = model_config.get("max_completion_tokens", 4096)
    timeout = model_config.get("timeout", 300)

    # 创建 LLM 实例
    if model_provider == "deepseek":
        llm = ChatDeepSeek(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            top_p=top_p
        )
    else:
        raise ValueError(f"不支持的模型提供商: {model_provider}")

    # 绑定工具
    llm_with_tools = llm.bind_tools(tools)

    # 创建 React Agent
    agent = create_react_agent(
        llm_with_tools,
        tools,
        checkpointer=checkpointer,
        # prompt=system_prompt  # 系统提示词
    )

    return agent