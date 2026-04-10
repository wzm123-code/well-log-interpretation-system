"""
数据智能体 - 负责测井数据的预览和预处理

【职责】
  - preview_data：数据预览，获取列名与统计，供任务规划使用
  - clean_data：缺失值/异常值处理
  - normalize_data：归一化

【调用方式】监督智能体通过 execute_tool(tool_name, parameters, file_path) 委派。
不构建 LangChain ReAct Agent：任务规划与推理由 SupervisorAgent 完成，此处仅同步调用工具，避免重复加载模型与多余 API 消耗。
"""
import asyncio
from typing import Any, Dict

from utils.agent_helpers import set_file_param

from tools.data_processing_tools import preview_data, clean_data, normalize_data

# 数据智能体负责的工具，工具名映射到实际函数，方便按名查找
DATA_AGENT_TOOLS = {
    "preview_data": preview_data,
    "clean_data": clean_data,
    "normalize_data": normalize_data,
}


class DataAgent:
    """
    数据智能体封装类 - 仅对外提供 execute_tool 异步接口

    工具由 LangChain @tool 包装，执行路径为 invoke，不经子 Agent 多步推理。
    """

    def __init__(self, ctx=None):
        self.ctx = ctx

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        file_path: str,
    ) -> str:
        """供监督智能体委派：根据 tool_name 查找工具，注入 file_path，在线程池中同步执行并返回结果字符串"""
        tool_func = DATA_AGENT_TOOLS.get(tool_name)
        if not tool_func:
            return f"错误: 数据智能体不负责工具 {tool_name}"
        params = set_file_param(tool_func, parameters or {}, file_path)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, lambda: tool_func.invoke(params))
            return str(result) if result is not None else ""
        except Exception as e:
            return f"错误: {tool_name} 执行失败 - {str(e)}"
