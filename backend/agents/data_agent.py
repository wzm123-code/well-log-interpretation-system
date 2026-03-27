"""
数据智能体 - 负责测井数据的预览和预处理

【职责】
  - preview_data：数据预览，获取列名与统计，供任务规划使用
  - clean_data：缺失值/异常值处理
  - normalize_data：归一化

【调用方式】监督智能体通过 execute_tool(tool_name, parameters, file_path) 委派，不对外暴露 execute_task。
"""
import os
import asyncio  
#导入第三方库
from typing import Dict, Any
from utils.agent_builder import build_agent
from utils.agent_helpers import set_file_param

# 导入工具
from tools.data_processing_tools import preview_data, clean_data, normalize_data

# 配置文件路径
DATA_AGENT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "../config/data_agent_deepseek_config.json"
)

# 数据智能体负责的工具，工具名映射到实际函数，方便按名查找
DATA_AGENT_TOOLS = {
    "preview_data": preview_data,#数据预览
    "clean_data": clean_data,#数据清洗
    "normalize_data": normalize_data,#数据归一化
}

#创建一个agent实例 返回给DataAgent类使用
def build_data_agent(ctx=None):
    """根据 DATA_AGENT_CONFIG 构建 LangChain ReAct Agent，绑定 preview/clean/normalize 工具"""
    agent = build_agent(
        config_path=DATA_AGENT_CONFIG,
        tools=list(DATA_AGENT_TOOLS.values()),
        model_provider="deepseek",
        ctx=ctx
    )

    return agent


class DataAgent:
    """
    数据智能体封装类 - 仅对外提供 execute_tool 异步接口

    【属性】_agent：内部 LangChain Agent，供扩展用；正常流程仅调用 execute_tool。
    """

    def __init__(self, ctx=None):
        self.ctx = ctx
        self._agent = build_data_agent(ctx)
    #把内部智能体设为只读属性，禁止外部直接修改
    @property
    def agent(self):
        """内部 Agent 实例，通常不直接调用"""
        return self._agent
    #监督智能体通过 execute_tool(tool_name, parameters, file_path) 委派，不对外暴露 execute_task。
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
        #获取事件循环，在线程池中执行工具函数
        loop = asyncio.get_event_loop()
        try:
            # 同步工具在默认线程池（None）中执行，避免阻塞事件循环
            #run_in_executor不能直接传带参数的函数调用，需要用lambda包装；lambda 把 “函数调用” 变成 “可传递的函数对象”
            result = await loop.run_in_executor(None, lambda: tool_func.invoke(params))
            return str(result) if result is not None else ""
        except Exception as e:
            return f"错误: {tool_name} 执行失败 - {str(e)}"