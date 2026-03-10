"""
专家智能体 - 负责地质解释与数据可视化

【职责】
  - interpret_lithology、identify_reservoir：地质解释，输出 CSV
  - plot_well_log_curves、plot_lithology_distribution、plot_crossplot、plot_heatmap、plot_reservoir_profile：matplotlib 输出 PNG

【调用方式】监督智能体通过 execute_tool 委派。绘图类工具受 _plot_semaphore 限制，最多 2 个并发，60 秒超时。
"""
#导入标准库
import os
import asyncio
#导入第三方库
from typing import Dict, Any
from utils.agent_builder import build_agent, BaseAgentState
from utils.agent_helpers import set_file_param
from storage.memory.memory_saver import get_memory_saver
from tools.interpretation_tools import interpret_lithology, identify_reservoir
from tools.visualization_tools import (
    plot_well_log_curves,
    plot_lithology_distribution,
    plot_crossplot,
    plot_heatmap,
    plot_reservoir_profile,
)

EXPERT_AGENT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "../config/expert_agent_deepseek_config.json"
)

# 绘图类工具：限制并发数以降低 matplotlib 多线程冲突/卡顿
PLOT_TOOLS = frozenset({
    "plot_well_log_curves", "plot_lithology_distribution", "plot_crossplot",
    "plot_heatmap", "plot_reservoir_profile",
})


# 专家智能体负责的工具
EXPERT_AGENT_TOOLS = {
    "interpret_lithology": interpret_lithology,
    "identify_reservoir": identify_reservoir,
    "plot_well_log_curves": plot_well_log_curves,
    "plot_lithology_distribution": plot_lithology_distribution,
    "plot_crossplot": plot_crossplot,
    "plot_heatmap": plot_heatmap,
    "plot_reservoir_profile": plot_reservoir_profile,
}


def build_expert_agent(ctx=None):
    """根据 EXPERT_AGENT_CONFIG 构建 LangChain ReAct Agent，绑定解释与绘图工具"""
    agent = build_agent(
        config_path=EXPERT_AGENT_CONFIG,
        tools=list(EXPERT_AGENT_TOOLS.values()),
        model_provider="deepseek",
        checkpointer=get_memory_saver(),
        state_schema=BaseAgentState,
        ctx=ctx
    )

    return agent

_plot_semaphore = asyncio.Semaphore(2)  # 最多 2 个绘图任务同时执行

class ExpertAgent:
    """专家智能体封装类 - 仅对外提供 execute_tool 异步接口"""

    def __init__(self, ctx=None):
        self.ctx = ctx
        self._agent = build_expert_agent(ctx)

    @property
    def agent(self):
        """内部 Agent 实例，通常不直接调用"""
        return self._agent

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        file_path: str,
    ) -> str:
        """供监督智能体委派：根据 tool_name 查找工具，注入 file_path；绘图类工具受信号量与 60s 超时限制"""
        tool_func = EXPERT_AGENT_TOOLS.get(tool_name)
        if not tool_func:
            return f"错误: 专家智能体不负责工具 {tool_name}"
        params = set_file_param(tool_func, parameters or {}, file_path)
        loop = asyncio.get_event_loop()

        async def _run_tool():
            def _invoke():
                return tool_func.invoke(params)
            # 绘图工具需获取信号量，限制同时执行的绘图任务数量
            if tool_name in PLOT_TOOLS:
                async with _plot_semaphore:
                    return await loop.run_in_executor(None, _invoke)
            return await loop.run_in_executor(None, _invoke)

        try:
            # 绘图类工具限时 120 秒，避免大数据卡死
            result = await asyncio.wait_for(_run_tool(), timeout=120.0)
            return str(result) if result is not None else ""
        except asyncio.TimeoutError:
            return f"错误: {tool_name} 执行超时（120秒），数据量可能过大，请尝试减少数据或分次分析"
        except Exception as e:
            return f"错误: {tool_name} 执行失败 - {str(e)}"