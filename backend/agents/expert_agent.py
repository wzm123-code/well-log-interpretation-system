"""
专家智能体 - 负责地质解释与数据可视化

【职责】
  - interpret_lithology、identify_reservoir：地质解释，输出 CSV
  - analyze_mud_gas_survey、plot_mud_gas_profile：录井气测
  - plot_* 绘图：Plotly 输出 HTML（可交互）+ Matplotlib 输出同名 PNG（供 Word 报告嵌入）

【调用方式】监督智能体通过 execute_tool 委派。绘图类工具受 _plot_semaphore 限制，最多 2 个并发，120 秒超时。

不构建 LangChain ReAct Agent：任务规划由 SupervisorAgent 完成，此处仅同步调用工具，避免重复加载模型与多余 API 消耗。
"""
import asyncio
from typing import Any, Dict

from utils.agent_helpers import set_file_param

from tools.interpretation_tools import interpret_lithology, identify_reservoir
from tools.mud_logging_tools import analyze_mud_gas_survey, plot_mud_gas_profile
from tools.visualization_tools import (
    plot_well_log_curves,
    plot_lithology_distribution,
    plot_crossplot,
    plot_heatmap,
    plot_reservoir_profile,
)

# 绘图类工具：限制并发数以降低多线程下绘图/IO 压力
PLOT_TOOLS = frozenset({
    "plot_well_log_curves", "plot_lithology_distribution", "plot_crossplot",
    "plot_heatmap", "plot_reservoir_profile", "plot_mud_gas_profile",
})

EXPERT_AGENT_TOOLS = {
    "interpret_lithology": interpret_lithology,
    "identify_reservoir": identify_reservoir,
    "analyze_mud_gas_survey": analyze_mud_gas_survey,
    "plot_well_log_curves": plot_well_log_curves,
    "plot_lithology_distribution": plot_lithology_distribution,
    "plot_crossplot": plot_crossplot,
    "plot_heatmap": plot_heatmap,
    "plot_reservoir_profile": plot_reservoir_profile,
    "plot_mud_gas_profile": plot_mud_gas_profile,
}

_plot_semaphore = asyncio.Semaphore(2)


class ExpertAgent:
    """专家智能体封装类 - 仅对外提供 execute_tool 异步接口"""

    def __init__(self, ctx=None):
        self.ctx = ctx

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        file_path: str,
    ) -> str:
        """供监督智能体委派：根据 tool_name 查找工具，注入 file_path；绘图类工具受信号量与超时限制"""
        tool_func = EXPERT_AGENT_TOOLS.get(tool_name)
        if not tool_func:
            return f"错误: 专家智能体不负责工具 {tool_name}"
        params = set_file_param(tool_func, parameters or {}, file_path)
        loop = asyncio.get_event_loop()

        async def _run_tool():
            def _invoke():
                return tool_func.invoke(params)

            if tool_name in PLOT_TOOLS:
                async with _plot_semaphore:
                    return await loop.run_in_executor(None, _invoke)
            return await loop.run_in_executor(None, _invoke)

        try:
            result = await asyncio.wait_for(_run_tool(), timeout=120.0)
            return str(result) if result is not None else ""
        except asyncio.TimeoutError:
            return f"错误: {tool_name} 执行超时（120秒），数据量可能过大，请尝试减少数据或分次分析"
        except Exception as e:
            return f"错误: {tool_name} 执行失败 - {str(e)}"
