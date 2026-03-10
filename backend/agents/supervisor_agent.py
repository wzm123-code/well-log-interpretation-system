"""
监督智能体 - 任务调度与多智能体协作核心

【流程】
1. 无文件：直接 LLM 对话（可提醒上传）
2. 有文件且非分析类：知识问答，不走工具
3. 有文件且分析类：preview → 校验测井数据 → LLM 任务规划(JSON) → 按 TASK_TIER 并行执行 → 流式生成报告
【数据链】clean/normalize 输出更新 path_state["data"]；interpret/identify 输出供 plot_lithology_distribution/plot_reservoir_profile 使用。

【工具模块】预览解析、数据校验、文本处理、报告生成等纯函数封装在 tools/supervisor_tools 中。
"""
#导入标准库
import os#操作文件和目录
import re#正则表达式
import json#处理JSON数据
import asyncio
import logging#日志记录
from datetime import datetime#日期和时间
from storage.conversation_history import get_history#获取对话历史
from typing import Dict, List, Optional, Callable, Awaitable, Any#类型提示

#导入第三方库
from utils.agent_builder import build_agent, BaseAgentState, load_config#构建智能体
from storage.memory.memory_saver import get_memory_saver#存储记忆
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage#消息类型

#导入工具
from tools.supervisor_tools import (
    compute_tool_output_path,
    extract_columns_from_preview,
    extract_data_files,
    extract_report_only,
    get_columns_list_from_preview,
    build_interpretation_report_docx,
    strip_task_json,
    validate_well_log_data,
)

# 配置文件路径
SUPERVISOR_AGENT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "../config/supervisor_agent_deepseek_config.json",
)

# 工具归属：data -> DataAgent, expert -> ExpertAgent
TOOL_AGENT_MAP = {
    "preview_data": "data",
    "clean_data": "data",
    "normalize_data": "data",
    "interpret_lithology": "expert",
    "identify_reservoir": "expert",
    "plot_well_log_curves": "expert",
    "plot_lithology_distribution": "expert",
    "plot_crossplot": "expert",
    "plot_heatmap": "expert",
    "plot_reservoir_profile": "expert",
}

# 任务依赖层级（数字越小越先执行）
TASK_TIER = {
    "preview_data": 0,
    "clean_data": 1,
    "normalize_data": 1,
    "interpret_lithology": 2,
    "identify_reservoir": 2,
    "plot_well_log_curves": 3,
    "plot_lithology_distribution": 3,
    "plot_crossplot": 3,
    "plot_heatmap": 3,
    "plot_reservoir_profile": 3,
}


def build_supervisor_agent(ctx=None):
    """构建监督智能体（LLM 不绑定工具，工具由子智能体执行）"""
    agent = build_agent(
        config_path=SUPERVISOR_AGENT_CONFIG,
        tools=[],
        model_provider="deepseek",
        checkpointer=get_memory_saver(),
        state_schema=BaseAgentState,
        ctx=ctx,
    )
    return agent


class SupervisorAgent:
    """
    监督智能体封装类 - 负责任务规划与多智能体调度

    【职责】判断是否走工具流程、调用 data/expert 的 execute_tool、按 TASK_TIER 并行执行、流式生成报告。
    """

    def __init__(self, ctx=None, expert_agent=None, data_agent=None):
        self.ctx = ctx
        self._agent = build_supervisor_agent(ctx)
        self.expert_agent = expert_agent
        self.data_agent = data_agent
        self._system_prompt = (load_config(SUPERVISOR_AGENT_CONFIG).get("sp") or "").strip()#获取提示词

    @property
    def agent(self):
        """内部 LangChain Agent 实例，用于 LLM 对话（不绑定工具）"""
        return self._agent

    def _should_use_tools(self, user_request: str) -> bool:
        """
        粗粒度判定：是否属于“数据分析/解释/出图/出报告”类请求
        只有这种请求才允许进入工具流程。
        """
        if not user_request:
            return False
        q = user_request.strip().lower()#去除空格并转换为小写

        keywords = [
            "测井", "录井", "曲线", "井", "深度",
            "解释", "岩性", "储层", "识别", "评价",
            "预览", "清洗", "标准化", "归一化",
            "图", "画", "绘制", "可视化", "曲线图", "柱状图", "热力图", "交会图", "剖面",
            "报告", "生成报告",
            "分析", "处理",
            # 一些常见英文列名
            "gr", "den", "cnl", "cali", "porosity", "permeability", "lithology",
        ]
        return any(k in q for k in keywords)

    async def _invoke_llm(
        self,
        user_content: str,
        thread_id: str,
        history: Optional[List[dict]] = None,
    ) -> str:
        """
        统一的 LLM 调用入口：把 config 里的 sp 作为 system message 注入
        历史对话：[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
        """
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        result = await self._agent.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content

    def _get_report_llm(self) -> ChatDeepSeek:
        """创建用于报告流式生成的 LLM 实例（无工具绑定）"""
        config = load_config(SUPERVISOR_AGENT_CONFIG)
        mc = config.get("config", {})
        return ChatDeepSeek(
            model=mc.get("model", "deepseek-chat"),
            temperature=mc.get("temperature", 0.3),
            max_tokens=mc.get("max_completion_tokens", 4096),
            timeout=mc.get("timeout", 300),
            top_p=mc.get("top_p", 0.9),
        )

    async def _invoke_llm_stream(
        self,
        user_content: str,
        event_callback: Callable[[Dict], Awaitable[None]],
        history: Optional[List[dict]] = None,
    ) -> str:
        """
        流式调用 LLM，每收到一个 chunk 即通过 event_callback 发送 summary_chunk。
        返回完整文本。
        """
        messages = []
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        if history:
            for h in history:
                role, content = h.get("role"), h.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_content))

        llm = self._get_report_llm()
        full_content: List[str] = []
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full_content.append(chunk.content)
                await event_callback({"type": "summary_chunk", "content": chunk.content})
        return "".join(full_content)

    async def execute_workflow(
        self,
        user_request: str,
        file_path: Optional[str] = None,
        event_callback: Optional[Callable[[Dict], Awaitable[None]]] = None,
        conversation_id: str = "",
    ) -> Dict:
        """
        异步执行工作流：普通对话 / 工具分析（任务分析、并发执行、报告生成）

        规则：
        - 没有文件：永远走普通对话（可提示上传文件以进行数据分析）
        - 有文件但请求不是数据分析类：普通对话
        - 有文件且请求是数据分析类：才进入工具流程
        """
        try:
            current_date = datetime.now().strftime("%Y年%m月%d日")

            has_file = bool(file_path and os.path.exists(file_path))
            if has_file:
                from utils.excel_utils import excel_to_csv
                file_path = excel_to_csv(file_path, suffix="_workflow")

            if not has_file:
                chat_prompt = f"""当前日期: {current_date}
用户请求: {user_request}

请直接以“对话”的方式回答用户问题：
- 不要输出 JSON
- 不要提及“工具调用/任务列表”等内部实现
- 回答内容请用清晰的 Markdown 排版（标题、列表、粗体、小结）

【专业回答要求】当用户询问地质、测井、岩性、储层等专业问题时，请结合地质学与测井学原理，给出专业、细致、有深度的解释。可适当引用行业常用标准（如 SY/T 标准）、典型参数范围和判断依据。回答要条理清晰、有理有据。

如果用户的请求明显是对测井数据做分析（如岩性解释、储层识别、绘制曲线/热力图/剖面图、生成分析报告），
请友好地提醒用户先上传 CSV 或 Excel 格式的测井数据文件，
并说明：通常需要包含的关键曲线/列，如深度、GR、DEN、CNL、CALI、孔隙度、渗透率等。
"""
                history = get_history(conversation_id) if conversation_id else []
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    answer = await self._invoke_llm_stream(chat_prompt, event_callback, history=history)
                else:
                    answer = await self._invoke_llm(chat_prompt, "supervisor-chat", history=history)
                return {
                    "status": "success",
                    "summary": answer,
                    "results": {},
                    "charts": [],
                    "has_report": False,
                }

            # 有文件，但不是分析类请求：也走普通对话（不触发工具）
            if not self._should_use_tools(user_request):
                chat_prompt = f"""当前日期: {current_date}
用户请求: {user_request}
数据文件: {file_path}

本轮对话属于知识问答 / 概念解释 / 方法说明。
请直接以讲解的方式回答用户问题：
- 不需要真正读取数据文件，不要调用任何数据处理、解释或绘图工具
- 不要输出 JSON
- 用结构清晰的 Markdown（小标题 + 列表 + 总结）组织回答

【专业回答要求】当用户询问地质、测井、岩性、储层等专业问题时，请结合地质学与测井学原理，给出专业、细致、有深度的解释。可适当引用行业常用标准、典型参数范围、物理解释和判断依据。回答要条理清晰、有理有据。

如果用户后续想基于该文件做分析，你可以提示他可以说：
“请进行岩性解释”“请进行储层识别”“请生成测井曲线图/交会图/热力图”“请生成综合解释报告”等。
"""
                history = get_history(conversation_id) if conversation_id else []
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    answer = await self._invoke_llm_stream(chat_prompt, event_callback, history=history)
                else:
                    answer = await self._invoke_llm(chat_prompt, "supervisor-chat", history=history)
                return {
                    "status": "success",
                    "summary": answer,
                    "results": {},
                    "charts": [],
                    "has_report": False,
                }

            # ---------- 工具流程：需 DataAgent 和 ExpertAgent ----------
            if not self.data_agent or not self.expert_agent:
                return {
                    "status": "error",
                    "message": "多智能体协作需要 DataAgent 和 ExpertAgent，请检查初始化配置。",
                    "has_report": False,
                }

            # ---------- 第 0 步：preview 先行（获取列信息）----------
            schema_info = ""
            preview_result = ""
            if self.data_agent:
                if event_callback:
                    await event_callback({
                        "type": "tool_start",
                        "tool": "preview_data",
                        "agent": "data",
                        "message": "开始执行 preview_data",
                    })
                try:
                    preview_result = await self.data_agent.execute_tool(
                        "preview_data", {"n_rows": 5}, file_path
                    )
                    schema_info = extract_columns_from_preview(preview_result)
                    if event_callback:
                        await event_callback({
                            "type": "tool_end",
                            "tool": "preview_data",
                            "agent": "data",
                            "message": "完成 preview_data",
                        })
                except Exception as e:
                    schema_info = f"（预览失败: {e}）"
                    if event_callback:
                        await event_callback({
                            "type": "tool_error",
                            "tool": "preview_data",
                            "agent": "data",
                            "message": f"preview_data 执行失败: {str(e)}",
                        })

            # ---------- 数据校验：若非测井数据，提前返回并提示 ----------
            cols_list = get_columns_list_from_preview(preview_result)
            is_valid, valid_msg = validate_well_log_data(preview_result, cols_list)
            if not is_valid:
                summary = f"数据格式提示\n\n{valid_msg}\n\n请您检查上传的文件内容，确认是否为测井数据（通常包含深度列及 GR、DEN、CNL 等测井曲线）。"
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    await event_callback({"type": "summary_chunk", "content": summary})
                return {
                    "status": "success",
                    "summary": summary,
                    "results": {"preview_data": preview_result},
                    "charts": [],
                    "has_report": False,
                }

            # ---------- 第 1 步：LLM 任务规划，输出 JSON（report_only/charts_only/tasks）----------
            analysis_prompt = f"""当前日期: {current_date}
用户请求: {user_request}
数据文件: {file_path}

【数据列信息】（请根据实际列名填写 parameters）
{schema_info}

输出 JSON 格式：{{"report_only": true/false, "charts_only": true/false, "tasks": [{{"tool_name": "工具名", "parameters": {{...}}}}]}}

【意图判断】根据用户表述自行理解：report_only（只要报告不要图）、charts_only（只要图不要报告），两者可均为 false。

【任务规划 - 严格按用户要求，禁止添加多余步骤】
- 用户说"只要X"或"只X"时，严格只规划 X，绝不多加其他任务。
- 只要清洗/只要清洗后的数据 -> 只规划 clean_data
- 只要归一化/只要归一化数据 -> 只规划 normalize_data
- 只要报告 -> 只 interpret_lithology + identify_reservoir，不规划 plot_*
- 只要图/只要曲线图等 -> 只规划对应的 plot_*
- report_only 时：只 interpret + identify，不 plot
- charts_only 时：只 plot_*，不 interpret/identify（除非图需要）
- 不要默认添加 clean_data、normalize_data，除非用户明确要求

可用 tool_name：preview_data, clean_data, normalize_data, interpret_lithology, identify_reservoir, plot_well_log_curves, plot_lithology_distribution, plot_crossplot, plot_heatmap, plot_reservoir_profile
"""
            history = get_history(conversation_id) if conversation_id else []
            analysis_content = await self._invoke_llm(
                analysis_prompt, "supervisor-analysis", history=history
            )
            parsed = self._parse_tasks_and_intent(analysis_content)

            # 解析失败：明确提示，流式输出
            if parsed is None:
                summary = strip_task_json(analysis_content)
                if not summary or summary.strip().startswith("{"):
                    summary = "未能解析您的需求，请具体说明，例如：请进行岩性解释和储层识别、只要清洗后的数据 等。"
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    await event_callback({"type": "summary_chunk", "content": summary})
                return {
                    "status": "success",
                    "summary": summary,
                    "results": {},
                    "charts": [],
                    "has_report": False,
                }

            # 任务校验与过滤
            tasks, report_only, charts_only = parsed
            tasks = self._validate_and_sort_tasks(tasks)

            # ---------- 按 LLM 判断的 report_only 过滤绘图任务 ----------
            data_only_tools = {"clean_data", "normalize_data"}
            is_data_only = all(t.get("tool_name") in data_only_tools for t in tasks) and len(tasks) > 0
            if report_only and not is_data_only:
                plot_tools = {"plot_well_log_curves", "plot_lithology_distribution", "plot_crossplot",
                              "plot_heatmap", "plot_reservoir_profile"}
                tasks = [t for t in tasks if t.get("tool_name") not in plot_tools]
                if not any(t.get("tool_name") == "plot_lithology_distribution" for t in tasks):
                    tasks = [t for t in tasks if t.get("tool_name") != "interpret_lithology"]
                if not any(t.get("tool_name") == "plot_reservoir_profile" for t in tasks):
                    tasks = [t for t in tasks if t.get("tool_name") != "identify_reservoir"]
                has_interpret = any(t.get("tool_name") == "interpret_lithology" for t in tasks)
                has_identify = any(t.get("tool_name") == "identify_reservoir" for t in tasks)
                if not has_interpret:
                    tasks.append({"tool_name": "interpret_lithology", "parameters": {}})
                if not has_identify:
                    tasks.append({"tool_name": "identify_reservoir", "parameters": {}})
                tasks.sort(key=lambda x: (TASK_TIER.get(x.get("tool_name"), 99), x.get("tool_name", "")))

            # 若 LLM 返回空任务但用户明确提到清洗/归一化，补全对应任务
            if not tasks and has_file:
                req_lower = user_request.lower().strip()
                if "清洗" in user_request or "cleaned" in req_lower:
                    tasks = [{"tool_name": "clean_data", "parameters": {}}]
                    is_data_only = True
                elif "归一化" in user_request or "标准化" in user_request or "normaliz" in req_lower:
                    tasks = [{"tool_name": "normalize_data", "parameters": {}}]
                    is_data_only = True
            if not tasks:
                msg = "未解析出有效的分析任务。请具体说明，例如：只要清洗后的数据、请进行岩性解释和储层识别。"
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    await event_callback({"type": "summary_chunk", "content": msg})
                return {
                    "status": "success",
                    "summary": msg,
                    "results": {},
                    "charts": [],
                    "has_report": False,
                }

            # ---------- 第 2 步：按 TASK_TIER 分组，同层级并行执行（tier 2/3 可提速）----------
            tool_results = await self._execute_with_agents(
                tasks, file_path, event_callback
            )
            if preview_result:
                tool_results["preview_data"] = preview_result

            # ---------- 第 3 步：流式生成报告；charts_only 时跳过报告，仅输出图表提示 ----------
            skip_report = charts_only
            report_url = None
            data_files: List[Dict[str, str]] = []
            charts: List[str] = []
            has_report: bool = False

            if skip_report:
                output = "图表已生成，请查看下方可视化区域。"
                charts = self._extract_chart_paths(file_path) if has_file else []
                has_report = False
                if has_file:
                    work_dir = os.path.dirname(file_path)
                    data_files = extract_data_files(work_dir)
            elif is_data_only and has_file:
                output = "数据已处理完成，请在下方「报告与任务状态」区域下载。"
                charts = []
                has_report = False
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    await event_callback({"type": "summary_chunk", "content": output})
                work_dir = os.path.dirname(file_path)
                data_files = extract_data_files(work_dir)
            else:
                summary_prompt = f"""用户请求: {user_request}
数据文件: {file_path}
报告生成日期: {current_date}

工具执行结果（JSON 形式，仅供你参考，不要原样输出给用户）:
{json.dumps(tool_results, indent=2, ensure_ascii=False)}

请基于这些结果，为用户生成报告，要求：
- 直接以 # 摘要 开头，不要任何对话式前导（如"根据您的要求""好的，以下是"等）
- 若用户明确只要报告、只要简要结论，则输出简洁版：摘要+核心结论，不展开长篇论述
- 使用 Markdown 排版：摘要、岩性解释（如适用）、储层识别（如适用）、结论
- 不要使用星号(*)做粗体/斜体，不要直接输出 JSON
- 输出内容中禁止出现 * 符号
"""
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    output = await self._invoke_llm_stream(summary_prompt, event_callback)
                else:
                    output = await self._invoke_llm(summary_prompt, "supervisor-final")
                output = strip_task_json(output)
                charts = self._extract_chart_paths(file_path) if has_file else []
                has_report = len(charts) > 0

                # 生成可下载的报告文件：含结构化报告 + 嵌入图表
                if has_file:
                    work_dir = os.path.dirname(file_path)
                    report_content = extract_report_only(output)
                    report_url = build_interpretation_report_docx(work_dir, report_content, file_path)
                    if report_url:
                        has_report = True
                    data_files = extract_data_files(work_dir)

            task_id_from_path = ""
            if has_file and file_path:
                task_id_from_path = os.path.basename(os.path.dirname(os.path.abspath(file_path)))

            return {
                "status": "success",
                "summary": output,
                "results": tool_results,
                "charts": charts,
                "has_report": has_report,
                "report_url": report_url,
                "data_files": data_files,
                "task_id": task_id_from_path,
            }

        except Exception as e:
            if event_callback:
                await event_callback({"type": "error", "message": str(e)})
            return {"status": "error", "message": str(e), "has_report": False}

    def _parse_tasks_and_intent(
        self, content: str
    ) -> Optional[tuple]:
        """
        解析任务列表及 LLM 判断的意图。
        返回 (tasks, report_only, charts_only)，解析失败返回 None。
        """
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content.strip()
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return (data, False, False)
            if not isinstance(data, dict):
                return None
            tasks = data.get("tasks") or data.get("task_list")
            if tasks is None:
                tasks = []
            if not isinstance(tasks, list):
                return None
            report_only = bool(data.get("report_only", False))
            charts_only = bool(data.get("charts_only", False))
            return (tasks, report_only, charts_only)
        except json.JSONDecodeError:
            return None

    def _validate_and_sort_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """校验任务、补全依赖、按层级排序。跳过 preview（已执行）"""
        valid = []
        names = set()
        for t in tasks:
            name = t.get("tool_name")
            if name and name in TOOL_AGENT_MAP and name != "preview_data":
                valid.append(t)
                names.add(name)
        # 补全数据清洗：interpret/identify 前置 clean_data（仅当非纯数据任务时）
        data_only = names <= {"clean_data", "normalize_data"} and len(names) > 0
        if not data_only and ("interpret_lithology" in names or "identify_reservoir" in names) and "clean_data" not in names:
            valid.insert(0, {"tool_name": "clean_data", "parameters": {}})
            names.add("clean_data")
        # 补全隐式依赖：plot_lithology_distribution 需要 interpret_lithology
        if "plot_lithology_distribution" in names and "interpret_lithology" not in names:
            valid.append({"tool_name": "interpret_lithology", "parameters": {}})
            names.add("interpret_lithology")
        # plot_reservoir_profile 需要 identify_reservoir（储层质量道）、interpret_lithology（岩性道）
        if "plot_reservoir_profile" in names:
            if "identify_reservoir" not in names:
                valid.append({"tool_name": "identify_reservoir", "parameters": {}})
                names.add("identify_reservoir")
            if "interpret_lithology" not in names:
                valid.append({"tool_name": "interpret_lithology", "parameters": {}})
                names.add("interpret_lithology")
        valid.sort(key=lambda x: (TASK_TIER.get(x.get("tool_name"), 99), x.get("tool_name", "")))
        return valid

    def _resolve_input_path(
        self,
        tool_name: str,
        task: Dict,
        path_state: Dict[str, str],
        file_path: str,
    ) -> str:
        """
        根据当前路径状态，确定该任务的输入文件路径。
        plot_lithology_distribution 必须使用 interpret_lithology 的输出。
        """
        if tool_name == "plot_lithology_distribution":
            return path_state.get("interpret_lithology", path_state.get("data", file_path))
        if tool_name == "plot_reservoir_profile":
            return path_state.get("identify_reservoir", path_state.get("data", file_path))
        return path_state.get("data", file_path)

    async def _run_single_task(
        self,
        task: Dict,
        path_state: Dict[str, str],
        file_path: str,
        merged: Dict[str, Any],
        event_callback: Optional[Callable[[Dict], Awaitable[None]]],
    ) -> Dict[str, str]:
        """执行单个任务，返回需要合并到 path_state 的更新。"""
        tool_name = task.get("tool_name")
        params = task.get("parameters", {}) or {}
        agent_type = TOOL_AGENT_MAP.get(tool_name)
        path_updates: Dict[str, str] = {}

        if not agent_type:
            merged[tool_name] = f"错误: 未知工具 {tool_name}"
            return path_updates

        input_path = self._resolve_input_path(tool_name, task, path_state, file_path)
        if not os.path.exists(input_path):
            merged[tool_name] = f"错误: 输入文件不存在 - {input_path}"
            return path_updates

        if event_callback:
            await event_callback({
                "type": "tool_start",
                "tool": tool_name,
                "agent": agent_type,
                "message": f"开始执行 {tool_name}",
            })

        tool_error = None
        try:
            if agent_type == "data" and self.data_agent:
                result = await self.data_agent.execute_tool(tool_name, params, input_path)
            elif agent_type == "expert" and self.expert_agent:
                result = await self.expert_agent.execute_tool(tool_name, params, input_path)
            else:
                result = f"错误: 未找到 {agent_type} 智能体"
            merged[tool_name] = result

            if tool_name in ("clean_data", "normalize_data"):
                path_updates["data"] = compute_tool_output_path(input_path, tool_name, params)
            elif tool_name == "interpret_lithology":
                path_updates["interpret_lithology"] = compute_tool_output_path(
                    input_path, tool_name, params
                )
            elif tool_name == "identify_reservoir":
                path_updates["identify_reservoir"] = compute_tool_output_path(
                    input_path, tool_name, params
                )
        except Exception as e:
            tool_error = str(e)
            merged[tool_name] = f"异常: {tool_error}"
            if event_callback:
                await event_callback({
                    "type": "tool_error",
                    "tool": tool_name,
                    "agent": agent_type,
                    "message": f"{tool_name} 执行失败: {tool_error}",
                })
        if event_callback and not tool_error:
            await event_callback({
                "type": "tool_end",
                "tool": tool_name,
                "agent": agent_type,
                "message": f"完成 {tool_name}",
            })
        return path_updates

    async def _execute_with_agents(
        self,
        tasks: List[Dict],
        file_path: str,
        event_callback: Optional[Callable[[Dict], Awaitable[None]]] = None,
    ) -> Dict:
        """
        按依赖层级执行任务，委派给 DataAgent / ExpertAgent。
        同层级内无依赖的任务（如 tier 2、tier 3）并行执行以加快图表生成。
        """
        path_state: Dict[str, str] = {"data": file_path}
        merged: Dict[str, Any] = {}

        # 按层级分组
        tiers: Dict[int, List[Dict]] = {}
        for task in tasks:
            tier = TASK_TIER.get(task.get("tool_name"), 99)
            tiers.setdefault(tier, []).append(task)

        for tier in sorted(tiers.keys()):
            tier_tasks = tiers[tier]
            if len(tier_tasks) <= 1:
                for task in tier_tasks:
                    updates = await self._run_single_task(
                        task, path_state, file_path, merged, event_callback
                    )
                    path_state.update(updates)
            else:
                # 同层级并行执行（tier 2/3 的图表类任务可显著提速）
                results = await asyncio.gather(*[
                    self._run_single_task(task, dict(path_state), file_path, merged, event_callback)
                    for task in tier_tasks
                ])
                for updates in results:
                    path_state.update(updates)

        return merged

    def _extract_chart_paths(self, file_path: str) -> List[str]:
        """从工作目录提取生成的图表文件路径（转换为 URL）"""
        work_dir = os.path.dirname(file_path)
        task_id = os.path.basename(work_dir)
        charts: List[str] = []
        for f in os.listdir(work_dir):
            if f.endswith((".png", ".html", ".jpg", ".jpeg", ".gif")):
                if f != os.path.basename(file_path):
                    charts.append(f"/static/{task_id}/{f}")
        return charts