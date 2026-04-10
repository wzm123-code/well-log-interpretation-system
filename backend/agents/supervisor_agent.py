"""
监督智能体 - 任务调度与多智能体协作核心

【整体流程】
- 无文件：直接走普通对话，可提示用户上传文件
- 有文件：LLM 判断用户意图
  - 知识/概念问答 → 普通对话
  - 分析/出图/报告 → 工具流程：preview → 数据校验 → 任务规划 → 按层级执行 → 报告生成

【职责】负责任务分解、调用 DataAgent/ExpertAgent 执行工具、按 TASK_TIER 分层并行执行、流式输出报告
"""
import asyncio
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from storage.conversation_history import get_history
from tools.supervisor_tools import (
    build_interpretation_report_docx,
    compute_tool_output_path,
    extract_columns_from_preview,
    extract_data_files,
    extract_report_only,
    get_columns_list_from_preview,
    streamable_text_for_report,
    strip_task_json,
    validate_mud_logging_data,
    validate_plot_task_feasibility,
    validate_well_log_data,
)
from tools.web_search import (
    format_search_raw_for_llm,
    format_search_result_for_reply,
    get_api_key,
    search_sync,
)
from utils.agent_builder import build_agent, load_config
from utils.excel_utils import excel_to_csv

logger = logging.getLogger(__name__)

# ---------- 常量配置 ----------
# 监督智能体 LLM 配置文件路径（包含 system prompt、模型参数等）
SUPERVISOR_AGENT_CONFIG = os.path.join(os.path.dirname(__file__), "../config/supervisor_agent_deepseek_config.json")
# 静态文件根目录：图表需放在此目录下才能被 Web 前端访问
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
# 工具与执行智能体的映射：data -> DataAgent，expert -> ExpertAgent
TOOL_AGENT_MAP = {
    "preview_data": "data", "clean_data": "data", "normalize_data": "data",
    "interpret_lithology": "expert", "identify_reservoir": "expert",
    "analyze_mud_gas_survey": "expert",
    "plot_well_log_curves": "expert", "plot_lithology_distribution": "expert",
    "plot_mud_gas_profile": "expert",
    "plot_crossplot": "expert", "plot_heatmap": "expert", "plot_reservoir_profile": "expert",
}
# 任务层级：数值越小越先执行；同层级任务可并行（0=预览, 1=清洗/归一化, 2=岩性/储层, 3=绘图）
TASK_TIER = {
    "preview_data": 0, "clean_data": 1, "normalize_data": 1,
    "interpret_lithology": 2, "identify_reservoir": 2, "analyze_mud_gas_survey": 2,
    "plot_well_log_curves": 3, "plot_lithology_distribution": 3,
    "plot_mud_gas_profile": 3,
    "plot_crossplot": 3, "plot_heatmap": 3, "plot_reservoir_profile": 3,
}
"""任务层级：0=preview，1=清洗/归一化，2=解释/储层，3=绘图；同层级任务可并行执行"""


def _parse_tasks_and_intent(content: str) -> Optional[tuple]:
    json_match = re.search(r"\{[\s\S]*\}", content)
    try:
        data = json.loads(json_match.group(0) if json_match else content.strip())
        if isinstance(data, list):
            return (data, False, False)
        if not isinstance(data, dict):
            return None
        tasks = data.get("tasks") or data.get("task_list") or []
        if not isinstance(tasks, list):
            return None
        return (tasks, bool(data.get("report_only", False)), bool(data.get("charts_only", False)))
    except (json.JSONDecodeError, AttributeError):
        return None


def _validate_and_sort_tasks(tasks: List[Dict], data_kind: str = "well_log") -> List[Dict]:
    """
    校验任务列表并补充依赖、按 TASK_TIER 排序。

    规则：
    - 过滤非法工具（不在 TOOL_AGENT_MAP 或为 preview_data）
    - 录井气测数据（mud_logging）仅保留录井/通用绘图类工具，剔除测井解释类
    - 若有 interpret/identify 但无 clean_data，自动补 clean_data
    - 若有 plot_lithology_distribution 但无 interpret_lithology，自动补 interpret
    - 若有 plot_reservoir_profile，自动补 identify_reservoir 与 interpret_lithology
    - 按 TASK_TIER 排序，保证 preview < clean < interpret < plot
    """
    valid, names = [], set()
    # 过滤有效任务并去重
    for t in tasks:
        name = t.get("tool_name")
        if name and name in TOOL_AGENT_MAP and name != "preview_data":
            valid.append(t)
            names.add(name)
    if data_kind == "mud_logging":
        mud_allow = {
            "clean_data",
            "normalize_data",
            "analyze_mud_gas_survey",
            "plot_mud_gas_profile",
            "plot_crossplot",
            "plot_heatmap",
        }
        valid = [t for t in valid if t.get("tool_name") in mud_allow]
        names = {t.get("tool_name") for t in valid}
    else:
        valid = [
            t
            for t in valid
            if t.get("tool_name") not in {"analyze_mud_gas_survey", "plot_mud_gas_profile"}
        ]
        names = {t.get("tool_name") for t in valid}
    data_only = names <= {"clean_data", "normalize_data"} and len(names) > 0
    if data_kind == "well_log":
        # 解释/储层前可选清洗：仅删「深度或 GR 缺失」行（勿用全表 dropna，否则测井分段缺失会把行删光）
        if not data_only and ("interpret_lithology" in names or "identify_reservoir" in names) and "clean_data" not in names:
            valid.insert(
                0,
                {
                    "tool_name": "clean_data",
                    "parameters": {
                        "handle_missing": "drop",
                        "drop_scope": "key_curves",
                    },
                },
            )
            names.add("clean_data")
        # 岩性分布图依赖岩性解释结果
        if "plot_lithology_distribution" in names and "interpret_lithology" not in names:
            valid.append({"tool_name": "interpret_lithology", "parameters": {}})
            names.add("interpret_lithology")
        # 储层剖面图依赖储层识别和岩性解释
        if "plot_reservoir_profile" in names:
            if "identify_reservoir" not in names:
                valid.append({"tool_name": "identify_reservoir", "parameters": {}})
                names.add("identify_reservoir")
            if "interpret_lithology" not in names:
                valid.append({"tool_name": "interpret_lithology", "parameters": {}})
                names.add("interpret_lithology")
    else:
        # 录井：分析前可选清洗
        if (
            not data_only
            and "analyze_mud_gas_survey" in names
            and "clean_data" not in names
        ):
            valid.insert(
                0,
                {
                    "tool_name": "clean_data",
                    "parameters": {
                        "handle_missing": "drop",
                        "drop_scope": "key_curves",
                    },
                },
            )
            names.add("clean_data")
    valid.sort(key=lambda x: (TASK_TIER.get(x.get("tool_name"), 99), x.get("tool_name", "")))
    return valid


def _apply_report_only_filter(
    tasks: List[Dict], report_only: bool, user_request: str, data_kind: str = "well_log"
) -> tuple:
    """
    根据 report_only 过滤任务；解析失败时按关键词兜底。

    report_only=True 时：测井保留 interpret+identify；录井保留 analyze_mud_gas_survey。
    返回 (过滤后的 tasks, is_data_only)。
    """
    data_only_tools = {"clean_data", "normalize_data"}
    is_data_only = all(t.get("tool_name") in data_only_tools for t in tasks) and len(tasks) > 0
    plot_tools = {
        "plot_well_log_curves",
        "plot_lithology_distribution",
        "plot_mud_gas_profile",
        "plot_crossplot",
        "plot_heatmap",
        "plot_reservoir_profile",
    }
    if report_only and not is_data_only:
        if data_kind == "mud_logging":
            tasks = [t for t in tasks if t.get("tool_name") not in plot_tools]
            tasks = [
                t
                for t in tasks
                if t.get("tool_name") not in {"interpret_lithology", "identify_reservoir"}
            ]
            if not any(t.get("tool_name") == "analyze_mud_gas_survey" for t in tasks):
                tasks.append({"tool_name": "analyze_mud_gas_survey", "parameters": {}})
        else:
            # 只要报告：移除绘图，保留 interpret+identify
            tasks = [t for t in tasks if t.get("tool_name") not in plot_tools]
            if not any(t.get("tool_name") == "plot_lithology_distribution" for t in tasks):
                tasks = [t for t in tasks if t.get("tool_name") != "interpret_lithology"]
            if not any(t.get("tool_name") == "plot_reservoir_profile" for t in tasks):
                tasks = [t for t in tasks if t.get("tool_name") != "identify_reservoir"]
            for tn in ["interpret_lithology", "identify_reservoir"]:
                if not any(t.get("tool_name") == tn for t in tasks):
                    tasks.append({"tool_name": tn, "parameters": {}})
        tasks.sort(key=lambda x: (TASK_TIER.get(x.get("tool_name"), 99), x.get("tool_name", "")))
    # 解析结果为空时，按用户关键词兜底
    if not tasks:
        req = user_request.lower().strip()
        if "清洗" in user_request or "cleaned" in req:
            tasks, is_data_only = [{"tool_name": "clean_data", "parameters": {}}], True
        elif "归一化" in user_request or "标准化" in user_request or "normaliz" in req:
            tasks, is_data_only = [{"tool_name": "normalize_data", "parameters": {}}], True
    return tasks, is_data_only


def build_supervisor_agent(ctx=None):
    """构建监督智能体（LLM 不绑定工具，工具由子智能体执行）"""
    agent = build_agent(
        config_path=SUPERVISOR_AGENT_CONFIG,
        tools=[],
        model_provider="deepseek",
        ctx=ctx,
    )
    return agent


class SupervisorAgent:
    """
    监督智能体封装类 - 负责任务规划与多智能体调度

    【职责】判断是否走工具流程、调用 data/expert 的 execute_tool、按 TASK_TIER 并行执行、流式生成报告。
    """

    def __init__(self, ctx=None, expert_agent=None, data_agent=None):
        """
        初始化监督智能体。
        :param ctx: 可选上下文（如配置、环境变量等）
        :param expert_agent: 专家智能体，负责岩性解释、储层识别、绘图
        :param data_agent: 数据智能体，负责 preview、清洗、归一化
        """
        self.ctx = ctx
        self._agent = build_supervisor_agent(ctx)
        self.expert_agent = expert_agent
        self.data_agent = data_agent
        self._system_prompt = (load_config(SUPERVISOR_AGENT_CONFIG).get("sp") or "").strip()

    @property
    def agent(self):
        """内部 LangChain Agent 实例，用于 LLM 对话（不绑定工具）"""
        return self._agent

    @staticmethod
    def _thread_id(conversation_id: str, purpose: str) -> str:
        """按会话生成 thread_id，确保不同会话的记忆隔离"""
        cid = conversation_id or "anon"
        return f"conv-{cid}-{purpose}"

    async def _emit_workflow_log(
        self,
        event_callback: Optional[Callable[[Dict], Awaitable[None]]],
        message: str,
    ) -> None:
        """向前端推送人类可读的工作流阶段说明（非工具调用）。"""
        if event_callback:
            await event_callback({"type": "workflow_log", "message": message})

    async def _llm_should_use_tools(
        self, user_request: str, conversation_id: str = ""
    ) -> bool:
        """
        由 LLM 判断用户是否希望对数据进行解析、分析、出图或生成报告。
        返回 True 表示应进入工具流程，False 表示走普通对话。
        解析失败时保守返回 False。
        """
        intent_prompt = f"""用户请求: {user_request}
用户已上传测井数据文件。
请判断：用户是否希望对数据进行解析、分析、解释、绘图或生成报告？
仅输出 JSON：{{"use_tools": true}} 或 {{"use_tools": false}}
- 若用户希望：预览/清洗/标准化数据、岩性解释、储层识别、画图、生成报告、总结数据、整理成文档 等 -> use_tools: true
- 若用户只是：概念咨询、方法说明、闲聊、与数据无关的问答 -> use_tools: false
"""
        history = get_history(conversation_id) if conversation_id else []
        try:
            content = await self._invoke_llm(
                intent_prompt,
                self._thread_id(conversation_id, "intent"),
                history=history,
            )
            json_match = re.search(r"\{[\s\S]*?\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
                return bool(data.get("use_tools", False))
        except (json.JSONDecodeError, KeyError) as e:
            logging.getLogger(__name__).debug(f"LLM 意图解析失败: {e}")
        return False

    async def _llm_should_use_web_search(
        self, user_request: str, conversation_id: str = ""
    ) -> bool:
        """
        由 LLM 判断用户问题是否需要联网搜索实时信息（天气、新闻、政策、当前事件等）。
        返回 True 表示应调用智能搜索生成 API，False 表示走普通 LLM 对话。
        """
        if not get_api_key():
            return False
        intent_prompt = f"""用户问题: {user_request}
请判断：该问题是否需要联网搜索才能获得准确、实时的答案？
仅输出 JSON：{{"use_search": true}} 或 {{"use_search": false}}
- 需要联网：天气、新闻、最新政策、当前事件、今天/本周/近期的XXX、实时行情、某日期的信息 等 -> use_search: true
- 不需要联网：概念解释、方法原理、通用知识、测井/地质专业常识、历史规律 等 -> use_search: false
"""
        history = get_history(conversation_id) if conversation_id else []
        try:
            content = await self._invoke_llm(
                intent_prompt,
                self._thread_id(conversation_id, "search-intent"),
                history=history,
            )
            json_match = re.search(r"\{[\s\S]*?\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
                return bool(data.get("use_search", False))
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"联网搜索意图解析失败: {e}")
        return False

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
        carry = ""
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                carry += chunk.content
                while True:
                    emit, carry = streamable_text_for_report(carry)
                    if not emit:
                        break
                    full_content.append(emit)
                    await event_callback({"type": "summary_chunk", "content": emit})
        # 流结束后：若缓冲中仍有可剥离前缀，继续吐出
        while True:
            emit, carry = streamable_text_for_report(carry)
            if not emit:
                break
            full_content.append(emit)
            await event_callback({"type": "summary_chunk", "content": emit})
        raw = "".join(full_content) + carry
        return strip_task_json(raw)

    async def _summarize_web_search_result(
        self,
        user_request: str,
        search_data: dict,
        conversation_id: str = "",
        event_callback: Optional[Callable[[Dict], Awaitable[None]]] = None,
    ) -> str:
        """
        将联网搜索的原始结果交给 LLM 归纳总结，只返回面向用户的最终回答。
        搜索失败或无可用内容时，退化为原有格式化文案（不经过二次 LLM）。
        """
        if search_data.get("error"):
            await self._emit_workflow_log(event_callback, "搜索接口返回错误，使用简短提示回复")
            text = format_search_result_for_reply(search_data)
            if event_callback:
                await event_callback({"type": "summary_chunk", "content": text})
            return text

        raw = format_search_raw_for_llm(search_data)
        if not raw.strip():
            await self._emit_workflow_log(event_callback, "无可用搜索摘要，使用简短提示回复")
            text = format_search_result_for_reply(search_data)
            if event_callback:
                await event_callback({"type": "summary_chunk", "content": text})
            return text

        current_date = datetime.now().strftime("%Y年%m月%d日")
        system_prompt = (
            "你是信息助手。用户的问题已通过联网得到下方原始材料。"
            "请用长短适中、准确的 Markdown 写出**唯一最终回答**：综合归纳、去重纠错，条理清晰。"
            "不要罗列「参考来源」或大量网页链接；除非用户明确要求查看来源，否则不要附链接列表。"
            "不要写「根据搜索」「网页显示」「综上所述」等生硬套话，直接像自然对话一样回答。"
        )
        user_content = (
            f"当前日期：{current_date}\n\n"
            f"【用户问题】\n{user_request}\n\n"
            f"【联网原始材料】\n{raw}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        llm = self._get_report_llm()

        if event_callback:
            full_content: List[str] = []
            async for chunk in llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    full_content.append(chunk.content)
                    await event_callback({"type": "summary_chunk", "content": chunk.content})
            return "".join(full_content)

        result = await llm.ainvoke(messages)
        out = getattr(result, "content", None) or str(result)
        return (out or "").strip()

    async def execute_workflow(
        self,
        user_request: str,
        file_path: Optional[str] = None,
        event_callback: Optional[Callable[[Dict], Awaitable[None]]] = None,
        conversation_id: str = "",
        task_id: str = "",
    ) -> Dict:
        """
        异步执行工作流：普通对话 / 工具分析（任务分析、并发执行、报告生成）

        规则：
        - 没有文件：永远走普通对话（可提示上传文件以进行数据分析）
        - 有文件：由 LLM 判断是否走工具流程（分析/出图/报告 -> 工具，概念问答 -> 普通对话）

        :param user_request: 用户输入的自然语言请求
        :param file_path: 用户上传的数据文件路径（CSV/Excel），为空则走对话
        :param event_callback: 异步回调，用于流式推送事件（report_stream_start、summary_chunk、tool_start 等）
        :param conversation_id: 会话 ID，用于加载历史对话
        :param task_id: 任务 ID，用于图表复制到 static 时的目录命名
        :return: {"status", "summary", "results", "charts", "has_report", "report_url", "data_files", "task_id"}
        """
        try:
            current_date = datetime.now().strftime("%Y年%m月%d日")

            # 判断是否有有效的数据文件；Excel 转为 CSV 便于后续处理
            has_file = bool(file_path and os.path.exists(file_path))
            if has_file:
                file_path = excel_to_csv(file_path, suffix="_workflow")

            # ---------- 分支 1：无文件 → 纯对话（可联网搜索），提示上传 ----------
            if not has_file:
                await self._emit_workflow_log(event_callback, "无上传文件：进入对话模式")
                await self._emit_workflow_log(event_callback, "正在判断是否需要联网搜索…")
                # 联网搜索：LLM 判断是否需要，若需要则调用智能搜索 API 并返回
                use_web_search = await self._llm_should_use_web_search(user_request, conversation_id)
                if use_web_search:
                    loop = asyncio.get_event_loop()
                    try:
                        await self._emit_workflow_log(event_callback, "意图判定：需要联网搜索")
                        await self._emit_workflow_log(event_callback, "正在调用联网搜索 API…")
                        data = await loop.run_in_executor(None, lambda: search_sync(user_request))
                        await self._emit_workflow_log(event_callback, "联网搜索完成，大模型归纳生成回复…")
                        if event_callback:
                            await event_callback({"type": "report_stream_start"})
                        answer = await self._summarize_web_search_result(
                            user_request, data, conversation_id, event_callback
                        )
                        return {
                            "status": "success",
                            "summary": answer,
                            "results": {},
                            "charts": [],
                            "has_report": False,
                        }
                    except Exception as e:
                        logger.warning(f"联网搜索失败，回退至普通对话: {e}")
                        await self._emit_workflow_log(
                            event_callback, f"联网搜索失败，回退为普通对话（{e}）"
                        )
                        # 失败时继续走普通 LLM
                else:
                    await self._emit_workflow_log(event_callback, "意图判定：无需联网")

                await self._emit_workflow_log(event_callback, "大模型流式生成回复（普通对话）…")
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
                    answer = await self._invoke_llm(
                        chat_prompt,
                        self._thread_id(conversation_id, "chat"),
                        history=history,
                    )
                return {
                    "status": "success",
                    "summary": answer,
                    "results": {},
                    "charts": [],
                    "has_report": False,
                }

            # ---------- 分支 2：有文件 → LLM 判断意图 ----------
            await self._emit_workflow_log(event_callback, "正在判断是否需要数据分析工具链…")
            use_tools = await self._llm_should_use_tools(user_request, conversation_id)
            if not use_tools:
                await self._emit_workflow_log(
                    event_callback, "已上传文件：判定为知识问答（不执行数据分析工具链）"
                )
                await self._emit_workflow_log(event_callback, "正在判断是否需要联网搜索…")
                # 联网搜索：LLM 判定需要实时信息时调用智能搜索 API
                use_web_search = await self._llm_should_use_web_search(user_request, conversation_id)
                if use_web_search:
                    loop = asyncio.get_event_loop()
                    try:
                        await self._emit_workflow_log(event_callback, "意图判定：需要联网搜索")
                        await self._emit_workflow_log(event_callback, "正在调用联网搜索 API…")
                        data = await loop.run_in_executor(None, lambda: search_sync(user_request))
                        await self._emit_workflow_log(event_callback, "联网搜索完成，大模型归纳生成回复…")
                        if event_callback:
                            await event_callback({"type": "report_stream_start"})
                        answer = await self._summarize_web_search_result(
                            user_request, data, conversation_id, event_callback
                        )
                        return {
                            "status": "success",
                            "summary": answer,
                            "results": {},
                            "charts": [],
                            "has_report": False,
                        }
                    except Exception as e:
                        logger.warning(f"联网搜索失败，回退至普通对话: {e}")
                        await self._emit_workflow_log(
                            event_callback, f"联网搜索失败，回退为普通对话（{e}）"
                        )
                        # 失败时继续走普通 LLM
                else:
                    await self._emit_workflow_log(event_callback, "意图判定：无需联网")

                await self._emit_workflow_log(event_callback, "大模型流式生成回复（知识问答）…")
                # LLM 判定为知识问答/概念解释，不调用工具，走普通对话
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
                    answer = await self._invoke_llm(
                        chat_prompt,
                        self._thread_id(conversation_id, "chat"),
                        history=history,
                    )
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

            await self._emit_workflow_log(
                event_callback,
                "已上传文件：判定为数据分析流程（预览 → 校验 → 规划 → 执行工具 → 报告）",
            )

            # ---------- 分支 3：有文件 + 工具流程 → 第 0 步：preview 先行（获取列信息）----------
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

            # ---------- 数据校验：测井 或 录井气测 ----------
            cols_list = get_columns_list_from_preview(preview_result)
            is_valid, valid_msg = validate_well_log_data(preview_result, cols_list)
            data_kind = "well_log"
            if not is_valid:
                is_mud, mud_msg = validate_mud_logging_data(cols_list)
                if is_mud:
                    is_valid = True
                    data_kind = "mud_logging"
                    await self._emit_workflow_log(
                        event_callback, "数据列符合录井气测表（深度+钻时+气测），进入录井分析流程"
                    )
                else:
                    await self._emit_workflow_log(event_callback, "数据格式校验未通过，返回说明（未执行后续工具）")
                    summary = (
                        f"数据格式提示\n\n{valid_msg}\n\n"
                        "若您上传的是录井气测表，请确认包含深度列（Depth/井深）、钻时（Rop/钻时）与气测列（如 Tg、C1、C2…）。\n"
                        f"录井格式说明：{mud_msg}"
                    )
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
            if data_kind == "mud_logging":
                analysis_prompt = f"""当前日期: {current_date}
用户请求: {user_request}
数据文件: {file_path}
数据类型: 录井气测（钻时+全烃/组分，非测井曲线解释）

【数据列信息】（请根据实际列名填写 parameters）
{schema_info}

输出 JSON 格式：{{"report_only": true/false, "charts_only": true/false, "tasks": [{{"tool_name": "工具名", "parameters": {{...}}}}]}}

【意图判断】report_only（只要报告不要图）、charts_only（只要图不要报告），两者可均为 false。

【任务规划 - 严格按用户要求，禁止添加多余步骤】
- 用户说"只要X"或"只X"时，严格只规划 X，绝不多加其他任务。
- 只要清洗/只要清洗后的数据 -> 只规划 clean_data
- 只要归一化 -> 只规划 normalize_data
- 录井综合分析/气测解释/显示评价 -> analyze_mud_gas_survey；需要深度剖面多道图 -> plot_mud_gas_profile；两参数交会 -> plot_crossplot；多列相关 -> plot_heatmap
- 不要规划 interpret_lithology、identify_reservoir、plot_well_log_curves、plot_lithology_distribution、plot_reservoir_profile（当前文件不是测井曲线表）
- 只要报告 -> 只 analyze_mud_gas_survey，不规划 plot_*
- report_only 时：只 analyze_mud_gas_survey，不 plot
- charts_only 时：只 plot_*（如 plot_mud_gas_profile），不重复 analyze（除非作图需要）
- 不要默认添加 clean_data、normalize_data，除非用户明确要求清洗/归一化

可用 tool_name：preview_data, clean_data, normalize_data, analyze_mud_gas_survey, plot_mud_gas_profile, plot_crossplot, plot_heatmap
"""
            else:
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
            await self._emit_workflow_log(event_callback, "LLM 任务规划（解析用户需求与任务列表）…")
            analysis_content = await self._invoke_llm(
                analysis_prompt,
                self._thread_id(conversation_id, "analysis"),
                history=history,
            )
            parsed = _parse_tasks_and_intent(analysis_content)

            # 解析失败：明确提示，流式输出
            if parsed is None:
                await self._emit_workflow_log(event_callback, "任务规划结果无法解析为有效 JSON，输出提示说明")
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
            tasks = _validate_and_sort_tasks(tasks, data_kind)
            tasks, is_data_only = _apply_report_only_filter(tasks, report_only, user_request, data_kind)

            if not tasks:
                await self._emit_workflow_log(event_callback, "未解析出有效工具任务，输出提示说明")
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
            await self._emit_workflow_log(
                event_callback, f"执行工具任务（共 {len(tasks)} 项，按层级并行）…"
            )
            tool_results = await self._execute_with_agents(
                tasks, file_path, event_callback
            )
            if preview_result:
                tool_results["preview_data"] = preview_result

            # ---------- 第 3 步：流式生成报告；charts_only 时跳过报告，仅输出图表提示 ----------
            skip_report = charts_only
            report_url = None
            data_files: List[Dict[str, str]] = []
            charts: List[Dict[str, str]] = []
            has_report: bool = False
            effective_tid: str = ""

            effective_tid = ""
            if skip_report:
                await self._emit_workflow_log(event_callback, "仅生成图表（跳过综合报告流式输出）")
                output = "图表已生成，请查看下方可视化区域。"
                charts, effective_tid = self._extract_chart_paths(file_path, task_id) if has_file else ([], "")
                has_report = False
                if has_file:
                    work_dir = os.path.dirname(file_path)
                    data_files = extract_data_files(work_dir)
            elif is_data_only and has_file:
                await self._emit_workflow_log(event_callback, "数据类任务完成，流式输出完成提示")
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
- **禁止**输出任务规划类 JSON（如 `{{"tasks":`、`task_id`、`report_generation` 等）；不要写代码围栏包裹的 JSON；仅输出 Markdown 报告正文
- 直接以 # 摘要 开头，不要任何对话式前导（如"根据您的要求""好的，以下是"等）
- 若用户明确只要报告、只要简要结论，则输出简洁版：摘要+核心结论；但若工具结果中含岩性解释（interpret_lithology），仍须在「岩性解释」小节保留测井机理与分类依据等核心专业表述，不得仅列百分比
- 当 interpret_lithology 的输出中含「一、数据概况」「四、深度段划分」「五、储层与含油气性提示」等结构化段落时，报告应**吸收并整理**其中的井段范围、有效段、深度段表与规则化储层提示，并在此基础上撰写专业叙述；勿忽略深度段 CSV（*_lithology_segments.csv）所对应的文字摘要
- 用户未强调「极简」时，岩性解释小节应写细写透：覆盖参与曲线列名、数据质量/无效段提示、各类岩性占比与主控岩性、至少若干代表性深度段（顶深、底深、厚度量级或合并段）的文字归纳，以及规则化储层提示；所有数字、井段、分箱米数须与工具 JSON 一致，禁止臆造
- 不要将「深度段划分」整表原样粘贴；应提炼为叙述，必要时仅保留关键几行说明
- 若工具结果中含录井气测分析（analyze_mud_gas_survey），须设「录井气测分析」小节：钻时统计、全烃与组分异常段、干湿趋势、显示段与工具阈值一致；明确录井半定量、迟到时间校正与试油验证等局限；勿与测井岩性解释混为一谈
- 使用 Markdown 排版：# 摘要、# 岩性解释（当工具结果含岩性解释或岩性统计时必填）、# 录井气测分析（当含 analyze_mud_gas_survey 时必填）、# 储层识别与评价（如适用）、# 结论
- 岩性解释章节须专业、详实（在工具结果适用时）：
  - 阐明自然伽马、密度、中子、深电阻率、自然电位、声波时差（若数据中存在且工具已用）的指示意义，及其与泥质、孔隙、流体性质的关系
  - 结合工具输出中的分类规则与阈值，说明各类岩性（含灰岩、泥岩、砂岩、粉砂岩、致密砂岩、泥质粉砂岩、含油气显示砂岩等；后者仅为测井规则化标签，非试油结论）的划分依据，避免只报数字
  - 定量综述各岩性厚度占比或样本占比，指出主控岩性；若数据支持，可简述垂向变化或沉积—成岩意义上的解读（表述需与数据一致，不作无依据推断）
  - 说明本解释方法的假设与局限（如缺密度曲线时灰岩判识的变化、井眼与流体影响等），并给出可复核或后续验证建议
  - 术语与表述宜符合油气测井解释惯例，可参照行业通用认识（如 SY/T 相关测井解释规范中的曲线选用与解释思路），文风严谨、条理分层；小节内可用多级列表，便于审阅
- 不要使用星号(*)做粗体/斜体，不要直接输出 JSON
- 输出内容中禁止出现 * 符号
- 若有工具返回"跳过: 数据无法生成该图表 - ..."，请在报告中明确写出：哪些图表未生成、未生成的原因，以便用户了解
"""
                await self._emit_workflow_log(event_callback, "流式生成综合解释报告…")
                if event_callback:
                    await event_callback({"type": "report_stream_start"})
                    output = await self._invoke_llm_stream(summary_prompt, event_callback)
                else:
                    output = await self._invoke_llm(
                        summary_prompt,
                        self._thread_id(conversation_id, "final"),
                    )
                output = strip_task_json(output)
                charts, effective_tid = self._extract_chart_paths(file_path, task_id) if has_file else ([], "")
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
            # 当图表从 test_data 等非 static 目录复制时，effective_tid 才正确
            result_task_id = effective_tid if effective_tid else task_id_from_path

            return {
                "status": "success",
                "summary": output,
                "results": tool_results,
                "charts": charts,
                "has_report": has_report,
                "report_url": report_url,
                "data_files": data_files,
                "task_id": result_task_id,
            }

        except Exception as e:
            if event_callback:
                await event_callback({"type": "error", "message": str(e)})
            return {"status": "error", "message": str(e), "has_report": False}

    def _resolve_input_path(
        self,
        tool_name: str,
        task: Dict,
        path_state: Dict[str, str],
        file_path: str,
    ) -> str:
        """
        根据当前路径状态，确定该任务的输入文件路径。

        数据链依赖：
        - plot_lithology_distribution 必须使用 interpret_lithology 输出的岩性 CSV
        - plot_reservoir_profile 必须使用 identify_reservoir 输出的储层 CSV
        - analyze_mud_gas_survey / plot_mud_gas_profile 使用清洗/归一化后的 data 或原始文件
        - 其他工具使用 data 路径（清洗/归一化后的数据或原始文件）
        """
        if tool_name == "plot_lithology_distribution":
            return path_state.get("interpret_lithology", path_state.get("data", file_path))
        if tool_name == "plot_reservoir_profile":
            return path_state.get("identify_reservoir", path_state.get("data", file_path))
        if tool_name in ("analyze_mud_gas_survey", "plot_mud_gas_profile"):
            return path_state.get("data", file_path)
        return path_state.get("data", file_path)

    async def _run_single_task(
        self,
        task: Dict,
        path_state: Dict[str, str],
        file_path: str,
        merged: Dict[str, Any],
        event_callback: Optional[Callable[[Dict], Awaitable[None]]],
    ) -> Dict[str, str]:
        """
        执行单个任务，返回需要合并到 path_state 的路径更新。

        流程：解析工具名 → 解析输入路径 → 绘图前可行性校验（若为绘图工具）
        → 调用 DataAgent/ExpertAgent.execute_tool → 根据工具类型更新 path_state
        """
        tool_name = task.get("tool_name")
        params = task.get("parameters", {}) or {}
        agent_type = TOOL_AGENT_MAP.get(tool_name)
        path_updates: Dict[str, str] = {}

        # 未知工具不执行，直接记录错误
        if not agent_type:
            merged[tool_name] = f"错误: 未知工具 {tool_name}"
            return path_updates

        # 解析输入文件路径（考虑岩性/储层图的数据链依赖）
        input_path = self._resolve_input_path(tool_name, task, path_state, file_path)
        if not os.path.exists(input_path):
            merged[tool_name] = f"错误: 输入文件不存在 - {input_path}"
            return path_updates

        # 绘图前可行性校验：若数据列不满足图表要求，则跳过执行并记录原因
        if agent_type == "expert" and tool_name in (
            "plot_well_log_curves", "plot_lithology_distribution", "plot_crossplot",
            "plot_heatmap", "plot_reservoir_profile", "plot_mud_gas_profile",
        ):
            can_run, skip_reason = validate_plot_task_feasibility(tool_name, input_path, params)
            if not can_run:
                merged[tool_name] = f"跳过: 数据无法生成该图表 - {skip_reason}"
                if event_callback:
                    await event_callback({
                        "type": "tool_end",
                        "tool": tool_name,
                        "agent": agent_type,
                        "message": f"跳过 {tool_name}: {skip_reason}",
                    })
                return path_updates

        # 通知前端：任务开始执行
        if event_callback:
            await event_callback({
                "type": "tool_start",
                "tool": tool_name,
                "agent": agent_type,
                "message": f"开始执行 {tool_name}",
            })

        tool_error = None
        try:
            # 根据工具类型委派给对应智能体执行
            if agent_type == "data" and self.data_agent:
                result = await self.data_agent.execute_tool(tool_name, params, input_path)
            elif agent_type == "expert" and self.expert_agent:
                result = await self.expert_agent.execute_tool(tool_name, params, input_path)
            else:
                result = f"错误: 未找到 {agent_type} 智能体"
            merged[tool_name] = result

            # 根据工具输出更新路径状态，供后续任务使用
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
            # 通知前端：任务执行失败
            if event_callback:
                await event_callback({
                    "type": "tool_error",
                    "tool": tool_name,
                    "agent": agent_type,
                    "message": f"{tool_name} 执行失败: {tool_error}",
                })
        # 通知前端：任务执行成功
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

        # 按 TASK_TIER 分组，保证层级顺序
        tiers: Dict[int, List[Dict]] = {}
        for task in tasks:
            tier = TASK_TIER.get(task.get("tool_name"), 99)
            tiers.setdefault(tier, []).append(task)

        for tier in sorted(tiers.keys()):
            tier_tasks = tiers[tier]
            # 同层级仅 1 个任务：串行执行，并累积 path_state
            if len(tier_tasks) <= 1:
                for task in tier_tasks:
                    updates = await self._run_single_task(
                        task, path_state, file_path, merged, event_callback
                    )
                    path_state.update(updates)
            else:
                # 同层级多任务：并行执行，每任务用当前 path_state 副本，结果合并
                results = await asyncio.gather(*[
                    self._run_single_task(task, dict(path_state), file_path, merged, event_callback)
                    for task in tier_tasks
                ])
                for updates in results:
                    path_state.update(updates)

        return merged

    @staticmethod
    def _chart_title_from_filename(filename: str) -> str:
        """根据图表文件名中的关键词推断中文标题（用于前端展示）"""
        name = (filename or "").lower().replace(".png", "").replace(".html", "").replace(".jpg", "")
        if "_curves_plot" in name:
            return "测井曲线图"
        if "_lithology_distribution" in name:
            return "岩性分布图"
        if "_crossplot" in name:
            return "交会图"
        if "_correlation_heatmap" in name or "_heatmap" in name:
            return "相关性热力图"
        if "_reservoir_profile" in name:
            return "储层剖面图"
        if "_mud_gas_profile" in name:
            return "录井气测剖面图"
        return "图表"

    def _extract_chart_paths(
        self, file_path: str, task_id_from_form: str = ""
    ) -> tuple:
        """
        从工作目录提取图表，转换为可访问的 URL 并附带类型标题。
        当 work_dir 不在 static 下（如 test_data）时，将图表复制到 static 以保证 Web 前端可访问。
        返回 (charts, effective_task_id)，charts 为 [{"url": "/api/chart/{tid}/xxx.png", "title": "测井曲线图"}, ...]
        """
        work_dir = os.path.dirname(file_path)
        abs_work = os.path.abspath(work_dir)
        abs_static = os.path.abspath(STATIC_DIR)

        # 收集工作目录下的图表文件（排除数据文件本身）
        raw_files = [
            f
            for f in os.listdir(work_dir)
            if f.endswith((".png", ".html", ".jpg", ".jpeg", ".gif"))
            and f != os.path.basename(file_path)
        ]
        if not raw_files:
            return [], ""

        # 同名同时存在 .html 与 .png 时，前端只展示 HTML（交互）；PNG 留给报告嵌入
        by_stem: Dict[str, List[str]] = {}
        for f in raw_files:
            stem, _ = os.path.splitext(f)
            by_stem.setdefault(stem, []).append(f)
        chart_files: List[str] = []
        for stem in sorted(by_stem.keys()):
            group = by_stem[stem]
            htmls = [x for x in group if x.lower().endswith(".html")]
            if htmls:
                chart_files.append(sorted(htmls)[0])
            else:
                chart_files.extend(sorted(group))

        # 若工作目录已在 static 下，直接构造 URL
        if abs_work.startswith(abs_static):
            tid = os.path.basename(work_dir)
            charts = [
                {"url": f"/api/chart/{tid}/{f}", "title": self._chart_title_from_filename(f)}
                for f in chart_files
            ]
            return charts, tid

        # 工作目录不在 static 下（如 test_data）：复制到 static 以保证 Web 可访问
        target_task_id = task_id_from_form.strip() or str(uuid.uuid4())
        target_dir = os.path.join(STATIC_DIR, target_task_id)
        os.makedirs(target_dir, exist_ok=True)
        # 复制全部原始文件（含 PNG），以便报告与链接一致
        for f in raw_files:
            src = os.path.join(work_dir, f)
            dst = os.path.join(target_dir, f)
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, dst)
                    logger.info(f"图表已复制至 static: {f}")
                except Exception as e:
                    logger.warning(f"复制图表失败 {f}: {e}")

        charts = [
            {"url": f"/api/chart/{target_task_id}/{f}", "title": self._chart_title_from_filename(f)}
            for f in chart_files
        ]
        return charts, target_task_id