"""
智能录井解释系统 — Streamlit 前端

使用前请先启动 FastAPI 后端（默认 http://127.0.0.1:8000）：
  cd backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000

再启动本界面：
  cd 项目根目录 && streamlit run frontend/main.py

文件结构概览：
  - 工具函数：URL 拼接、Markdown 清洗、按标题截取章节、SSE 解析
  - init_session：会话状态默认值（对话、任务 ID、右侧各 Tab 展示数据）
  - main：侧栏上传 → 双栏布局 → 消费 /chat_stream 的 SSE 更新界面
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, Generator, List, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------- 配置与工具 ----------

DEFAULT_API = "http://127.0.0.1:8000"

WELCOME_MESSAGE = {
    "role": "assistant",
    "content": "您好，我是**智能录井解释助手**。请在侧栏上传测井数据（CSV / Excel），然后描述分析需求，例如「请进行岩性解释和储层识别」。",
}


def messages_for_archive(state_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """去掉首条欢迎语，供后端生成标题与归档（与 WELCOME_MESSAGE 一致则跳过）。"""
    out: List[Dict[str, str]] = []
    skip_welcome = True
    welcome = (WELCOME_MESSAGE.get("content") or "").strip()
    for m in state_messages:
        if skip_welcome and m.get("role") == "assistant":
            if (m.get("content") or "").strip() == welcome:
                skip_welcome = False
                continue
        skip_welcome = False
        out.append(
            {
                "role": str(m.get("role") or ""),
                "content": (m.get("content") or "")[:20000],
            }
        )
    return out


def reset_chat_state() -> None:
    """新建对话：新会话 ID，清空消息为欢迎语，并清空任务与右侧分析区。"""
    st.session_state.conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
    st.session_state.messages = [dict(WELCOME_MESSAGE)]
    st.session_state.tool_events = []
    st.session_state.pending_response = False
    st.session_state.task_id = ""
    st.session_state.file_path = ""
    st.session_state.file_name = ""
    st.session_state.overview_text = ""
    st.session_state.lithology_text = ""
    st.session_state.reservoir_text = ""
    st.session_state.charts = []
    st.session_state.report_url = ""
    st.session_state.data_files = []
    st.session_state.used_tools = False
    st.session_state.status = "idle"


TOOL_LABELS = {
    "preview_data": "数据预览",
    "clean_data": "数据清洗",
    "normalize_data": "数据标准化",
    "interpret_lithology": "岩性解释",
    "identify_reservoir": "储层识别",
    "plot_well_log_curves": "测井曲线图",
    "plot_lithology_distribution": "岩性分布图",
    "plot_crossplot": "交会图",
    "plot_heatmap": "相关性热力图",
    "plot_reservoir_profile": "储层剖面图",
}


def format_workflow_log_display(
    evs: Optional[List[Dict[str, Any]]], *, pending: bool
) -> str:
    """将 tool_events 格式化为多行文本；pending 且无事件时显示等待提示。"""
    evs = evs or []
    if not evs:
        if pending:
            return "连接中，等待工作流事件…"
        return "暂无工作流记录（发起对话后，此处会显示阶段说明与工具调用）"
    lines: List[str] = []
    for e in evs[-40:]:
        t = e.get("type", "")
        if t == "workflow_log":
            lines.append(f"◆ {e.get('message', '')}")
            continue
        tool = e.get("tool", "")
        msg = e.get("message", "")
        label = TOOL_LABELS.get(tool, tool)
        tag = {"tool_start": "开始", "tool_end": "完成", "tool_error": "失败"}.get(t, t)
        lines.append(f"· [{tag}] {label} — {msg}")
    return "\n".join(lines)


def update_workflow_log_placeholder(
    ph: Any, evs: Optional[List[Dict[str, Any]]], *, pending: bool
) -> None:
    """刷新工作流日志占位符（SSE 循环中调用以实现实时刷新）。"""
    if ph is None:
        return
    ph.text(format_workflow_log_display(evs, pending=pending))


def abs_url(api_base: str, path: str) -> str:
    if not path:
        return path
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base = api_base.rstrip("/")
    if path.startswith("/"):
        return base + path
    return base + "/" + path


def sanitize_assistant_markdown(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned, flags=re.IGNORECASE)
    trimmed = cleaned.lstrip()
    if trimmed.startswith("{"):
        depth, end = 0, -1
        for i, c in enumerate(trimmed):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end >= 0:
            rest = trimmed[end + 1 :].lstrip()
            cleaned = rest if not rest.startswith("{") else sanitize_assistant_markdown(rest)
    return cleaned.strip()


def extract_section(md: str, keyword: str) -> str:
    if not md:
        return ""
    lines = md.splitlines()
    collecting = False
    buf: List[str] = []
    for line in lines:
        t = line.strip()
        if t.startswith("#"):
            if collecting:
                break
            if keyword in t:
                collecting = True
                buf.append(line)
        elif collecting:
            buf.append(line)
    return "\n".join(buf).strip()


def is_chart_html_url(url: str) -> bool:
    if not url:
        return False
    path = url.split("?")[0].split("#")[0]
    return path.lower().endswith(".html")


def iter_chat_sse(
    api_base: str,
    task_id: str,
    file_path: str,
    user_request: str,
    conversation_id: str,
    timeout: int = 600,
) -> Generator[Dict[str, Any], None, None]:
    data = {
        "user_request": user_request,
        "conversation_id": conversation_id,
        "task_id": task_id or "",
        "file_path": file_path or "",
    }
    url = abs_url(api_base, "/chat_stream")
    with requests.post(url, data=data, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def inject_css() -> None:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;1,9..40,400&family=Noto+Sans+SC:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans SC', 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(165deg, #0f172a 0%, #1e293b 40%, #0f172a 100%); }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important; border-right: 1px solid rgba(148,163,184,0.15); }
    section[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stChatMessage"]) { max-width: 100%; }
    [data-testid="stChatMessage"] { background: rgba(30,41,59,0.55); border-radius: 14px; border: 1px solid rgba(148,163,184,0.12); padding: 0.75rem 1rem; margin-bottom: 0.6rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: rgba(15,23,42,0.5); padding: 8px; border-radius: 12px; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #3b82f6, #6366f1) !important; color: white !important; border-radius: 10px !important; }
    h1 { color: #f1f5f9 !important; font-weight: 600 !important; letter-spacing: -0.02em; }
    .status-pill { display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.8rem; font-weight: 500; }
    .status-idle { background: rgba(148,163,184,0.2); color: #cbd5e1; }
    .status-run { background: rgba(59,130,246,0.25); color: #93c5fd; }
    .status-ok { background: rgba(34,197,94,0.2); color: #86efac; }
    .status-err { background: rgba(239,68,68,0.2); color: #fca5a5; }
</style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    defaults = {
        "messages": [dict(WELCOME_MESSAGE)],
        "task_id": "",
        "file_path": "",
        "file_name": "",
        "conversation_id": f"conv_{uuid.uuid4().hex[:12]}",
        "pending_response": False,
        "overview_text": "",
        "lithology_text": "",
        "reservoir_text": "",
        "charts": [],
        "report_url": "",
        "data_files": [],
        "tool_events": [],
        "used_tools": False,
        "status": "idle",
        "_restore_cid": "",  # 侧栏选择「打开」后待恢复的会话 ID
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main() -> None:
    st.set_page_config(
        page_title="智能录井解释系统",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_session()

    # 本轮开始处理 SSE 前清空，避免右侧先画出上一轮日志；与下方 col_chat 内追加事件同步
    if st.session_state.pending_response:
        st.session_state.tool_events = []

    _arch_title = st.session_state.pop("_flash_archived_title", "")
    if _arch_title:
        st.sidebar.success(f"已自动保存上一会话：{_arch_title}")

    st.sidebar.markdown("### ⚙️ 连接与数据")
    api_base = st.sidebar.text_input("API 根地址", value=DEFAULT_API, help="FastAPI 后端地址，需先启动 uvicorn")

    # 从历史列表触发「打开」后，拉取后端消息与元数据并灌入界面
    rid = st.session_state.pop("_restore_cid", "") or ""
    if rid:
        try:
            rr = requests.get(abs_url(api_base, f"/api/conversations/{rid}/restore"), timeout=30)
            rr.raise_for_status()
            payload = rr.json()
            meta = payload.get("meta") or {}
            hist = payload.get("messages") or []
            st.session_state.conversation_id = meta.get("conversation_id") or rid
            st.session_state.messages = [dict(WELCOME_MESSAGE)] + list(hist)
            st.session_state.task_id = meta.get("task_id") or ""
            st.session_state.file_path = meta.get("file_path") or ""
            st.session_state.file_name = meta.get("file_name") or ""
            st.session_state.overview_text = ""
            st.session_state.lithology_text = ""
            st.session_state.reservoir_text = ""
            st.session_state.charts = []
            st.session_state.report_url = ""
            st.session_state.data_files = []
            st.session_state.tool_events = []
            st.session_state.used_tools = False
            st.session_state.status = "idle"
            st.session_state.pending_response = False
            st.sidebar.success("已打开历史会话")
        except requests.RequestException as ex:
            st.sidebar.error(f"加载历史会话失败：{ex}")

    st.sidebar.markdown("### 💬 对话管理")
    if st.sidebar.button("➕ 新建会话"):
        try:
            ar = requests.post(
                abs_url(api_base, "/api/conversations/archive_before_new"),
                json={
                    "conversation_id": st.session_state.conversation_id,
                    "task_id": st.session_state.task_id or "",
                    "file_path": st.session_state.file_path or "",
                    "file_name": st.session_state.file_name or "",
                    "messages": messages_for_archive(st.session_state.messages),
                },
                timeout=120,
            )
            if ar.ok:
                j = ar.json()
                if j.get("ok") and not j.get("skipped") and j.get("title"):
                    st.session_state["_flash_archived_title"] = j["title"]
        except requests.RequestException:
            pass
        reset_chat_state()
        st.rerun()

    with st.sidebar.expander("历史对话", expanded=False):
        try:
            lr = requests.get(abs_url(api_base, "/api/conversations"), timeout=15)
            lr.raise_for_status()
            sessions: List[Dict[str, Any]] = lr.json().get("sessions") or []
        except requests.RequestException:
            sessions = []
            st.caption("无法连接后端，历史列表不可用")

        if not sessions:
            st.caption("暂无已保存会话；「新建会话」将自动保存上一会话并归档")
        else:
            st.caption("点击条目打开 · 右侧 ✕ 删除")
            for s in sessions[:25]:
                cid = s.get("conversation_id") or ""
                title = (s.get("title") or "").strip() or cid
                label = title if len(title) <= 40 else title[:39] + "…"
                row_open, row_x = st.columns([11, 1])
                with row_open:
                    if st.button(
                        label,
                        key=f"hist_open_{cid}",
                        type="secondary",
                        use_container_width=True,
                        help="打开此会话",
                    ):
                        st.session_state["_restore_cid"] = cid
                        st.rerun()
                with row_x:
                    if st.button(
                        "✕",
                        key=f"hist_del_{cid}",
                        type="tertiary",
                        help="删除此会话",
                    ):
                        try:
                            dr = requests.delete(
                                abs_url(api_base, f"/api/conversations/{cid}"), timeout=20
                            )
                            dr.raise_for_status()
                            if st.session_state.conversation_id == cid:
                                reset_chat_state()
                            st.rerun()
                        except requests.RequestException as ex:
                            st.error(f"删除失败：{ex}")

    uploaded = st.sidebar.file_uploader("上传测井数据", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        if st.sidebar.button("上传到服务器", type="primary"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
                r = requests.post(abs_url(api_base, "/upload"), files=files, timeout=120)
                r.raise_for_status()
                data = r.json()
                st.session_state.task_id = data.get("task_id", "")
                st.session_state.file_path = data.get("file_path", "")
                st.session_state.file_name = uploaded.name
                st.sidebar.success(f"已上传：{uploaded.name}")
            except Exception as e:
                st.sidebar.error(f"上传失败：{e}")

    st.sidebar.caption("当前文件")
    st.sidebar.text(st.session_state.file_name or "（未上传）")
    if st.session_state.task_id:
        st.sidebar.code(st.session_state.task_id, language=None)

    st.markdown("# 📊 智能地质录井解释系统")
    st.caption("左侧对话 · 右侧分析结果与图表 · 后端需保持运行")

    workflow_log_ph: Any = None
    col_chat, col_right = st.columns([1.05, 1], gap="large")

    with col_right:
        st.markdown("##### 分析结果")
        tab_ov, tab_lith, tab_res, tab_ch, tab_dl, tab_log = st.tabs(
            ["摘要概览", "岩性解释", "储层评价", "可视化", "下载", "工作流日志"]
        )
        with tab_ov:
            if st.session_state.overview_text:
                st.markdown(st.session_state.overview_text)
            else:
                st.info("完成一次带数据的分析后，此处显示报告摘要。")
        with tab_lith:
            if st.session_state.lithology_text:
                st.markdown(st.session_state.lithology_text)
            else:
                st.caption("暂无岩性章节")
        with tab_res:
            if st.session_state.reservoir_text:
                st.markdown(st.session_state.reservoir_text)
            else:
                st.caption("暂无储层章节")
        with tab_ch:
            chart_list: List[Dict[str, Any]] = st.session_state.charts or []
            if not chart_list:
                st.caption("暂无图表")
            else:
                ts_bust = str(uuid.uuid4().hex[:10])
                for i, c in enumerate(chart_list[:8]):
                    url = c.get("url", "") if isinstance(c, dict) else str(c)
                    title = c.get("title", f"图表 {i+1}") if isinstance(c, dict) else "图表"
                    if not url:
                        continue
                    path_only = url.split("?")[0].split("#")[0]
                    iframe_src = abs_url(api_base, path_only) + "?t=" + ts_bust
                    img_src = iframe_src
                    st.markdown(f"**{title}**")
                    if is_chart_html_url(url):
                        components.iframe(iframe_src, height=420, scrolling=True)
                    else:
                        st.image(img_src, use_container_width=True)
        with tab_dl:
            tid = st.session_state.task_id or ""
            if tid:
                report_link = abs_url(api_base, f"/api/download_report/{tid}")
                st.markdown(f"[📄 下载 Word 报告]({report_link})")
            else:
                st.caption("分析完成后可下载报告")
            dfs = st.session_state.data_files or []
            if dfs:
                st.markdown("**数据文件**")
                for f in dfs:
                    fn = f.get("filename") or f.get("name") or ""
                    if not fn:
                        continue
                    href = abs_url(api_base, f"/api/download_file/{tid}/{fn}")
                    st.markdown(f"- [{fn}]({href})")
        with tab_log:
            workflow_log_ph = st.empty()
            update_workflow_log_placeholder(
                workflow_log_ph,
                st.session_state.tool_events,
                pending=st.session_state.pending_response,
            )

    with col_chat:
        status = st.session_state.status
        pill = "status-idle"
        label = "空闲"
        if status == "processing":
            pill, label = "status-run", "处理中…"
        elif status == "done":
            pill, label = "status-ok", "分析完成"
        elif status == "error":
            pill, label = "status-err", "出错"
        st.markdown(f'<span class="status-pill {pill}">{label}</span>', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.pending_response:
            assistant_text = ""
            used_tools = False
            st.session_state.status = "processing"
            last_user = ""
            for m in reversed(st.session_state.messages):
                if m.get("role") == "user":
                    last_user = m.get("content", "")
                    break
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    for event in iter_chat_sse(
                        api_base,
                        st.session_state.task_id,
                        st.session_state.file_path,
                        last_user,
                        st.session_state.conversation_id,
                    ):
                        et = event.get("type", "")
                        if et in ("tool_start", "tool_end", "tool_error"):
                            used_tools = True
                            st.session_state.tool_events = st.session_state.tool_events + [event]
                            update_workflow_log_placeholder(
                                workflow_log_ph, st.session_state.tool_events, pending=True
                            )
                        elif et == "workflow_log":
                            st.session_state.tool_events = st.session_state.tool_events + [event]
                            update_workflow_log_placeholder(
                                workflow_log_ph, st.session_state.tool_events, pending=True
                            )
                        elif et == "report_stream_start":
                            assistant_text = ""
                            placeholder.markdown(assistant_text)
                        elif et == "summary_chunk":
                            assistant_text += event.get("content") or ""
                            placeholder.markdown(assistant_text)
                        elif et == "final":
                            assistant_text = sanitize_assistant_markdown(event.get("summary") or "")
                            placeholder.markdown(assistant_text)
                            ts = str(uuid.uuid4().hex[:8])
                            raw_charts = event.get("charts") or []
                            if raw_charts:
                                used_tools = True
                            charts_out = []
                            for p in raw_charts:
                                if isinstance(p, dict) and p.get("url"):
                                    u = p["url"]
                                    sep = "&" if "?" in u else "?"
                                    charts_out.append(
                                        {
                                            "url": u + sep + f"t={ts}",
                                            "title": p.get("title") or "图表",
                                        }
                                    )
                                elif isinstance(p, str):
                                    sep = "&" if "?" in p else "?"
                                    charts_out.append({"url": p + sep + f"t={ts}", "title": "图表"})
                            st.session_state.charts = charts_out
                            ru = event.get("report_url")
                            st.session_state.report_url = abs_url(api_base, ru) if ru else ""
                            st.session_state.data_files = event.get("data_files") or []
                            tid = event.get("task_id") or st.session_state.task_id
                            if tid:
                                st.session_state.task_id = tid
                                try:
                                    dr = requests.get(
                                        abs_url(api_base, f"/api/data_files/{tid}"), timeout=30
                                    )
                                    if dr.ok:
                                        j = dr.json()
                                        if isinstance(j.get("files"), list) and j["files"]:
                                            st.session_state.data_files = j["files"]
                                except requests.RequestException:
                                    pass
                            if assistant_text:
                                if used_tools:
                                    st.session_state.overview_text = extract_section(
                                        assistant_text, "摘要"
                                    ) or assistant_text
                                    st.session_state.lithology_text = extract_section(
                                        assistant_text, "岩性解释"
                                    ) or "（本轮未单独拆分岩性章节）"
                                    st.session_state.reservoir_text = (
                                        extract_section(assistant_text, "储层识别与评价")
                                        or extract_section(assistant_text, "储层识别")
                                        or "（本轮未单独拆分储层章节）"
                                    )
                                else:
                                    st.session_state.overview_text = assistant_text
                                    st.session_state.lithology_text = ""
                                    st.session_state.reservoir_text = ""
                            st.session_state.used_tools = used_tools
                            st.session_state.status = "done" if used_tools else "idle"
                        elif et == "error":
                            assistant_text = f"**错误：** {event.get('message', '未知错误')}"
                            placeholder.markdown(assistant_text)
                            st.session_state.status = "error"
                except requests.RequestException as ex:
                    assistant_text = f"请求失败：{ex}"
                    placeholder.markdown(assistant_text)
                    st.session_state.status = "error"
                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.session_state.pending_response = False
            st.rerun()

        if prompt := st.chat_input("描述您的分析需求…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.pending_response = True
            st.session_state.status = "processing"
            st.rerun()


if __name__ == "__main__":
    main()
