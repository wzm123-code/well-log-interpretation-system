"""
智能录井解释系统 - 后端主入口

【架构】FastAPI 提供 REST 接口：文件上传、流式对话、报告下载。
工作流由 SupervisorAgent 驱动，通过 event_callback 将 tool_start/tool_end/summary_chunk 等事件推送到前端的 SSE 流。
"""
import os
import uuid
import shutil
import json
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.data_agent import DataAgent
from agents.expert_agent import ExpertAgent
from agents.supervisor_agent import SupervisorAgent
from storage.conversation_history import (
    append_pair,
    delete_session,
    get_history,
    get_messages_json,
    get_session_meta,
    list_saved_sessions,
    save_session_meta,
    update_session_title,
)
from utils.conversation_title_llm import generate_session_title_llm
from tools.supervisor_tools import extract_data_files
from utils.excel_utils import excel_to_csv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保目录存在（使用绝对路径，避免运行目录影响）
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="智能录井解释系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8501",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# 专用图表接口：显式 FileResponse 返回，避免 StaticFiles 可能的跨域/路径问题
@app.get("/api/chart/{task_id}/{filename:path}")
async def serve_chart(task_id: str, filename: str):
    """提供图表图片，确保正确 Content-Type 和 Cache 头"""
    if not filename or ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="无效的文件名")
    safe = os.path.basename(filename)
    if not any(safe.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".html")):
        raise HTTPException(status_code=400, detail="仅支持图表格式")
    path = os.path.join(STATIC_DIR, task_id, safe)
    if not os.path.isfile(path):
        logger.warning(f"图表不存在: {path}")
        raise HTTPException(status_code=404, detail="图表不存在")
    ext = safe.lower().split(".")[-1]
    media = "text/html" if ext == "html" else "image/png" if ext == "png" else "image/jpeg" if ext in ("jpg", "jpeg") else "image/gif"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "public, max-age=3600"})

# 挂载静态文件（任务上传文件：/static/{task_id}/...）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: StarletteRequest, exc: RequestValidationError):
    logger.error(f"422错误: {exc.errors()}")
    logger.error(f"请求体: {await request.body()}")
    return StarletteJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": (await request.body()).decode()}
    )


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """API 根路径说明（界面请使用 Streamlit：streamlit run frontend/main.py）"""
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"/><title>智能录井解释系统 · API</title>
<style>body{font-family:system-ui,sans-serif;max-width:42rem;margin:3rem auto;padding:0 1rem;color:#1e293b;line-height:1.6;}
code{background:#f1f5f9;padding:2px 6px;border-radius:4px}</style></head>
<body>
<h1>智能录井解释系统</h1>
<p>本端口提供 <strong>REST / SSE</strong> 接口。请在项目根目录启动 Streamlit 界面：</p>
<pre><code>streamlit run frontend/main.py</code></pre>
<p>默认打开 <a href="http://127.0.0.1:8501">http://127.0.0.1:8501</a>，并在侧栏填写 API 地址为 <code>http://127.0.0.1:8000</code>。</p>
<p><a href="/docs">OpenAPI 文档 (/docs)</a></p>
</body></html>"""
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/api/download_report/{task_id}")
async def download_report(task_id: str):
    """下载解释报告，强制 attachment 模式，不触发浏览器预览"""
    file_path = os.path.join(STATIC_DIR, task_id, "interpretation_report.docx")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="报告不存在，请先完成分析任务")
    return FileResponse(
        file_path,
        filename="interpretation_report.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": "attachment; filename=interpretation_report.docx"},
    )


@app.get("/api/data_files/{task_id}")
async def list_data_files(task_id: str):
    """列出任务目录中可下载的数据文件（清洗、归一化、岩性、储层等 CSV）"""
    work_dir = os.path.join(STATIC_DIR, task_id)
    files = extract_data_files(work_dir)
    return {"files": files}


@app.get("/api/download_file/{task_id}/{filename}")
async def download_file(task_id: str, filename: str):
    """下载任务目录中的数据处理结果文件（如清洗后 CSV、岩性结果等）。"""
    if not filename or ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="无效的文件名")
    safe_name = os.path.basename(filename)
    if not safe_name.endswith(".csv"):
        raise HTTPException(status_code=400, detail="仅支持下载 CSV 文件")
    file_path = os.path.join(STATIC_DIR, task_id, safe_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(
        file_path,
        filename=safe_name,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    work_dir = os.path.join(STATIC_DIR, task_id)
    os.makedirs(work_dir, exist_ok=True)

    safe_name = file.filename or "data"
    file_path = os.path.join(work_dir, safe_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_path = excel_to_csv(file_path, suffix="_data")

    return JSONResponse(content={
        "task_id": task_id,
        "file_path": file_path,
        "file_url": f"/static/{task_id}/{os.path.basename(file_path)}"
    })


class SaveConversationBody(BaseModel):
    conversation_id: str
    title: str = ""
    task_id: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None


class ArchiveBeforeNewBody(BaseModel):
    """新建对话前归档当前会话：由服务端根据对话生成标题并写入书签。"""

    conversation_id: str
    task_id: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


class UpdateTitleBody(BaseModel):
    title: str


def _build_dialogue_for_title(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        c = (m.get("content") or "").strip()
        if not c:
            continue
        if role == "user":
            parts.append(f"用户：{c[:8000]}")
        elif role == "assistant":
            parts.append(f"助手：{c[:8000]}")
    return "\n".join(parts)[:12000]


@app.get("/api/conversations")
async def api_list_conversations():
    """列出已保存的会话（书签），含可选的 task_id / 文件路径，供前端恢复。"""
    return {"sessions": list_saved_sessions()}


@app.post("/api/conversations/archive_before_new")
async def api_archive_before_new(body: ArchiveBeforeNewBody):
    """
    用户点击「新建对话」时调用：用 LLM 根据对话内容生成标题并保存上一会话书签。
    若无有效用户发言则跳过。
    """
    cid = (body.conversation_id or "").strip()
    if not cid:
        return {"ok": True, "skipped": True, "reason": "no_conversation_id"}
    msgs = body.messages if body.messages else []
    if msgs:
        dialogue = _build_dialogue_for_title(msgs)
    else:
        dialogue = _build_dialogue_for_title(get_history(cid))
    if not dialogue.strip():
        return {"ok": True, "skipped": True, "reason": "no_messages"}
    if "用户：" not in dialogue:
        return {"ok": True, "skipped": True, "reason": "no_user_messages"}
    title = await asyncio.to_thread(generate_session_title_llm, dialogue)
    save_session_meta(
        cid,
        title,
        task_id=body.task_id or "",
        file_path=body.file_path or "",
        file_name=body.file_name or "",
    )
    return {"ok": True, "skipped": False, "title": title, "conversation_id": cid}


@app.patch("/api/conversations/{conversation_id}/title")
async def api_update_conversation_title(conversation_id: str, body: UpdateTitleBody):
    """修改已保存会话的标题（用户主动改名）。"""
    cid = (conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="无效的会话 ID")
    if not get_session_meta(cid):
        raise HTTPException(status_code=404, detail="会话不存在，请先保存")
    if not update_session_title(cid, body.title):
        raise HTTPException(status_code=400, detail="标题无效或未变更")
    return {"ok": True, "title": (body.title or "").strip()}


@app.post("/api/conversations/save")
async def api_save_conversation(body: SaveConversationBody):
    """保存或更新当前会话书签（标题与上传任务信息）。"""
    cid = (body.conversation_id or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="conversation_id 不能为空")
    save_session_meta(
        cid,
        body.title or "",
        task_id=body.task_id or "",
        file_path=body.file_path or "",
        file_name=body.file_name or "",
    )
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    """删除会话书签及该会话在库中的全部问答记录。"""
    if not conversation_id.strip():
        raise HTTPException(status_code=400, detail="无效的会话 ID")
    delete_session(conversation_id.strip())
    return {"ok": True}


@app.get("/api/conversations/{conversation_id}/restore")
async def api_restore_conversation(conversation_id: str):
    """恢复已保存会话：元数据 + 问答消息列表（不含前端欢迎语）。"""
    cid = conversation_id.strip()
    if not cid:
        raise HTTPException(status_code=400, detail="无效的会话 ID")
    meta = get_session_meta(cid)
    if not meta:
        raise HTTPException(status_code=404, detail="会话不存在或尚未保存")
    messages = get_messages_json(cid)
    return {"meta": meta, "messages": messages}


@app.post("/chat_stream")
async def chat_stream(
    task_id: str = Form(""),
    file_path: str = Form(""),
    user_request: str = Form(...),
    conversation_id: str = Form(""),
):
    """流式聊天接口，通过 SSE 发送工具执行日志和最终报告"""
    data_agent = DataAgent()
    expert_agent = ExpertAgent()
    supervisor = SupervisorAgent(expert_agent=expert_agent, data_agent=data_agent)

    event_queue = asyncio.Queue()

    async def event_callback(event: dict):
        await event_queue.put(event)

    async def event_generator() -> AsyncGenerator[str, None]:
        # 工作流在后台运行，将事件通过 event_callback 塞入 event_queue
        workflow_task = asyncio.create_task(
            supervisor.execute_workflow(
                user_request,
                file_path if file_path else None,
                event_callback=event_callback,
                conversation_id=conversation_id,
                task_id=task_id,
            )
        )

        # 循环消费队列中的事件，以 SSE 格式推送给前端；超时 0.5s 用于检测工作流是否结束
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                if workflow_task.done() and event_queue.empty():
                    break
                continue

        # 工作流结束后，发送 final 事件（含 summary、charts、report_url）
        try:
            result = workflow_task.result()
            if result["status"] == "success":
                if conversation_id:
                    append_pair(conversation_id, user_request, result["summary"])
                tid = task_id or result.get("task_id") or ""
                if not tid and result.get("report_url"):
                    parts = result["report_url"].strip("/").split("/")
                    if len(parts) >= 2 and parts[0] == "static":
                        tid = parts[1]
                data_files = result.get("data_files", [])
                final_event = {
                    "type": "final",
                    "summary": result["summary"],
                    "charts": result.get("charts", []),
                    "has_report": result.get("has_report", False),
                    "report_url": result.get("report_url"),
                    "data_files": data_files,
                    "task_id": tid,
                }
                yield f"data: {json.dumps(final_event)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_event)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    logger.info("启动服务: http://127.0.0.1:8000/  (Ctrl+C 停止)")
    uvicorn.run(app, host="127.0.0.1", port=8000)