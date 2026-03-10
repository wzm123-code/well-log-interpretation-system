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
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request as StarletteRequest
from starlette.responses import JSONResponse as StarletteJSONResponse
from typing import AsyncGenerator
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agents.supervisor_agent import SupervisorAgent
from agents.data_agent import DataAgent
from agents.expert_agent import ExpertAgent
from storage.conversation_history import append_pair
from utils.excel_utils import excel_to_csv

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 确保目录存在
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app = FastAPI(title="智能录井解释系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        # "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 模板配置
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: StarletteRequest, exc: RequestValidationError):
    logger.error(f"422错误: {exc.errors()}")
    logger.error(f"请求体: {await request.body()}")
    return StarletteJSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": (await request.body()).decode()}
    )


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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


def _get_task_work_dir(task_id: str):
    """任务工作目录"""
    return os.path.join(STATIC_DIR, task_id)


@app.get("/api/data_files/{task_id}")
async def list_data_files(task_id: str):
    """列出任务目录中可下载的数据文件（清洗、归一化、岩性、储层等 CSV）"""
    from tools.supervisor_tools import extract_data_files
    work_dir = _get_task_work_dir(task_id)
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
    uvicorn.run(app, host="127.0.0.1", port=8000)