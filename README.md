智能地质录井解释系统

多智能体协作完成测井数据分析、岩性解释、储层识别与报告生成。

 项目结构

项目根目录/
├── backend/                 # FastAPI 后端（工作目录常为 backend/）
│   ├── app.py               # 入口：上传、SSE、下载、会话 API
│   ├── agents/              # supervisor / data / expert 智能体
│   ├── tools/               # 数据处理、解释、可视化、联网搜索等
│   ├── utils/               # Agent 构建、Excel、会话标题 LLM 等
│   ├── config/              # 各智能体 DeepSeek JSON 配置
│   └── storage/             # SQLite 对话持久化
├── frontend/                # Streamlit 界面（原 streamlit_app）
│   └── main.py
├── scripts/                 # Windows 启动脚本（后端 / 前端 / 一键启动）
├── docs/                    # 项目文档（纯文本 .txt，便于复制到 Word）
├── test_data/               # 示例与测试用测井 CSV
├── .streamlit/              # Streamlit 主题等配置
├── 录井智能体.bat      # 双击：调用 scripts\start_all.bat
├── requirements-venv.txt
└── README.md
```

## 运行方式

1. 安装依赖：`pip install -r requirements-venv.txt`
2. 配置 `backend/.env`：`DEEPSEEK_API_KEY=your_key` 等
3. 启动后端：`cd backend && python app.py`（或 `uvicorn app:app --host 127.0.0.1 --port 8000`）
4. 启动界面（项目根目录）：`streamlit run frontend/main.py`，浏览器访问 `http://127.0.0.1:8501`，侧栏 API 填 `http://127.0.0.1:8000`（或直接双击根目录 **`启动智能录井系统.bat`**，由 `scripts\` 内脚本启动后端与前端）

### 在桌面双击启动

- **推荐**：在项目根目录对 **`启动智能录井系统.bat`** 右键 → **发送到 → 桌面快捷方式**（不要复制 bat 到桌面，路径会错）。
- **或**：复制根目录的 **`DesktopLaunch.bat`** 到桌面，用记事本打开，把其中的 `set APP_ROOT=D:\Python\app` 改成你的项目路径，保存后双击。
5. 访问 `http://127.0.0.1:8000/` 可查看 API 说明与 `/docs` OpenAPI 文档

## 数据流概览

- **无文件** → 普通对话
- **有文件** → LLM 判断意图
  - 知识问答 → 普通对话
  - 分析/出图/报告 → preview → 校验 → LLM 任务规划 → 并行执行 → 流式报告
