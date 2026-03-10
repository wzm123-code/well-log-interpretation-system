/**
 * 智能录井解释系统 - 前端主应用
 *
 * 【结构】左侧对话栏（流式 SSE）+ 右侧结果区（概览/岩性/储层/图表/工作流日志）。
 * 【流式】chat_stream 返回 SSE，事件类型：tool_start、tool_end、report_stream_start、summary_chunk、final。
 * 【滚动】仅在 scrollTrigger 变化时滚动到底部，流式 chunk 不触发，避免界面自动下滑。
 */
const { useState, useEffect, useRef } = React;

// 可根据部署修改，例如 '/api'
const API_BASE = "http://127.0.0.1:8000";

// ------------ 工具函数 ------------

// 简单 CSV 解析（只为预览前 50 行）
function parseCsv(text, maxRows = 50) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return { headers: [], rows: [] };

  const parseLine = (line) => {
    const res = [];
    let cur = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (c === "," && !inQuotes) {
        res.push(cur);
        cur = "";
      } else {
        cur += c;
      }
    }
    res.push(cur);
    return res;
  };

  const headers = parseLine(lines[0]);
  const rows = lines.slice(1, maxRows + 1).map(parseLine);
  return { headers, rows };
}

// 将后端返回的 /static/... 转为指向 8000 的绝对 URL
function toAbsoluteUrl(path) {
  if (!path) return path;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  if (path.startsWith("/")) {
    return API_BASE.replace(/\/$/, "") + path;
  }
  return path;
}

// 去掉模型生成的内部 JSON 代码块（任务规划等），避免在对话和报告中展示
function sanitizeAssistantMarkdown(text) {
  if (!text) return "";
  let cleaned = text;

  // 1) 删除所有代码块
  cleaned = cleaned.replace(/```[\s\S]*?```/gi, "");
  // 2) 移除开头的 JSON 对象（如 {"task_list": []}）
  const trimmed = cleaned.trimStart();
  if (trimmed.startsWith("{")) {
    let depth = 0, end = -1;
    for (let i = 0; i < trimmed.length; i++) {
      if (trimmed[i] === "{") depth++;
      else if (trimmed[i] === "}") { depth--; if (depth === 0) { end = i; break; } }
    }
    if (end >= 0) {
      const rest = trimmed.slice(end + 1).trimStart();
      cleaned = rest.startsWith("{") ? sanitizeAssistantMarkdown(rest) : rest;
    }
  }
  // 3) 兜底：移除行内 JSON 块
  cleaned = cleaned.replace(/^\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\n?/gm, "");
  return cleaned.trim();
}

// Markdown 渲染（用于左侧对话 + 右侧卡片）
function renderMarkdown(content) {
  if (!content) return "";
  if (window.marked) {
    return window.marked.parse(content, { breaks: true });
  }
  return content.replace(/\n/g, "<br/>");
}

// 工具名 -> 中文显示
const TOOL_LABELS = {
  preview_data: "数据预览",
  clean_data: "数据清洗",
  normalize_data: "数据标准化",
  interpret_lithology: "岩性解释",
  identify_reservoir: "储层识别",
  plot_well_log_curves: "测井曲线图",
  plot_lithology_distribution: "岩性分布图",
  plot_crossplot: "交会图",
  plot_heatmap: "相关性热力图",
  plot_reservoir_profile: "储层剖面图",
};

function getToolLabel(tool) {
  return TOOL_LABELS[tool] || tool;
}

// 智能体标识 -> 中文显示
function getAgentLabel(agent) {
  if (agent === "data") return "数据智能体";
  if (agent === "expert") return "专家智能体";
  return agent || "智能体";
}

// 从总报告中按标题关键字提取某一节（用于右侧三个卡片）
function extractSection(md, titleKeyword) {
  if (!md) return "";
  const lines = md.split(/\r?\n/);
  let collecting = false;
  const buf = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("#")) {
      if (collecting) break;
      if (trimmed.includes(titleKeyword)) {
        collecting = true;
        buf.push(line);
      }
    } else if (collecting) {
      buf.push(line);
    }
  }
  return buf.join("\n").trim();
}

// ------------ 主应用组件 ------------

function App() {
  const [status, setStatus] = useState("idle");
  const [statusText, setStatusText] = useState("空闲");
  const [messages, setMessages] = useState([
    {
      id: "hello",
      role: "assistant",
      content:
        "您好，我是智能录井解释助手。请上传测井数据并描述您的分析需求，例如“请进行岩性解释和储层识别”。",
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [taskId, setTaskId] = useState("");
  const [filePath, setFilePath] = useState("");
  const [fileName, setFileName] = useState("");
  const [filePreview, setFilePreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [lithologyText, setLithologyText] = useState("");
  const [reservoirText, setReservoirText] = useState("");
  const [overviewText, setOverviewText] = useState("");
  const [charts, setCharts] = useState([]);
  const [reportUrl, setReportUrl] = useState("");
  const [dataFiles, setDataFiles] = useState([]);
  const [toolEvents, setToolEvents] = useState([]);
  const [chartModal, setChartModal] = useState(null);
  const [conversationId] = useState(
    () => "conv_" + Date.now() + "_" + Math.random().toString(36).slice(2)
  );
  const messagesEndRef = useRef(null);
  const workflowLogRef = useRef(null);
  const [scrollTrigger, setScrollTrigger] = useState(0);

  // 仅在 scrollTrigger 变化时滚动（新消息、报告开始、完成），流式 chunk 不触发
  useEffect(() => {
    if (scrollTrigger > 0 && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [scrollTrigger]);

  useEffect(() => {
    if (workflowLogRef.current) {
      workflowLogRef.current.scrollTop = workflowLogRef.current.scrollHeight;
    }
  }, [toolEvents]);

  function statusDotClass() {
    if (status === "processing") return "status-dot processing";
    if (status === "done") return "status-dot done";
    if (status === "error") return "status-dot error";
    return "status-dot idle";
  }

  // 上传文件
  async function handleFile(file) {
    if (!file) return;
    setStatus("processing");
    setStatusText("数据上传中…");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`上传失败：${res.status}`);
      }
      const data = await res.json();
      setTaskId(data.task_id || "");
      setFilePath(data.file_path || "");
      setFileName(file.name);
      if (file.name.toLowerCase().endsWith(".csv")) {
        const text = await file.text();
        const parsed = parseCsv(text);
        setFilePreview(parsed);
      } else {
        setFilePreview(null);
      }
      setStatus("idle");
      setStatusText("文件已上传，等待分析请求");
    } catch (err) {
      console.error(err);
      setStatus("error");
      setStatusText("文件上传失败");
    }
  }

  function onFileInputChange(e) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

    // 发送聊天（流式解析 /chat_stream）
    async function sendMessage() {
        const trimmed = input.trim();
        if (!trimmed || sending) return;
    
        const userMsg = {
          id: `u_${Date.now()}`,
          role: "user",
          content: trimmed,
        };
    
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setSending(true);
        setStatus("processing");
        setStatusText("对话中…");
    
        setToolEvents([]);
    
        const assistantId = `a_${Date.now()}`;
        setMessages((prev) => [
          ...prev,
          {
            id: assistantId,
            role: "assistant",
            content: "正在思考…",
          },
        ]);
        setScrollTrigger((t) => t + 1);

        const formData = new FormData();
        formData.append("user_request", trimmed);
        formData.append("conversation_id", conversationId);
        if (taskId) formData.append("task_id", taskId);
        if (filePath) formData.append("file_path", filePath);
    
        try {
          const res = await fetch(`${API_BASE}/chat_stream`, {
            method: "POST",
            body: formData,
          });
          if (!res.ok || !res.body) {
            throw new Error(`请求失败：${res.status}`);
          }
    
          const reader = res.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = "";
          let localSummary = "";
          let localCharts = [];
          let localReportUrl = "";
    
          let usedTools = false;
    
          // 解析 SSE 流，按事件类型更新 UI
          async function readChunk() {
            const { value, done } = await reader.read();
            if (done) return;

            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";

            for (const part of parts) {
              const line = part.trim();
              if (!line.startsWith("data:")) continue;
              const dataStr = line.slice(5).trim();
              if (dataStr === "[DONE]") continue;
    
              let event;
              try {
                event = JSON.parse(dataStr);
              } catch (e) {
                console.warn("无法解析事件：", dataStr);
                continue;
              }
    
              if (
                event.type === "tool_start" ||
                event.type === "tool_end" ||
                event.type === "tool_error"
              ) {
                usedTools = true;
                setToolEvents((prev) => [
                  ...prev,
                  { ...event, ts: Date.now() },
                ]);
                setStatus("processing");
                const toolLabel = event.tool && TOOL_LABELS[event.tool];
                setStatusText(
                  event.type === "tool_error"
                    ? "数据处理中（部分失败）…"
                    : toolLabel
                      ? `正在执行: ${toolLabel}…`
                      : "数据处理中…"
                );
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId && (m.content || "").startsWith("正在思考")
                      ? { ...m, content: "正在分析并调用工具…" }
                      : m
                  )
                );
              } else if (event.type === "final") {
                // 先清理内部 JSON 代码块
                localSummary = sanitizeAssistantMarkdown(event.summary || "");
                localCharts = (event.charts || []).map((p) =>
                  typeof p === "string" ? toAbsoluteUrl(p) + "?t=" + Date.now() : p
                );
                if (event.report_url)
                  localReportUrl = toAbsoluteUrl(event.report_url);
                const localDataFiles = event.data_files || [];

                // 左侧：显示干净的 Markdown 内容
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: localSummary || "（无回复内容）" }
                      : m
                  )
                );
    
                if (usedTools) {
                  // 右侧：从干净报告中拆章节
                  const overviewSection =
                    extractSection(localSummary, "摘要") || localSummary;
                  const lithSection =
                    extractSection(localSummary, "岩性解释") ||
                    "本轮未生成单独的岩性解释章节，可参考综合报告。";
                  const reservSection =
                    extractSection(localSummary, "储层识别与评价") ||
                    extractSection(localSummary, "储层识别") ||
                    "本轮未生成单独的储层识别章节，可参考综合报告。";

                  setOverviewText(overviewSection);
                  setLithologyText(lithSection);
                  setReservoirText(reservSection);

                  setCharts(localCharts);
                  setReportUrl(localReportUrl);
                  setDataFiles(localDataFiles);

                  // 分析完成后，从服务端拉取数据文件列表（确保包含清洗后等 CSV）
                  const tid = event.task_id || taskId || (localReportUrl && localReportUrl.match(/\/static\/([^/]+)\//)?.[1]);
                  if (tid) {
                    if (!taskId) setTaskId(tid);
                    fetch(`${API_BASE}/api/data_files/${tid}`)
                      .then((r) => r.ok ? r.json() : { files: [] })
                      .then((data) => {
                        if (Array.isArray(data.files) && data.files.length > 0) {
                          setDataFiles(data.files);
                        }
                      })
                      .catch(() => {});
                  }

                  setStatus("done");
                  setStatusText("分析完成");
                } else {
                  setStatus("idle");
                  setStatusText("对话完成");
                }
                setScrollTrigger((t) => t + 1);
              } else if (event.type === "error") {
                const msg = event.message || "分析出错";
                setStatus("error");
                setStatusText(msg);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: `发生错误：${msg}` }
                      : m
                  )
                );
              } else if (event.type === "report_stream_start") {
                // 开始流式输出报告，清空占位符
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId ? { ...m, content: "" } : m
                  )
                );
                setScrollTrigger((t) => t + 1);
              } else if (event.type === "summary_chunk") {
                const chunk = event.content || "";
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: (m.content || "") + chunk }
                      : m
                  )
                );
              } else if (event.type === "delta") {
                const deltaText = event.text || "";
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: (m.content || "") + deltaText }
                      : m
                  )
                );
              } else if (event.summary || event.text) {
                const text = event.summary || event.text;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, content: (m.content || "") + text }
                      : m
                  )
                );
              }
            }
    
            await readChunk();
          }
    
          await readChunk();
        } catch (err) {
          console.error(err);
          setStatus("error");
          setStatusText("请求失败：" + err.message);
          setMessages((prev) =>
            prev.concat({
              id: `sys_${Date.now()}`,
              role: "system",
              content: "分析过程中发生错误，请稍后重试。",
            })
          );
        } finally {
          setSending(false);
        }
      }
    
      function getTid() {
        return taskId || (reportUrl && reportUrl.match(/\/static\/([^/]+)\//)?.[1]);
      }

      async function downloadFile(url, filename) {
        try {
          const res = await fetch(url, { method: "GET", mode: "cors" });
          if (!res.ok) throw new Error(res.statusText);
          const blob = await res.blob();
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = blobUrl;
          a.download = filename;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(blobUrl);
        } catch (e) {
          window.open(url, "_blank");
        }
      }

      async function handleDownloadReport() {
        if (!reportUrl) {
          alert("当前无可用报告，请先完成一次带数据分析的任务。");
          return;
        }
        const tid = getTid();
        if (!tid) {
          alert("无法获取报告信息，请刷新后重试。");
          return;
        }
        await downloadFile(`${API_BASE}/api/download_report/${tid}`, "interpretation_report.docx");
      }

      async function handleDownloadDataFile(filename) {
        const tid = getTid();
        if (!tid) {
          alert("无法获取任务信息，请先上传数据并完成分析。");
          return;
        }
        await downloadFile(
          `${API_BASE}/api/download_file/${tid}/${encodeURIComponent(filename)}`,
          filename
        );
      }
    
      const statusLabelMap = {
        idle: "空闲",
        processing: "数据处理中",
        done: "分析完成",
        error: "错误",
      };
    
      // ------------ JSX ------------
    
      return (
        <div className="app-shell">
          {/* 顶部导航栏 */}
          <header className="app-nav">
            <div className="app-nav-left">
              <div className="app-logo" />
              <div className="app-title-block">
                <div className="app-title">
                  智能地质录井解释系统
                  <span className="app-badge">AI 测井分析</span>
                </div>
                <div className="app-subtitle">
                  面向测井数据的岩性解释与储层识别 · 大模型驱动智能体工作流
                </div>
              </div>
            </div>
            <div className="app-nav-right">
              <div className="status-pill">
                <span className={statusDotClass()} />
                <span>{statusLabelMap[status] || "状态未知"}</span>
              </div>
              {status === "processing" && <div className="spinner" />}
            </div>
          </header>
    
          {/* 主体：上方结果区 / 下方对话区 */}
          <main className="app-main">
            {/* 上方：分析结果区 - 岩性、储层、概览、报告 */}
            <section className="card results-grid" style={{ background: "linear-gradient(160deg, #ecfdf5, #d1fae5)", borderColor: "rgba(134, 239, 172, 0.5)" }}>
              {/* 上方：综合概览 + 报告 */}
              <div className="overview-row">
                <div className="card" style={{ background: "linear-gradient(135deg, #e0f2fe, #bae6fd)", borderColor: "rgba(56, 189, 248, 0.5)" }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-title-dot" />
                      综合解释概览
                    </div>
                    <div className="card-subtitle">
                      汇总岩性、储层与相关参数的整体解释结论
                    </div>
                  </div>
                  <div className="card-body rich-text">
                    {overviewText ? (
                      <div
                        className="rich-text"
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdown(overviewText),
                        }}
                      />
                    ) : (
                      <div className="skeleton" style={{ height: 100 }} />
                    )}
                  </div>
                </div>
    
                <div className="card report-status-card" style={{ background: "linear-gradient(135deg, #ede9fe, #ddd6fe)", borderColor: "rgba(167, 139, 250, 0.5)" }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-title-dot" />
                      报告与任务状态
                    </div>
                  </div>
                  <div className="card-body">
                    <p className="card-subtitle" style={{ marginBottom: 4 }}>
                      当前任务：{statusText}
                    </p>
    
                    <button
                      className="btn-ghost"
                      onClick={handleDownloadReport}
                      disabled={!reportUrl}
                    >
                      <span>📄</span>
                      <span>下载解释报告</span>
                    </button>

                    {dataFiles.length > 0 && (
                      <div className="data-files-downloads">
                        <div className="data-files-title">数据文件下载</div>
                        <div className="data-files-buttons">
                          {dataFiles.map((f) => (
                            <button
                              key={f.filename}
                              className="btn-ghost"
                              onClick={() => handleDownloadDataFile(f.filename)}
                            >
                              <span>📊</span>
                              <span>下载{f.label}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {!reportUrl && dataFiles.length === 0 && toolEvents.length === 0 && (
                      <p className="card-subtitle" style={{ marginTop: 4, fontSize: 11 }}>
                        提示：上传数据并发送分析请求后，此处将实时显示智能体工作流。
                      </p>
                    )}
    
                    <div className="workflow-log">
                      <div className="workflow-log-header">
                        智能体工作流（本轮）
                      </div>
                      <div className="workflow-log-list" ref={workflowLogRef}>
                        {toolEvents.length === 0 ? (
                          <p className="card-subtitle" style={{ padding: 12, fontSize: 11 }}>
                            等待任务执行…
                          </p>
                        ) : (
                          toolEvents.map((e, i) => {
                            const eventClass = e.type === "tool_start" ? "start" : e.type === "tool_error" ? "error" : "end";
                            const agentLabel = getAgentLabel(e.agent);
                            const toolLabel = getToolLabel(e.tool);
                            let statusText = "";
                            let statusIcon = "";
                            if (e.type === "tool_start") {
                              statusText = "执行中";
                              statusIcon = "◐";
                            } else if (e.type === "tool_error") {
                              statusText = e.message || "失败";
                              statusIcon = "✕";
                            } else {
                              statusText = "完成";
                              statusIcon = "✓";
                            }
                            return (
                              <div key={i} className={`workflow-event workflow-event-${eventClass}`}>
                                <div className="workflow-event-line" />
                                <div className="workflow-event-icon">{statusIcon}</div>
                                <div className="workflow-event-body">
                                  <div className="workflow-event-row">
                                    <span className={`workflow-agent-badge ${e.agent || "data"}`}>
                                      {agentLabel}
                                    </span>
                                    <span className="workflow-event-tool">{toolLabel}</span>
                                  </div>
                                  <div className="workflow-event-msg">{statusText}</div>
                                </div>
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 下方：岩性 + 储层 */}
              <div className="results-top">
                <div className="card" style={{ background: "linear-gradient(135deg, #fef9c3, #fef08a)", borderColor: "rgba(253, 224, 71, 0.5)" }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-title-dot" />
                      岩性解释结果
                    </div>
                    <div className="card-subtitle">针对测井曲线的岩性识别与解释</div>
                  </div>
                  <div className="card-body rich-text">
                    {lithologyText ? (
                      <div
                        className="rich-text"
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdown(lithologyText),
                        }}
                      />
                    ) : (
                      <div className="skeleton" style={{ height: 80 }} />
                    )}
                  </div>
                </div>

                <div className="card" style={{ background: "linear-gradient(135deg, #fee2e2, #fecaca)", borderColor: "rgba(248, 113, 113, 0.5)" }}>
                  <div className="card-header">
                    <div className="card-title">
                      <span className="card-title-dot" />
                      储层识别结果
                    </div>
                    <div className="card-subtitle">储层有效性评价与分类结果</div>
                  </div>
                  <div className="card-body rich-text">
                    {reservoirText ? (
                      <div
                        className="rich-text"
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdown(reservoirText),
                        }}
                      />
                    ) : (
                      <div className="skeleton" style={{ height: 80 }} />
                    )}
                  </div>
                </div>
              </div>
            </section>

            {/* 下方：对话交互区 - 浅蓝色底色 */}
            <section className="card chat-panel" style={{ background: "linear-gradient(145deg, #dbeafe, #bfdbfe)", borderColor: "rgba(147, 197, 253, 0.6)" }}>
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-dot" />
                  对话交互
                </div>
                <div className="card-subtitle">与录井解释智能体进行自然语言交互</div>
              </div>
              <div className="card-body chat-messages" style={{ background: "rgba(224, 242, 254, 0.7)", borderColor: "rgba(147, 197, 253, 0.5)" }}>
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`chat-message ${m.role}`}
                    style={
                      m.role === "system"
                        ? { fontStyle: "italic", opacity: 0.76 }
                        : undefined
                    }
                  >
                    {m.role === "assistant" ? (
                      <div
                        className="rich-text"
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdown(
                            sanitizeAssistantMarkdown(m.content)
                          ),
                        }}
                      />
                    ) : (
                      m.content
                    )}
                  </div>
                ))}
                <div ref={messagesEndRef} />
              </div>
              <div className="chat-footer">
                <div className="chat-input-wrapper">
                  <input
                    className="chat-input"
                    placeholder={
                      taskId
                        ? "请输入分析指令，例如：请根据当前测井数据进行岩性解释和储层识别…"
                        : "可先上传测井数据，再输入分析需求…"
                    }
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                      }
                    }}
                    disabled={sending}
                  />
                  <span className="chat-input-send-icon">↩︎</span>
                </div>
                <button
                  className="chat-send-btn"
                  onClick={sendMessage}
                  disabled={sending || !input.trim()}
                >
                  {sending ? (
                    <>
                      <div className="spinner" />
                      处理中…
                    </>
                  ) : (
                    <>
                      发送
                      <span>⮞</span>
                    </>
                  )}
                </button>
              </div>
            </section>
          </main>
    
          {/* 下方：文件上传 + 图表区域 */}
          <section className="app-bottom">
            {/* 上传 + 数据预览 */}
            <section className="card upload-card" style={{ background: "linear-gradient(135deg, #ffedd5, #fed7aa)", borderColor: "rgba(251, 146, 60, 0.5)" }}>
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-dot" />
                  测井数据文件
                </div>
                <div className="card-subtitle">
                  支持 CSV / Excel，建议包含深度、GR、DEN、CALI 等曲线
                </div>
              </div>
              <div className="card-body">
                <div
                  className={"upload-dropzone " + (dragOver ? "drag-over" : "")}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                  }}
                  onDrop={onDrop}
                >
                  <div className="upload-main">
                    <div className="upload-left">
                      <div className="upload-icon">⇪</div>
                      <div>
                        <div className="upload-text-title">
                          {fileName || "拖拽文件到此处，或点击选择"}
                        </div>
                        <div className="upload-text-sub">
                          支持 CSV / XLSX / XLS · 文件将保存在后端工作目录
                        </div>
                      </div>
                    </div>
                    <div className="upload-right">
                      <label className="btn-primary-outline">
                        <span>选择文件</span>
                        <input
                          className="upload-input"
                          type="file"
                          accept=".csv,.xlsx,.xls"
                          onChange={onFileInputChange}
                        />
                      </label>
                      {fileName && (
                        <div className="upload-file-info">当前文件：{fileName}</div>
                      )}
                    </div>
                  </div>
                </div>
    
                {/* 数据预览表格 */}
                {filePreview && filePreview.headers.length > 0 && (
                  <div className="table-container">
                    <table className="table">
                      <thead>
                        <tr>
                          {filePreview.headers.map((h, i) => (
                            <th key={i}>{h || `列${i + 1}`}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filePreview.rows.map((row, ri) => (
                          <tr key={ri}>
                            {row.map((cell, ci) => (
                              <td key={ci}>{cell}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </section>
    
            {/* 图表区域 */}
            <section className="card charts-card" style={{ background: "linear-gradient(135deg, #d1fae5, #a7f3d0)", borderColor: "rgba(52, 211, 153, 0.5)" }}>
              <div className="card-header">
                <div className="card-title">
                  <span className="card-title-dot" />
                  可视化图表
                </div>
                <div className="card-subtitle">
                  曲线图 / 柱状图 / 热力图，自动适配屏幕宽度
                </div>
              </div>
              <div className="card-body charts-grid">
                {charts && charts.length > 0 ? (
                  charts.slice(0, 6).map((c, i) => {
                    if (typeof c === "string") {
                      const url = c.includes("?") ? c.split("?")[0] : c;
                      const isHtml = url.toLowerCase().endsWith(".html");
                      const chartTitle = (url.split("/").pop() || "")
                        .replace(/\?.*$/, "")
                        .replace(/\.(png|html|jpg)$/i, "")
                        .replace(/_/g, " ") || `图表 ${i + 1}`;
                      return (
                        <div
                          className="chart-item chart-clickable"
                          key={`${url}-${i}`}
                          role="button"
                          tabIndex={0}
                          onClick={() => setChartModal({ url: c, isHtml, title: chartTitle })}
                          onKeyDown={(e) => e.key === "Enter" && setChartModal({ url: c, isHtml, title: chartTitle })}
                        >
                          <div className="chart-title">{chartTitle} · 点击放大</div>
                          <div className="chart-canvas-wrapper chart-click-wrapper" style={{ position: "relative", minHeight: 280 }}>
                            {isHtml ? (
                              <iframe
                                src={c}
                                title={`chart-${i}`}
                                style={{
                                  width: "100%",
                                  minHeight: 280,
                                  height: "100%",
                                  border: "none",
                                  borderRadius: 8,
                                  background: "white",
                                  pointerEvents: "none",
                                }}
                              />
                            ) : (
                              <img
                                src={c}
                                alt={chartTitle}
                                style={{
                                  width: "100%",
                                  minHeight: 280,
                                  height: "auto",
                                  objectFit: "contain",
                                  borderRadius: 8,
                                  background: "#f8fafc",
                                  pointerEvents: "none",
                                }}
                                onError={(e) => {
                                  e.target.style.background = "linear-gradient(135deg,#fee2e2,#fecaca)";
                                  e.target.alt = "图表加载失败，请刷新重试";
                                  e.target.onerror = null;
                                }}
                              />
                            )}
                          </div>
                        </div>
                      );
                    }
                    return null;
                  })
                ) : (
                  <>
                    <div className="chart-item">
                      <div className="chart-title">测井曲线示意</div>
                      <div
                        className="chart-canvas-wrapper skeleton"
                        style={{ borderRadius: 8 }}
                      />
                    </div>
                    <div className="chart-item">
                      <div className="chart-title">岩性比例柱状图示意</div>
                      <div
                        className="chart-canvas-wrapper skeleton"
                        style={{ borderRadius: 8 }}
                      />
                    </div>
                    <div className="chart-item">
                      <div className="chart-title">储层属性热力图示意</div>
                      <div
                        className="chart-canvas-wrapper skeleton"
                        style={{ borderRadius: 8 }}
                      />
                    </div>
                  </>
                )}
              </div>
            </section>
          </section>

          {chartModal && (
            <div
              className="chart-modal-overlay"
              onClick={() => setChartModal(null)}
            >
              <div className="chart-modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="chart-modal-header">
                  <span>{chartModal.title}</span>
                  <button className="chart-modal-close" onClick={() => setChartModal(null)}>×</button>
                </div>
                <div className="chart-modal-body">
                  {chartModal.isHtml ? (
                    <iframe src={chartModal.url} title={chartModal.title} />
                  ) : (
                    <img src={chartModal.url} alt={chartModal.title} />
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      );
    }
    
    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
    