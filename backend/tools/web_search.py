"""
联网搜索工具 - 百度千帆智能搜索生成 API

用于普通对话时，当用户问题需要实时信息（天气、新闻、政策等）时调用。
API Key 从环境变量 QIANFAN_AI_SEARCH_API_KEY 或 QIANFAN_API_KEY 读取。
"""
import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

API_URL = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"


def get_api_key() -> str:
    """从环境变量读取 API Key"""
    return os.environ.get("QIANFAN_AI_SEARCH_API_KEY") or os.environ.get("QIANFAN_API_KEY") or ""


def search_sync(query: str, top_k: int = 5, site_filter: list = None, recency: str = None) -> dict:
    """
    同步调用智能搜索生成 API（供 asyncio.run_in_executor 使用）。

    :param query: 搜索问题（72字符以内）
    :param top_k: 返回前几条网页结果，默认 5
    :param site_filter: 可选，限制搜索站点
    :param recency: 可选，时间过滤 "day" | "week" | "month"
    :return: {"answer": str, "references": list} 或 {"error": str}
    """
    api_key = get_api_key()
    if not api_key:
        return {"error": "未配置 QIANFAN_AI_SEARCH_API_KEY，联网搜索功能不可用"}

    if len(query) > 72:
        query = query[:72]

    payload = {
        "messages": [{"role": "user", "content": query}],
        "search_source": "baidu_search_v2",
        "resource_type_filter": [{"type": "web", "top_k": top_k}],
    }
    if site_filter:
        payload["search_filter"] = {"match": {"site": site_filter}}
    if recency:
        payload["search_recency_filter"] = recency

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        result = response.json()

        if response.status_code != 200:
            return {"error": f"API 错误 {response.status_code}: {result.get('error_msg', response.text)}"}

        answer = ""
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            answer = msg.get("content", "")
        if not answer:
            answer = result.get("result", "") or result.get("answer", "")

        references = result.get("references", [])

        return {"answer": answer, "references": references}

    except requests.RequestException as e:
        logger.warning(f"联网搜索请求失败: {e}")
        return {"error": f"请求失败: {e}"}
    except json.JSONDecodeError as e:
        logger.warning(f"联网搜索响应解析失败: {e}")
        return {"error": f"响应解析失败: {e}"}


def format_search_result_for_reply(data: dict) -> str:
    """
    将搜索结果格式化为可展示给用户的 Markdown 文本。
    """
    if "error" in data:
        return f"联网搜索暂时不可用：{data['error']}\n\n您可以尝试直接描述您的问题，我将基于已有知识为您解答。"

    answer = data.get("answer", "").strip()
    references = data.get("references", [])

    parts = []
    if answer:
        parts.append(answer)

    if references:
        parts.append("\n\n**参考来源：**")
        for i, ref in enumerate(references[:5], 1):
            title = ref.get("title", "无标题")
            url = ref.get("url", "")
            content = ref.get("content", "") or ref.get("snippet", "")
            preview = (content[:120] + "…") if len(content) > 120 else content
            if url:
                parts.append(f"\n{i}. [{title}]({url})")
            else:
                parts.append(f"\n{i}. {title}")
            if preview:
                parts.append(f"\n   {preview}")

    return "\n".join(parts) if parts else "未找到相关结果，请换个方式提问。"


def format_search_raw_for_llm(data: dict, max_ref_chars: int = 600, max_total_chars: int = 12000) -> str:
    """
    将搜索结果整理为供「二次 LLM 总结」使用的纯文本上下文（不直接展示给用户）。
    若 data 含 error 或无可归纳内容，返回空字符串。
    """
    if "error" in data:
        return ""

    answer = (data.get("answer") or "").strip()
    references = data.get("references") or []

    parts: list[str] = []
    if answer:
        parts.append("【搜索 API 综合摘要】\n" + answer)

    if references:
        parts.append("\n【网页条目】")
        for i, ref in enumerate(references[:8], 1):
            title = ref.get("title", "无标题")
            url = ref.get("url", "")
            content = ref.get("content", "") or ref.get("snippet", "") or ""
            if len(content) > max_ref_chars:
                content = content[:max_ref_chars] + "…"
            line = f"\n{i}. {title}"
            if url:
                line += f"\n   链接: {url}"
            if content:
                line += f"\n   摘要: {content}"
            parts.append(line)

    text = "\n".join(parts).strip()
    if len(text) > max_total_chars:
        text = text[:max_total_chars] + "\n…（内容已截断）"
    return text
