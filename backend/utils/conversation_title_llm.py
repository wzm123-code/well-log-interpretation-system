"""
根据对话文本生成简短会话标题（供「新建对话」归档上一会话时使用）。
"""
import os
import re
import logging

from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from utils.agent_builder import load_config

logger = logging.getLogger(__name__)

SUPERVISOR_AGENT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "config", "supervisor_agent_deepseek_config.json"
)


def _fallback_title_from_text(dialogue: str) -> str:
    """无 API 或失败时，从首行「用户：」内容截取。"""
    for line in dialogue.splitlines():
        line = line.strip()
        if line.startswith("用户："):
            c = line.replace("用户：", "", 1).strip()
            if c:
                return (c[:48] + "…") if len(c) > 48 else c
    return "未命名会话"


def generate_session_title_llm(dialogue: str) -> str:
    """
    调用 LLM 生成不超过约 24 字的中文标题；失败时回退为截断首条用户话。
    dialogue: 已截断的对话文本。
    """
    dialogue = (dialogue or "").strip()
    if not dialogue:
        return "未命名会话"
    try:
        config = load_config(SUPERVISOR_AGENT_CONFIG)
        mc = config.get("config", {})
        llm = ChatDeepSeek(
            model=mc.get("model", "deepseek-chat"),
            temperature=0.2,
            max_tokens=128,
            timeout=60,
            top_p=0.9,
        )
        prompt = (
            "你是标题助手。根据下方对话摘录，生成一个不超过 24 个汉字的中文标题，概括对话主题。"
            "只输出标题本身：不要引号、不要书名号、不要「标题：」等前缀、不要换行、不要多余说明。\n\n"
            f"【对话摘录】\n{dialogue[:12000]}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = (getattr(resp, "content", None) or str(resp) or "").strip()
        text = text.splitlines()[0].strip() if text else ""
        text = re.sub(r"^(标题|题目)[：:\s]+", "", text)
        text = text.strip("「」『』\"' ")
        if not text:
            return _fallback_title_from_text(dialogue)
        if len(text) > 40:
            text = text[:24] + "…"
        return text
    except Exception as e:
        logger.warning("生成会话标题失败，使用回退: %s", e)
        return _fallback_title_from_text(dialogue)
