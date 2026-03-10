"""
对话历史存储 - 按 conversation_id 保存最近 N 轮问答

【用途】多轮对话上下文，供 LLM 生成回答时参考。监督智能体在无文件/知识问答场景会拉取历史注入 prompt。
"""
from collections import deque
from typing import Dict, List

# 内存存储：conversation_id -> deque of {"user": str, "assistant": str}
_store: Dict[str, deque] = {}
MAX_PAIRS = 10  # 每个会话保留最近 10 对问答


def get_history(conversation_id: str) -> List[dict]:
    """获取最近 10 对问答，展平为 [user, assistant, user, assistant, ...]"""
    if not conversation_id:
        return []
    d = _store.get(conversation_id)
    if not d:
        return []
    out = []
    for pair in d:
        out.append({"role": "user", "content": pair["user"]})
        out.append({"role": "assistant", "content": pair["assistant"]})
    return out


def append_pair(conversation_id: str, user_msg: str, assistant_msg: str) -> None:
    """追加一轮问答，保留最近 MAX_PAIRS 对"""
    if not conversation_id:
        return
    if conversation_id not in _store:
        _store[conversation_id] = deque(maxlen=MAX_PAIRS)
    _store[conversation_id].append({"user": user_msg, "assistant": assistant_msg})