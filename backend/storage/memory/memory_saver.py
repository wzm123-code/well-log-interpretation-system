# storage/memory/memory_saver.py
"""
内存检查点 - LangGraph 对话状态持久化

【用途】各智能体（监督、数据、专家）的 create_react_agent 使用此 checkpointer，
实现按 thread_id 存储多轮对话的消息状态（用于 ReAct 工具调用的上下文）。
"""
from langgraph.checkpoint.memory import MemorySaver

_memory_saver = MemorySaver

def get_memory_saver():
    """
    获取内存检查点存储器实例
    """
    return _memory_saver()