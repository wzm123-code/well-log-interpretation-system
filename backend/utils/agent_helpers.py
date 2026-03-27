"""
智能体公共工具 - 参数注入

根据工具的 args_schema 自动注入 file_path 或 data_path，
确保 DataAgent/ExpertAgent 调用工具时能正确传递文件路径。
"""
from typing import Any, Dict


def set_file_param(tool_func, params: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """
    根据工具的 args_schema 自动注入 file_path 或 data_path。

    参数:
        tool_func: LangChain Tool 函数（含 args_schema）
        params: 工具参数字典
        file_path: 数据文件路径

    返回:
        注入后的参数字典
    """
    params = dict(params) if params else {}
    try:
        schema = getattr(tool_func, "args_schema", None)
        if schema is None:
            params.setdefault("file_path", file_path)
            return params
        fields = getattr(schema, "model_fields", None) or getattr(schema, "__fields__", None)
        if fields:
            if "file_path" in fields:
                params["file_path"] = file_path
            elif "data_path" in fields:
                params["data_path"] = file_path
            else:
                params.setdefault("file_path", file_path)
        else:
            params.setdefault("file_path", file_path)
    except Exception:
        params.setdefault("file_path", file_path)
    return params
