"""智能体公共工具函数"""
from typing import Dict, Any


def set_file_param(tool_func, params: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    """根据工具的 args_schema 自动注入 file_path 或 data_path，兼容不同命名"""
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
