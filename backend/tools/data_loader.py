"""
统一的数据加载逻辑，解决 Excel 引擎识别错误。

核心策略：
1. 扩展名为 .csv → 按 CSV 读取
2. 扩展名为 .xlsx/.xls → 先尝试 Excel；失败则尝试按 CSV 解析（兼容扩展名错误或历史遗留文件）

【模块职责】所有工具（预览、清洗、标准化、岩性解释、储层识别、绘图）均通过此模块加载数据。
"""
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _read_csv_safe(path: str) -> pd.DataFrame:
    """安全读取 CSV，尝试多种编码"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(path)


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    加载 CSV 或 Excel 为 DataFrame。
    .xlsx 读取失败时尝试按 CSV 解析（兼容 clean_data 历史输出的扩展名错误文件）。
    """
    if not file_path:
        raise ValueError("file_path 不能为空")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    lower = file_path.lower()

    # 1. 明确为 CSV
    if lower.endswith(".csv"):
        return _read_csv_safe(file_path)

    # 2. Excel：先尝试 Excel 引擎，失败则按 CSV 解析
    if lower.endswith(".xlsx"):
        try:
            return pd.read_excel(file_path, engine="openpyxl")
        except Exception as e:
            logger.warning(f"Excel(.xlsx) 读取失败，尝试按 CSV 解析: {e}")
            return _read_csv_safe(file_path)

    if lower.endswith(".xls"):
        try:
            return pd.read_excel(file_path, engine="xlrd")
        except Exception as e:
            logger.warning(f"Excel(.xls) 读取失败，尝试按 CSV 解析: {e}")
            return _read_csv_safe(file_path)

    # 3. 其他：尝试 CSV
    try:
        return _read_csv_safe(file_path)
    except Exception:
        return pd.read_excel(file_path)


def resolve_column(df: pd.DataFrame, name: str) -> str:
    """解析列名，支持大小写不敏感匹配。"""
    if not name or name in df.columns:
        return name or ""
    lower = name.lower()
    for col in df.columns:
        if col.lower() == lower:
            return col
    return ""
