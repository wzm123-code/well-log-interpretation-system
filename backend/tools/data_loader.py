"""
统一的数据加载逻辑，解决 Excel 引擎识别错误。

核心策略：
1. 扩展名为 .csv → 按 CSV 读取
2. 扩展名为 .xlsx/.xls → 先尝试 Excel；失败则尝试按 CSV 解析（兼容扩展名错误或历史遗留文件）

【模块职责】所有工具（预览、清洗、标准化、岩性解释、储层识别、绘图）均通过此模块加载数据。

【测井 Excel】首行可为重复表头；曲线常为文本型数值；-999.25 等为无效占位。
sanitize_well_log_dataframe 会剔除重复表头、转数值并将占位置为缺失。
"""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 行业常见无效值占位
_WELL_LOG_SENTINEL_VALUES = frozenset(
    {
        -999.0,
        -999.25,
        -9999.0,
        -9999.25,
        -32768.0,
    }
)


def _read_csv_safe(path: str) -> pd.DataFrame:
    """安全读取 CSV，尝试多种编码"""
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(path)


def maybe_drop_mud_log_chinese_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    录井/气测 Excel 常见双表头：首行为英文列名，第二行为中文列名（井深、钻时、甲烷…）。
    若 pandas 将首行作为列名，则第二行会被误读为数据；本函数在首行深度格为中文标签时删除该行。
    """
    if df is None or len(df) < 2:
        return df
    depth_c = None
    for c in df.columns:
        cs = str(c)
        cl = cs.lower()
        if "depth" in cl or "井深" in cs or "深度" in cs or "测深" in cs:
            depth_c = c
            break
    if not depth_c:
        return df
    v0 = df.iloc[0][depth_c]
    v1 = df.iloc[1][depth_c]
    try:
        float(v1)
    except (TypeError, ValueError):
        return df
    if isinstance(v0, str):
        s = v0.strip()
        if any(x in s for x in ("井深", "深度", "测深", "Depth", "MD")) and not s.replace(".", "").isdigit():
            logger.info("检测到录井/气测中文表头行（深度列），已跳过该行")
            return df.iloc[1:].reset_index(drop=True)
    return df


def maybe_drop_duplicate_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    若首行为表头重复行（常见于导出 Excel：Depth 单元格为「深度」、WellName 为「井名」等），则删除该行。
    """
    if df is None or len(df) < 2:
        return df
    row0 = df.iloc[0]
    # 强特征：深度列首格为中文「深度」或英文 depth 标签
    for c in df.columns:
        cs = str(c)
        cl = cs.lower()
        if "depth" in cl or "深度" in cs:
            v = row0[c]
            if isinstance(v, str) and ("深度" in v or v.strip().lower() in ("depth", "md")):
                logger.info("检测到 Excel 重复表头行（深度列），已跳过首行")
                return df.iloc[1:].reset_index(drop=True)
    # 首行大量「列名与单元格相同」（如 Gr 列格为 Gr）
    matches = 0
    for c in df.columns:
        try:
            if str(row0[c]).strip() == str(c).strip():
                matches += 1
        except Exception:
            pass
    if matches >= max(3, len(df.columns) // 2):
        logger.info("检测到疑似重复表头行（列名重复），已跳过首行")
        return df.iloc[1:].reset_index(drop=True)
    # 井名列为「井名」等占位
    for c in df.columns:
        if "well" in str(c).lower() or "井" in str(c):
            v = row0[c]
            if isinstance(v, str) and ("井名" in v or v.strip() == str(c)):
                logger.info("检测到 Excel 重复表头行（井名列），已跳过首行")
                return df.iloc[1:].reset_index(drop=True)
    return df


def _coerce_object_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """将可解析为数值的 object 列转为数值，便于统计与解释算法。"""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            conv = pd.to_numeric(out[col], errors="coerce")
            if conv.notna().sum() >= max(1, int(len(out) * 0.3)):
                out[col] = conv
    return out


def replace_well_log_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """将测井常见无效占位（-999.25 等）替换为 NaN。"""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        s = out[col]
        mask = np.zeros(len(out), dtype=bool)
        for sentinel in _WELL_LOG_SENTINEL_VALUES:
            mask |= np.isclose(s, sentinel, rtol=0, atol=0, equal_nan=False)
        out.loc[mask, col] = np.nan
    return out


def sanitize_well_log_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """测井表通用清洗：去重复表头行、数值化、无效占位转空。"""
    if df is None or df.empty:
        return df
    df = maybe_drop_duplicate_header_row(df)
    df = maybe_drop_mud_log_chinese_header_row(df)
    df = _coerce_object_columns_to_numeric(df)
    df = replace_well_log_sentinels(df)
    return df


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
        return sanitize_well_log_dataframe(_read_csv_safe(file_path))

    # 2. Excel：先尝试 Excel 引擎，失败则按 CSV 解析
    if lower.endswith(".xlsx"):
        try:
            return sanitize_well_log_dataframe(pd.read_excel(file_path, engine="openpyxl"))
        except Exception as e:
            logger.warning(f"Excel(.xlsx) 读取失败，尝试按 CSV 解析: {e}")
            return sanitize_well_log_dataframe(_read_csv_safe(file_path))

    if lower.endswith(".xls"):
        try:
            return sanitize_well_log_dataframe(pd.read_excel(file_path, engine="xlrd"))
        except Exception as e:
            logger.warning(f"Excel(.xls) 读取失败，尝试按 CSV 解析: {e}")
            return sanitize_well_log_dataframe(_read_csv_safe(file_path))

    # 3. 其他：尝试 CSV
    try:
        return sanitize_well_log_dataframe(_read_csv_safe(file_path))
    except Exception:
        return sanitize_well_log_dataframe(pd.read_excel(file_path))


def resolve_column(df: pd.DataFrame, name: str) -> str:
    """解析列名，支持大小写不敏感匹配。"""
    if not name or name in df.columns:
        return name or ""
    lower = name.lower()
    for col in df.columns:
        if col.lower() == lower:
            return col
    return ""
