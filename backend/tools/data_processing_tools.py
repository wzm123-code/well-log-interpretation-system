"""
数据预处理工具 - 预览、清洗、标准化

【工具】preview_data（获取列名与统计供任务规划）、clean_data（缺失值/异常值）、normalize_data（归一化）。
输出均为 CSV，供下游 interpret/identify/plot 使用。
"""
import os
import pandas as pd
import numpy as np
from langchain_core.tools import tool
import logging

from tools.data_loader import load_dataframe, resolve_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_depth_column_name(df: pd.DataFrame) -> str:
    for cand in ("Depth", "depth", "DEPTH", "MD", "TVD", "DEPT", "井深", "测深"):
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c and c in df.columns:
            return c
    return ""


def _resolve_gr_column_name(df: pd.DataFrame) -> str:
    for cand in ("GR", "Gr", "gr", "伽马", "自然伽马"):
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c and c in df.columns:
            return c
    return ""


def _drop_missing_rows_well_log(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode:
      - key_curves: 仅删除「深度或 GR 缺失」的行（测井常规：各曲线可缺，不应因某一列空删整行）
      - any_column: 任一行任一列有 NaN 即删（旧行为，易把数据删光）
    """
    if mode == "any_column":
        return df.dropna()
    depth_c = _resolve_depth_column_name(df)
    gr_c = _resolve_gr_column_name(df)
    subset = [c for c in (depth_c, gr_c) if c]
    if subset:
        out = df.dropna(subset=subset, how="any")
        logger.info(
            "clean_data: 按关键列删行 subset=%s, 行数 %d -> %d",
            subset,
            len(df),
            len(out),
        )
        return out
    # 无深度/GR 列名时：退化为「至少半数列非空」保留行，避免删光
    if len(df.columns) == 0:
        return df
    thresh = max(1, len(df.columns) // 4)
    return df.dropna(thresh=thresh)


@tool
def preview_data(file_path: str, n_rows: int = 5) -> str:
    """
    预览数据文件的基本信息

    参数:
        file_path: 数据文件路径（支持CSV、Excel格式）
        n_rows: 预览的行数，默认5行

    返回:
        数据的基本信息摘要
    """
    logger.info(f"开始预览数据: {file_path}")
    try:
        df = load_dataframe(file_path)

        info = f"""
        === 数据预览 ===
        文件路径: {file_path}

        【数据形状】
        行数: {df.shape[0]}
        列数: {df.shape[1]}

        【列名】
        {', '.join(df.columns.tolist())}

        【数据类型】
        {df.dtypes.astype(str).to_dict()}

        【缺失值统计】
        {df.isnull().sum().to_dict()}

        【前{n_rows}行数据】
        {df.head(n_rows).to_dict()}

        【基本统计信息】
        {df.describe().to_dict()}
        """
        return info

    except FileNotFoundError:
        return f"错误: 文件未找到 - {file_path}"
    except Exception as e:
        logger.error(f"预览数据失败: {str(e)}")
        return f"错误: 预览数据时发生异常 - {str(e)}"


@tool
def clean_data(
    file_path: str,
    handle_missing: str = "drop",
    remove_outliers: bool = False,
    numeric_columns: str = None,
    drop_scope: str = "key_curves",
) -> str:
    """
    数据清洗：处理缺失值、异常值等

    参数:
        file_path: 输入文件路径（支持CSV、Excel格式）
        handle_missing: 缺失值处理方式
            - "drop": 删除行（默认与 drop_scope 配合，见下）
            - "fill": 前向填充缺失值
        remove_outliers: 是否移除异常值（使用IQR方法）
        numeric_columns: 需要处理异常值的数值列列表（逗号分隔的字符串）
        drop_scope: 当 handle_missing="drop" 时生效
            - "key_curves"（默认）：仅删除「深度或自然伽马」缺失的行，其余列为空保留（测井数据常见）
            - "any_column": 任一行任一列有 NaN 即删（旧行为，易导致清洗后行数为 0）

    返回:
        清洗结果摘要
    """
    logger.info(f"开始清洗数据: {file_path}")
    try:
        df = load_dataframe(file_path)

        if not isinstance(df, pd.DataFrame):
            return f"错误: 读取文件失败，无法转换为DataFrame"

        original_rows = len(df)
        original_missing = df.isnull().sum().sum()

        # 处理缺失值（测井数据勿用全表 dropna，否则各曲线分段缺失会导致整表被删空）
        if handle_missing == "drop":
            scope = (drop_scope or "key_curves").strip().lower()
            if scope == "any_column":
                df = _drop_missing_rows_well_log(df, "any_column")
            else:
                df = _drop_missing_rows_well_log(df, "key_curves")
        elif handle_missing == "fill":
            df = df.ffill()

        # 处理异常值
        if remove_outliers and numeric_columns:
            numeric_cols_list = [col.strip() for col in numeric_columns.split(',')]
            for col in numeric_cols_list:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[~((df[col] < lower_bound) | (df[col] > upper_bound))]

        # 生成输出路径（统一保存为 CSV，避免扩展名与格式不一致导致下游读取失败）
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_cleaned.csv"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)

        cleaned_rows = len(df)
        if cleaned_rows == 0:
            return (
                "错误: 清洗后有效行数为 0，未写入文件。\n"
                "原因多为：handle_missing=\"drop\" 且深度/GR 全部缺失，或曾使用「任意列有缺失即删」把数据删光。\n"
                "建议：使用 drop_scope=\"key_curves\"（默认，仅删深度或 GR 缺失行），或 handle_missing=\"fill\"，"
                "或跳过清洗直接进行岩性解释（interpret_lithology 会自行处理无效占位）。\n"
                f"输入行数: {original_rows}"
            )

        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        cleaned_missing = df.isnull().sum().sum()

        result = f"""
        === 数据清洗完成 ===
        输入文件: {file_path}
        输出文件: {output_path}
        缺失处理: handle_missing={handle_missing}, drop_scope={drop_scope if handle_missing == 'drop' else '—'}

        【清洗前】
        行数: {original_rows}
        缺失值总数: {original_missing}

        【清洗后】
        行数: {cleaned_rows}
        缺失值总数: {cleaned_missing}

        【变更】
        删除行数: {original_rows - cleaned_rows}
        处理缺失值: {original_missing - cleaned_missing}

        数据已成功清洗并保存！
        """
        return result

    except FileNotFoundError:
        return f"错误: 文件未找到 - {file_path}"
    except Exception as e:
        logger.error(f"清洗数据失败: {str(e)}")
        return f"错误: 清洗数据时发生异常 - {str(e)}"


@tool
def normalize_data(file_path: str, method: str = 'minmax') -> str:
    """
    数据标准化

    参数:
        file_path: 输入文件路径（支持CSV、Excel格式）
        method: 标准化方法
            - "minmax": Min-Max标准化（默认）
            - "standard": Z-score标准化
            - "robust": Robust标准化（使用中位数和IQR）

    返回:
        标准化结果摘要
    """
    logger.info(f"开始标准化数据: {file_path}")
    try:
        df = load_dataframe(file_path)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        original_stats = df[numeric_cols].describe().to_dict()

        if method == 'minmax':
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        elif method == 'standard':
            df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        elif method == 'robust':
            for col in numeric_cols:
                median = df[col].median()
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    df[col] = (df[col] - median) / IQR

         # 生成输出路径（统一保存为 CSV，避免扩展名与格式不一致导致下游读取失败）
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_{method}_normalized.csv"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        normalized_stats = df[numeric_cols].describe().to_dict()

        result = f"""
        === 数据标准化完成 ===
        输入文件: {file_path}
        输出文件: {output_path}
        标准化方法: {method}
        处理列数: {len(numeric_cols)}

        【标准化前统计】
        {original_stats}

        【标准化后统计】
        {normalized_stats}

        数据已成功标准化并保存！
        """
        return result

    except FileNotFoundError:
        return f"错误: 文件未找到 - {file_path}"
    except Exception as e:
        logger.error(f"标准化数据失败: {str(e)}")
        return f"错误: 标准化数据时发生异常 - {str(e)}"