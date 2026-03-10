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

from tools.data_loader import load_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
def clean_data(file_path: str, handle_missing: str = "drop", remove_outliers: bool = False, numeric_columns: str = None) -> str:
    """
    数据清洗：处理缺失值、异常值等

    参数:
        file_path: 输入文件路径（支持CSV、Excel格式）
        handle_missing: 缺失值处理方式
            - "drop": 删除包含缺失值的行（默认）
            - "fill": 前向填充缺失值
        remove_outliers: 是否移除异常值（使用IQR方法）
        numeric_columns: 需要处理异常值的数值列列表（逗号分隔的字符串）

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

        # 处理缺失值
        if handle_missing == 'drop':
            df = df.dropna()
        elif handle_missing == 'fill':
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
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        cleaned_rows = len(df)
        cleaned_missing = df.isnull().sum().sum()

        result = f"""
        === 数据清洗完成 ===
        输入文件: {file_path}
        输出文件: {output_path}

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