"""
Excel 转 CSV 工具 - 统一分析流程中的文件格式

支持 .xlsx / .xls，输出 UTF-8 BOM CSV 便于 downstream 工具读取。
"""
import logging
import os

import pandas as pd

from tools.data_loader import sanitize_well_log_dataframe

logger = logging.getLogger(__name__)


def excel_to_csv(file_path: str, suffix: str = "_data") -> str:
    """
    若为 Excel 文件，转换为 CSV 并返回新路径；已是 CSV 则原样返回。

    参数:
        file_path: 输入文件路径
        suffix: 输出文件名后缀，如 _data -> xxx_data.csv，_workflow -> xxx_workflow.csv

    返回:
        CSV 文件路径
    """
    if not file_path or not os.path.exists(file_path):
        return file_path
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return file_path
    if not (lower.endswith(".xlsx") or lower.endswith(".xls")):
        return file_path
    try:
        engine = "openpyxl" if lower.endswith(".xlsx") else "xlrd"
        df = pd.read_excel(file_path, engine=engine)
    except Exception as e:
        logger.warning(f"Excel 读取失败，尝试无 engine 读取: {e}")
        try:
            df = pd.read_excel(file_path)
        except Exception as e2:
            logger.error(f"Excel 转 CSV 失败: {e2}")
            return file_path
    df = sanitize_well_log_dataframe(df)
    dir_name = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    csv_path = os.path.join(dir_name, f"{base}{suffix}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Excel 已转换为 CSV: {csv_path}")
    return csv_path
