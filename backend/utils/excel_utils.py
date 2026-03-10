"""Excel 转 CSV 工具，统一分析流程中的文件格式"""
import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def excel_to_csv(file_path: str, suffix: str = "_data") -> str:
    """
    若为 Excel 文件，转换为 CSV 并返回 CSV 路径。
    suffix: 输出文件后缀，如 _data 得 xxx_data.csv，_workflow 得 xxx_workflow.csv
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
    dir_name = os.path.dirname(file_path)
    base = os.path.splitext(os.path.basename(file_path))[0]
    csv_path = os.path.join(dir_name, f"{base}{suffix}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Excel 已转换为 CSV: {csv_path}")
    return csv_path
