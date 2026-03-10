"""
监督智能体配套工具 - preview 解析、数据校验、文本处理、报告生成

将 supervisor_agent 中的纯函数封装到此模块，便于复用与维护。
"""
import ast
import os
import re
from typing import Dict, List, Optional

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm


# ---------- 路径与数据链 ----------


def extract_data_files(work_dir: str) -> List[Dict[str, str]]:
    """
    从工作目录提取可下载的数据文件（清洗、归一化、岩性、储层等 CSV）。
    返回 [{"filename": "xxx_cleaned.csv", "label": "清洗后数据"}, ...]
    """
    result: List[Dict[str, str]] = []
    if not work_dir or not os.path.isdir(work_dir):
        return result
    labels = (
        ("_cleaned.csv", "清洗后数据"),
        ("_minmax_normalized.csv", "归一化数据"),
        ("_zscore_normalized.csv", "归一化数据"),
        ("_normalized.csv", "归一化数据"),
        ("_lithology.csv", "岩性解释结果"),
        ("_reservoir.csv", "储层识别结果"),
    )
    for f in sorted(os.listdir(work_dir)):
        if not f.endswith(".csv"):
            continue
        for suffix, label in labels:
            if f.endswith(suffix):
                result.append({"filename": f, "label": label})
                break
    return result


def compute_tool_output_path(input_path: str, tool_name: str, params: Optional[Dict] = None) -> str:
    """根据工具和输入路径计算输出路径（用于数据链传递）。clean/normalize 统一输出 CSV。"""
    if not input_path or not os.path.exists(input_path):
        return input_path
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, _ = os.path.splitext(base_name)
    if tool_name == "clean_data":
        return os.path.join(dir_name, f"{name}_cleaned.csv")
    if tool_name == "normalize_data":
        method = (params or {}).get("method", "minmax")
        return os.path.join(dir_name, f"{name}_{method}_normalized.csv")
    if tool_name == "interpret_lithology":
        return os.path.join(dir_name, f"{name}_lithology.csv")
    if tool_name == "identify_reservoir":
        return os.path.join(dir_name, f"{name}_reservoir.csv")
    return input_path


# ---------- Preview 解析 ----------

def extract_columns_from_preview(preview_result: str) -> str:
    """从 preview 数据预览文本结果中提取列名信息，供任务规划使用。"""
    if not preview_result:
        return "（未能获取列信息）"
    lines = preview_result.split("\n")
    cols_raw = ""
    for i, line in enumerate(lines):
        if "【列名】" in line or "列名" in line:
            if "：" in line or ":" in line:
                part = line.split("：")[-1].split(":")[-1].strip()
                if part and len(part) > 2:
                    cols_raw = part
                    break
            if i + 1 < len(lines):
                cols_raw = lines[i + 1].strip()
                break
    if not cols_raw and "列名" in preview_result:
        match = re.search(r"列名[：:]\s*(.+)", preview_result)
        if match:
            cols_raw = match.group(1).strip()
    if not cols_raw:
        return preview_result[:500] + "..." if len(preview_result) > 500 else preview_result
    cols_list = [c.strip() for c in cols_raw.replace("，", ",").split(",") if c.strip()]
    cols_str = ", ".join(cols_list)
    return f"""可用列名（parameters 中必须原样使用以下名称）: {cols_str}
提示：深度列常见为 depth 或 Depth，请根据实际列名填写 depth_column。"""


def get_columns_list_from_preview(preview_result: str) -> List[str]:
    """从 preview 结果中提取列名列表"""
    if not preview_result:
        return []
    lines = preview_result.split("\n")
    for i, line in enumerate(lines):
        if "【列名】" in line or "列名" in line:
            if "：" in line or ":" in line:
                part = line.split("：")[-1].split(":")[-1].strip()
                if part and len(part) > 2:
                    return [c.strip() for c in part.replace("，", ",").split(",") if c.strip()]
            if i + 1 < len(lines):
                raw = lines[i + 1].strip()
                return [c.strip() for c in raw.replace("，", ",").split(",") if c.strip()]
    match = re.search(r"列名[：:]\s*(.+)", preview_result)
    if match:
        raw = match.group(1).strip()
        return [c.strip() for c in raw.replace("，", ",").split(",") if c.strip()]
    return []


def parse_preview_metadata(preview_result: str) -> tuple:
    """
    从 preview 文本中解析列类型与样本数据，用于结构推断。
    返回 (col_dtypes: dict, col_samples: dict)，解析失败返回 ({}, {})。
    """
    col_dtypes, col_samples = {}, {}
    if not preview_result:
        return col_dtypes, col_samples
    dtype_match = re.search(r"【数据类型】\s*\n?\s*(\{[^{}]+\})", preview_result)
    if dtype_match:
        try:
            raw = dtype_match.group(1).strip()
            col_dtypes = ast.literal_eval(raw)
        except Exception:
            pass
    sample_start = preview_result.find("【前")
    if sample_start >= 0:
        tail = preview_result[sample_start:]
        brace_start = tail.find("{")
        if brace_start >= 0:
            raw = tail[brace_start:]
            depth, i, n = 0, 0, len(raw)
            for i in range(n):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        break
            try:
                full = ast.literal_eval(raw[: i + 1])
                for col, idx_map in (full or {}).items():
                    if isinstance(idx_map, dict):
                        vals = [idx_map.get(k) for k in sorted(idx_map.keys())]
                        col_samples[col] = vals
            except Exception:
                pass
    return col_dtypes, col_samples


# ---------- 数据校验 ----------

def _is_numeric_str(s: str) -> bool:
    """判断字符串是否可解析为数值。"""
    try:
        float(s.replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def _is_monotonic_numeric(values: list) -> bool:
    """判断数值序列是否单调（升序或降序），用于识别深度列。"""
    nums = []
    for v in values:
        if v is None or (isinstance(v, float) and (pd.isna(v) or v != v)):
            continue
        try:
            n = float(v) if not isinstance(v, (int, float)) else v
            nums.append(n)
        except (ValueError, TypeError):
            return False
    if len(nums) < 2:
        return True
    diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
    all_pos = all(d >= -1e-9 for d in diffs)
    all_neg = all(d <= 1e-9 for d in diffs)
    return all_pos or all_neg


def validate_well_log_data(preview_result: str, cols_list: list) -> tuple:
    """
    基于数据结构智能校验是否为测井/地质数据。
    返回 (True, None) 表示通过；(False, 提示信息) 表示非测井数据。
    """
    if not cols_list or len(cols_list) < 2:
        return False, "数据列数过少，请确认是否为测井数据文件。"

    col_dtypes, col_samples = parse_preview_metadata(preview_result)

    numeric_cols = set()
    dtypes_map = col_dtypes or {}
    for col in cols_list:
        if not col:
            continue
        dtype_str = str(dtypes_map.get(col, dtypes_map.get(col.strip(), ""))).lower()
        if "float" in dtype_str or "int" in dtype_str:
            numeric_cols.add(col)

    if not numeric_cols and col_samples:
        for col, vals in col_samples.items():
            if vals:
                valid = [v for v in vals[:10] if v is not None and not (isinstance(v, float) and pd.isna(v))]
                if valid and all(isinstance(v, (int, float)) for v in valid):
                    numeric_cols.add(col)

    depth_col = None
    for col in cols_list:
        if not col:
            continue
        cl = col.strip().lower()
        if "depth" in cl or "深度" in cl:
            depth_col = col
            numeric_cols.add(col)
            break
    if not depth_col and col_samples:
        for col in cols_list:
            vals = col_samples.get(col, [])
            if len(vals) >= 2 and _is_monotonic_numeric(vals):
                depth_col = col
                numeric_cols.add(col)
                break

    n_numeric = len(numeric_cols)
    n_total = len(cols_list)
    numeric_ratio = n_numeric / n_total if n_total else 0

    if n_numeric < 2:
        return False, (
            "数据中数值列过少（测井数据通常有多条曲线如 GR、DEN、CNL 等均为数值）。"
            "请确认上传的是包含深度及若干测井曲线的 CSV 或 Excel 文件。"
        )

    if not depth_col:
        return False, (
            "未检测到深度列（测井数据通常有一列随井深单调变化的数值）。"
            "请确认数据中包含深度列（列名可为 depth、Depth、深度 等），或确保至少有一列数值随深度单调递变。"
        )

    if numeric_ratio < 0.4:
        return False, (
            "数据中数值列占比过低（测井数据以数值曲线为主）。"
            "请确认上传的是测井数据文件，而非人员、订单等业务表格。"
        )

    return True, None


# ---------- 文本处理 ----------

def strip_task_json(text: str) -> str:
    """清理模型输出中的内部 JSON 代码块（任务规划等），避免展示到报告/界面。"""
    if not text:
        return text
    text = re.sub(r"```[\s\S]*?```", "", text, flags=re.IGNORECASE)
    stripped = text.lstrip()
    if stripped.startswith("{"):
        depth, end = 0, -1
        for i, c in enumerate(stripped):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end >= 0:
            rest = stripped[end + 1 :].lstrip()
            text = strip_task_json(rest) if rest.startswith("{") else rest
    return text.strip()


# ---------- 报告生成 ----------

def extract_report_only(text: str) -> str:
    """从 LLM 输出中提取仅报告部分，去除对话式前导。保留从 # 开头标题开始的内容。"""
    if not text or not text.strip():
        return text
    text = text.strip()
    lines = text.split("\n")
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## ") or stripped.startswith("### "):
            start_idx = i
            break
    return "\n".join(lines[start_idx:]).strip() if start_idx > 0 else text


def _strip_asterisks_for_docx(text: str) -> str:
    """移除文本中的星号，避免 docx 报告中出现 * 符号。"""
    if not text:
        return text
    return text.replace("*", "").strip()


def build_interpretation_report_docx(
    work_dir: str, report_content: str, file_path: str
) -> Optional[str]:
    """
    将 Markdown 报告内容写入 Word 文档，并嵌入同目录下的 PNG 图表。
    返回报告相对路径（如 /static/{task_id}/interpretation_report.docx），失败返回 None。
    """
    try:
        doc = Document()
        for section in doc.sections:
            section.top_margin = Cm(1.5)
            section.bottom_margin = Cm(1.5)
            section.left_margin = Cm(2)
            section.right_margin = Cm(2)
        normal_style = doc.styles["Normal"]
        normal_style.font.name = "宋体"
        normal_style.font.size = Pt(10.5)
        normal_style.paragraph_format.space_after = Pt(6)
        normal_style.paragraph_format.line_spacing = 1.25
        for level in range(1, 4):
            style_name = f"Heading {level}"
            if style_name in doc.styles:
                h = doc.styles[style_name]
                h.font.name = "黑体"
                h.font.size = Pt(16 - level * 2)
                h.font.bold = True
                h.paragraph_format.space_before = Pt(12)
                h.paragraph_format.space_after = Pt(6)

        clean = _strip_asterisks_for_docx
        for line in report_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("### "):
                doc.add_heading(clean(stripped[4:]), level=3)
            elif stripped.startswith("## "):
                doc.add_heading(clean(stripped[3:]), level=2)
            elif stripped.startswith("# "):
                doc.add_heading(clean(stripped[2:]), level=1)
            elif stripped.startswith("- "):
                p = doc.add_paragraph(clean(stripped[2:]))
                p.style = "List Bullet"
            else:
                doc.add_paragraph(clean(stripped))

        png_charts = [
            f for f in os.listdir(work_dir)
            if f.endswith(".png") and f != os.path.basename(file_path) and not f.startswith(".")
        ]
        if png_charts:
            doc.add_heading("图表", level=1)
            for f in sorted(png_charts):
                try:
                    img_path = os.path.join(work_dir, f)
                    if os.path.isfile(img_path):
                        title = os.path.splitext(f)[0].replace("_", " ").strip()
                        doc.add_paragraph(title, style="Normal")
                        doc.add_picture(img_path, width=Inches(5.5))
                        doc.add_paragraph()
                except Exception:
                    pass

        doc.save(os.path.join(work_dir, "interpretation_report.docx"))
        task_id = os.path.basename(work_dir)
        return f"/static/{task_id}/interpretation_report.docx"
    except Exception:
        return None
