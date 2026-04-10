"""
录井 / 气测数据分析工具（钻时、全烃、组分气）

与测井曲线解释独立：适用于 GEO_GasSurvey 等钻时+气测表。
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from langchain_core.tools import tool
from plotly.subplots import make_subplots

from tools.data_loader import load_dataframe, resolve_column
from tools.visualization_tools import (
    FONT_FAMILY,
    _downsample_for_plot,
    _result_msg,
    _sort_by_depth,
    _write_plotly_outputs,
)

logger = logging.getLogger(__name__)

MAX_PLOT_POINTS = 1500

_MUD_GAS_CURVE_PRIORITY = ["Rop", "Tg", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "CO2", "Other"]

# 气测解释常用分级参考（行业资料归纳；不同盆地/单位需调整，非试油结论）
# 干燥系数 C1/(C2+C3)：干气 / 湿气-凝析 / 油型重组分（地球化学录井常用区间）
_DRYNESS_DRY_GAS = 100.0
_DRYNESS_WET_LOW = 20.0
_DRYNESS_OIL_HIGH = 20.0
# 气测录井「三、气测录井解释」简版：油层 C1/(C2+C3)&lt;10，气层 &gt;20，湿气/凝析 10–20
_DRYNESS_PIXLER_OIL = 10.0
_DRYNESS_PIXLER_GAS = 20.0


def _col_ci(df: pd.DataFrame, name: str) -> str:
    """大小写不敏感匹配列名，如 C1、c1。"""
    for c in df.columns:
        if str(c).strip().upper() == name.strip().upper():
            return c
    return ""


def _resolve_depth_col(df: pd.DataFrame) -> str:
    for cand in ("Depth", "depth", "DEPTH", "MD", "井深", "测深", "DEPT"):
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c and c in df.columns:
            return c
    return ""


def _resolve_rop_col(df: pd.DataFrame) -> str:
    for cand in ("Rop", "ROP", "rop", "钻时", "机械钻速"):
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c and c in df.columns:
            return c
    return ""


def _resolve_tg_col(df: pd.DataFrame) -> str:
    for cand in ("Tg", "TG", "tg", "全烃", "全量"):
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c and c in df.columns:
            return c
    return ""


def _auto_component_columns(df: pd.DataFrame, depth_col: str, rop_col: str, tg_col: str) -> List[str]:
    out: List[str] = []
    skip = {depth_col, rop_col, tg_col, "", None}
    for c in df.columns:
        if c in skip:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cs = str(c).strip()
        uc = cs.upper()
        if re.match(r"^(C[1-5]|IC4|NC4|IC5|NC5|TG|CO2|OTHER)$", uc):
            out.append(c)
            continue
        if any(x in cs for x in ("甲烷", "乙烷", "丙烷", "丁烷", "戊烷", "二氧化碳", "非烃", "全烃")):
            out.append(c)
    seen = set()
    ordered: List[str] = []
    for p in _MUD_GAS_CURVE_PRIORITY:
        for c in out:
            if c in seen:
                continue
            if str(c).strip().upper().replace(" ", "") == p.upper():
                ordered.append(c)
                seen.add(c)
    for c in out:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def _merge_bool_segments(depth: np.ndarray, flag: np.ndarray) -> List[Tuple[float, float]]:
    """flag 为 True 的连续深度段合并为 (top, bot)。"""
    if len(depth) == 0:
        return []
    idx = np.where(flag)[0]
    if len(idx) == 0:
        return []
    segs: List[Tuple[float, float]] = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((float(depth[start]), float(depth[prev])))
            start = i
            prev = i
    segs.append((float(depth[start]), float(depth[prev])))
    return segs


def _numeric_col(df: pd.DataFrame, col: str) -> np.ndarray:
    if not col or col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").values.astype(float)


def _dryness_c1_over_c2c3(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], str]:
    """干燥系数 C1/(C2+C3)，与行业「干燥系数」定义一致。"""
    c1, c2, c3 = _col_ci(df, "C1"), _col_ci(df, "C2"), _col_ci(df, "C3")
    if not (c1 and c2 and c3):
        return None, "缺少 C1、C2、C3 中某列，无法计算 C1/(C2+C3)"
    v1, v2, v3 = _numeric_col(df, c1), _numeric_col(df, c2), _numeric_col(df, c3)
    denom = v2 + v3
    denom = np.where(np.abs(denom) < 1e-30, np.nan, denom)
    out = v1 / denom
    return out, ""


def _c1_percent_series(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], str]:
    """C1% = C1 / Σ(C1…C5 已识别烃组分)×100，与资料中 C1% 定义一致。"""
    parts = []
    for name in ("C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5"):
        cn = _col_ci(df, name)
        if cn:
            parts.append((name, _numeric_col(df, cn)))
    if not parts:
        return None, "无 C1–C5 组分列"
    stack = np.column_stack([p[1] for p in parts])
    s = np.nansum(np.nan_to_num(stack, nan=0.0), axis=1)
    s = np.where(s < 1e-30, np.nan, s)
    c1v = parts[0][1] if parts[0][0] == "C1" else None
    if c1v is None:
        c1n = _col_ci(df, "C1")
        c1v = _numeric_col(df, c1n) if c1n else np.zeros(len(df))
    pct = c1v / s * 100.0
    return pct, ""


def _picket_ratio_arrays(df: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
    """皮克斯勒图版常用比值：C1/C2、C1/C3、C1/C4（C4 取 nC4 或 iC4）。"""
    c1 = _col_ci(df, "C1")
    out: Dict[str, Optional[np.ndarray]] = {"r12": None, "r13": None, "r14": None}
    if not c1:
        return out
    v1 = _numeric_col(df, c1)
    c2 = _col_ci(df, "C2")
    if c2:
        d = _numeric_col(df, c2)
        out["r12"] = np.where(np.abs(d) < 1e-30, np.nan, v1 / d)
    c3 = _col_ci(df, "C3")
    if c3:
        d = _numeric_col(df, c3)
        out["r13"] = np.where(np.abs(d) < 1e-30, np.nan, v1 / d)
    c4 = _col_ci(df, "nC4") or _col_ci(df, "iC4")
    if c4:
        d = _numeric_col(df, c4)
        out["r14"] = np.where(np.abs(d) < 1e-30, np.nan, v1 / d)
    return out


def _picket_tri_zone(val: float, oil_max: float, gas_max: float) -> str:
    """皮克斯勒区间：&lt;2 或 &gt;gas_max 常归为非工业；2–oil_max 油区；之上至 gas_max 气区。"""
    if not np.isfinite(val) or val <= 0:
        return "无效"
    if val < 2.0 or val > gas_max:
        return "非工业/背景"
    if val <= oil_max:
        return "油区"
    if val <= gas_max:
        return "气区"
    return "非工业/背景"


def _gadkari_lm_series(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Gadkari/气体比率法中的轻中比 LM = 10×C1/(C2+C3)²（资料公式，便于与文献对照）。"""
    c1, c2, c3 = _col_ci(df, "C1"), _col_ci(df, "C2"), _col_ci(df, "C3")
    if not (c1 and c2 and c3):
        return None
    v1, v2, v3 = _numeric_col(df, c1), _numeric_col(df, c2), _numeric_col(df, c3)
    s = v2 + v3
    s = np.where(np.abs(s) < 1e-30, np.nan, s)
    return 10.0 * v1 / (s**2)


def _median_finite(a: Optional[np.ndarray]) -> float:
    if a is None:
        return float("nan")
    x = a[np.isfinite(a)]
    return float(np.median(x)) if len(x) else float("nan")


def _gas_reference_md() -> str:
    return (
        "| 特征 | 油层（参考） | 气层（参考） | 水层（参考） |\n"
        "| --- | --- | --- | --- |\n"
        "| 全烃 TG | 中–高（资料常举 5%–20% 量级，视单位） | 极高（可 &gt;20%） | 低（&lt;2%） |\n"
        "| C1 占比 | 约 60%–80% | 常 &gt;90% | 低（&lt;50%，易混背景气） |\n"
        "| C2+ | 相对较高 | 极低 | 无或极低 |\n"
        "| 干燥系数 C1/(C2+C3) | 常 &lt;20（重组分多） | 干气可很高（资料 &gt;100 为干气型） | — |\n"
        "\n*上表为行业归纳示意，单位与区块需自洽；本工具数值为按列计算结果。*"
    )


@tool
def analyze_mud_gas_survey(
    data_path: str,
    depth_column: str = "",
    rop_column: str = "",
    tg_column: str = "",
    show_percentile: float = 90.0,
) -> str:
    """
    录井气测综合分析：钻时、全烃、烃组分统计，异常显示段（分位数阈值），干湿程度提示。

    适用于钻时 + 全烃(Tg) + 甲烷~戊烷/二氧化碳等列；深度列可为 Depth/井深/MD。
    结果含方法与局限说明（录井气测半定量，需结合测井与试油）。

    参数:
        data_path: CSV/Excel 路径
        depth_column: 深度列名，默认识别 Depth/井深
        rop_column: 钻时列名，默认识别 Rop/钻时
        tg_column: 全烃列名，默认识别 Tg/全烃
        show_percentile: 全烃异常段阈值分位（默认 90，即高于 P90 视为高值段）

    返回:
        Markdown 结构化文字分析；可写出 _mud_gas_zones.csv（高全烃段）、
        _mud_gas_metrics.csv（逐点：干燥系数、C1%、皮克斯勒比值等，列齐全时）
    """
    logger.info(f"录井气测分析: {data_path}")
    try:
        df = load_dataframe(data_path)
        depth_col = resolve_column(df, depth_column) if depth_column else _resolve_depth_col(df)
        rop_col = resolve_column(df, rop_column) if rop_column else _resolve_rop_col(df)
        tg_col = resolve_column(df, tg_column) if tg_column else _resolve_tg_col(df)
        if not depth_col:
            return f"错误: 未找到深度列。可用列: {', '.join(map(str, df.columns.tolist()))}"

        df = df.dropna(subset=[depth_col]).copy()
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
        df = df.dropna(subset=[depth_col])
        df = _sort_by_depth(df, depth_col)

        comp_cols = _auto_component_columns(df, depth_col, rop_col or "", tg_col or "")
        d0, d1 = float(df[depth_col].min()), float(df[depth_col].max())
        n = len(df)

        lines: List[str] = []
        lines.append("=== 录井气测综合分析 ===")
        lines.append(f"输入文件: {data_path}")
        lines.append(f"深度范围: {d0:.2f} – {d1:.2f} m | 有效样点数: {n}")
        lines.append(f"使用深度列: {depth_col}")
        if rop_col:
            lines.append(f"钻时列: {rop_col}")
        else:
            lines.append("钻时列: （未识别，以下仅侧重气测组分）")
        if tg_col:
            lines.append(f"全烃列: {tg_col}")
        else:
            lines.append("全烃列: （未识别，可用组分之和近似全烃趋势）")
        lines.append(f"参与统计的烃类/相关数值列: {', '.join(comp_cols) if comp_cols else '（无）'}")
        lines.append("")
        lines.append("## 一、方法说明")
        lines.append(
            "- 钻时(Rop)降低常反映可钻性变差或地层变化，需结合岩性/压力；单独不能定岩性。\n"
            "- 全烃(Tg)与各组分浓度为录井现场半定量指标，受脱气效率、钻井液、迟到时间校正等影响。\n"
            "- 下文「高值段」按全烃（无 Tg 时用 C1 与重烃之和近似）相对分位数划定，仅作显示**线索**，非试油结论。"
        )
        lines.append("")

        if rop_col and rop_col in df.columns:
            rp = pd.to_numeric(df[rop_col], errors="coerce")
            lines.append("## 二、钻时统计")
            lines.append(
                f"- 范围: {np.nanmin(rp):.3f} – {np.nanmax(rp):.3f} | 均值: {np.nanmean(rp):.3f} | "
                f"中位数: {np.nanmedian(rp):.3f}"
            )
            lines.append("- 解释提示：钻时突降可能对应疏松层或钻参变化；突升可能对应可钻性变差，应结合气测与邻井资料。")
        else:
            lines.append("## 二、钻时统计")
            lines.append("（无钻时列，跳过）")
        lines.append("")

        tg_series = None
        if tg_col and tg_col in df.columns:
            tg_series = pd.to_numeric(df[tg_col], errors="coerce").fillna(0.0).values
        elif comp_cols:
            sub = df[comp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            tg_series = sub.sum(axis=1).values
        else:
            tg_series = np.zeros(n)

        lines.append("## 三、全烃/组分含量统计")
        if not tg_col and comp_cols:
            lines.append("- 未识别全烃列时，以组分之和近似全烃趋势，用于分位与异常段。")
        if tg_col and tg_col in df.columns:
            tgv = pd.to_numeric(df[tg_col], errors="coerce")
            lines.append(
                f"- {tg_col}: min={np.nanmin(tgv):.6f}, max={np.nanmax(tgv):.6f}, "
                f"mean={np.nanmean(tgv):.6f}, P50={np.nanpercentile(tgv.dropna(), 50):.6f}"
            )
        for c in comp_cols[:8]:
            v = pd.to_numeric(df[c], errors="coerce")
            if v.notna().sum() == 0:
                continue
            lines.append(
                f"- {c}: max={np.nanmax(v):.6f}, mean={np.nanmean(v):.6f}"
            )
        lines.append("")

        lines.append("## 四、气测解释综合指标（参考《气测录井解释》等常用分级，本井为计算值）")
        lines.append(
            "- **干燥系数**定义为 **C1/(C2+C3)**。地球化学录井常见分级："
            f"干气型常 **&gt;{_DRYNESS_DRY_GAS:.0f}**；湿气/凝析约 **{_DRYNESS_WET_LOW:.0f}–{_DRYNESS_DRY_GAS:.0f}**；"
            f"油型重组分多时常 **&lt;{_DRYNESS_OIL_HIGH:.0f}**。气测录井简版常取：油显示 **&lt;{_DRYNESS_PIXLER_OIL:.0f}**，气显示 **&gt;{_DRYNESS_PIXLER_GAS:.0f}**，中间为湿气/凝析带（区块差异大）。"
        )
        dry_arr, dry_note = _dryness_c1_over_c2c3(df)
        if dry_arr is not None:
            dv = dry_arr[np.isfinite(dry_arr) & (dry_arr > 0)]
            if len(dv) > 0:
                lines.append(
                    f"- 本井干燥系数 **C1/(C2+C3)**：中位数 **{float(np.median(dv)):.4f}**，"
                    f"P10–P90：{float(np.percentile(dv, 10)):.4f} – {float(np.percentile(dv, 90)):.4f}。"
                )
        else:
            lines.append(f"- （{dry_note}）")

        c1_pct, c1p_note = _c1_percent_series(df)
        if c1_pct is not None:
            pv = c1_pct[np.isfinite(c1_pct)]
            if len(pv) > 0:
                lines.append(
                    f"- **C1%**（甲烷占 C1–C5 组分之和的百分数）：中位数 **{float(np.median(pv)):.2f}%**；"
                    "资料常述气层 **&gt;90%**、油层约 **60%–80%**（均为经验区间）。"
                )
        else:
            lines.append(f"- （{c1p_note}）")

        pr = _picket_ratio_arrays(df)
        lines.append("- **皮克斯勒型比值**（中位数与常用分区对照；同列单位需一致）：")
        if pr.get("r12") is not None:
            m = _median_finite(pr["r12"])
            z = _picket_tri_zone(m, 15.0, 65.0)
            lines.append(
                f"  - **C1/C2** 中位数={m:.4f} → 对照区：**{z}**（油区常 2–15，气区常 15–65，&lt;2 或 &gt;65 多归非工业/背景）。"
            )
        else:
            lines.append("  - C1/C2：（缺 C1 或 C2）")
        if pr.get("r13") is not None:
            m = _median_finite(pr["r13"])
            z = _picket_tri_zone(m, 20.0, 100.0)
            lines.append(
                f"  - **C1/C3** 中位数={m:.4f} → 对照区：**{z}**（油区常 2–20，气区 20–100）。"
            )
        else:
            lines.append("  - C1/C3：（缺列）")
        if pr.get("r14") is not None:
            m = _median_finite(pr["r14"])
            z = _picket_tri_zone(m, 21.0, 200.0)
            lines.append(
                f"  - **C1/C4** 中位数={m:.4f} → 对照区：**{z}**（油区常 2–21，气区 21–200；C4 取 nC4 或 iC4）。"
            )
        else:
            lines.append("  - C1/C4：（缺 C4 列）")

        lm = _gadkari_lm_series(df)
        if lm is not None:
            lv = lm[np.isfinite(lm)]
            if len(lv) > 0:
                lines.append(
                    f"- **Gadkari 轻中比** LM=10×C1/(C2+C3)² 中位数 **{float(np.median(lv)):.6f}**（文献用于气体比率法连续分析，需结合区带图版）。"
                )

        lines.append("")
        lines.append("### 油气水层特征对照（行业归纳，非本井结论）")
        lines.append(_gas_reference_md())
        lines.append("")

        depth_arr = df[depth_col].values.astype(float)
        metrics_path = ""
        try:
            mdict = {"Depth_m": depth_arr}
            if dry_arr is not None:
                mdict["dryness_C1_over_C2C3"] = dry_arr
            if c1_pct is not None:
                mdict["C1_pct"] = c1_pct
            if pr.get("r12") is not None:
                mdict["C1_over_C2"] = pr["r12"]
            if pr.get("r13") is not None:
                mdict["C1_over_C3"] = pr["r13"]
            if pr.get("r14") is not None:
                mdict["C1_over_C4"] = pr["r14"]
            if lm is not None:
                mdict["Gadkari_LM"] = lm
            if len(mdict) > 1:
                dir_name = os.path.dirname(data_path)
                base_name = os.path.basename(data_path)
                name, _ = os.path.splitext(base_name)
                metrics_path = os.path.join(dir_name, f"{name}_mud_gas_metrics.csv")
                pd.DataFrame(mdict).to_csv(metrics_path, index=False, encoding="utf-8-sig")
                lines.append(f"逐点解释指标已写入: {metrics_path}")
                lines.append("")
        except Exception:
            pass

        p = float(show_percentile)
        p = max(50.0, min(99.9, p))
        pos = tg_series[tg_series > 0]
        thr = float(np.percentile(pos, p)) if len(pos) > 0 else 0.0
        flag = tg_series >= thr if thr > 0 else np.zeros(n, dtype=bool)
        segs = _merge_bool_segments(depth_arr, flag) if thr > 0 else []

        lines.append(f"## 五、相对高值段（全烃或组分合计 ≥ P{p:.0f}={thr:.6f}）")
        if not segs:
            lines.append("- 未划分出明显高值段（或全烃近零）；可检查迟到时间、脱气器工况或换分位阈值。")
        else:
            lines.append(f"- 共 {len(segs)} 段（合并连续深度点）:")
            for i, (a, b) in enumerate(segs[:25], 1):
                lines.append(f"  - 段{i}: {a:.2f} – {b:.2f} m")
            if len(segs) > 25:
                lines.append(f"  - … 其余 {len(segs) - 25} 段略")
        lines.append("")

        zones_path = ""
        if segs:
            dir_name = os.path.dirname(data_path)
            base_name = os.path.basename(data_path)
            name, _ = os.path.splitext(base_name)
            zones_path = os.path.join(dir_name, f"{name}_mud_gas_zones.csv")
            zrows = []
            for i, (a, b) in enumerate(segs, 1):
                m = (depth_arr >= a) & (depth_arr <= b)
                zrows.append(
                    {
                        "segment_id": i,
                        "depth_top_m": a,
                        "depth_bottom_m": b,
                        "Tg_or_sum_max": float(np.nanmax(tg_series[m])) if m.any() else "",
                        "points": int(m.sum()),
                    }
                )
            pd.DataFrame(zrows).to_csv(zones_path, index=False, encoding="utf-8-sig")
            lines.append(f"高值段汇总已写入: {zones_path}")
        lines.append("")

        lines.append("## 六、局限与建议")
        lines.append(
            "- 录井气测受钻井液密度、起下钻、接单根、脱气效率影响，深度需做迟到时间校正后方可与测井深度严格对比。\n"
            "- 高全烃段不等于一定具备工业油气流，需结合测井孔隙度、电阻率、压力与试油。\n"
            "- 建议：对异常段做交会图（plot_crossplot）与连井剖面对比；若有测井曲线可叠合分析。"
        )

        return "\n".join(lines)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.exception("录井气测分析失败")
        return f"错误: 录井气测分析异常 - {str(e)}"


@tool
def plot_mud_gas_profile(
    data_path: str,
    depth_column: str = "",
    curves: Optional[List[str]] = None,
) -> str:
    """
    绘制录井气测剖面：深度为纵轴，多道展示钻时、全烃与主要烃组分（Plotly HTML + 可选 PNG）。

    参数:
        data_path: 数据路径
        depth_column: 深度列，默认识别
        curves: 要绘制的列名列表；默认自动选 Rop、Tg、C1、C2 等（最多 8 道）
    """
    logger.info(f"录井气测剖面图: {data_path}")
    try:
        df = load_dataframe(data_path)
        depth_col = resolve_column(df, depth_column) if depth_column else _resolve_depth_col(df)
        if not depth_col:
            return f"错误: 未找到深度列。可用列: {', '.join(map(str, df.columns.tolist()))}"

        df = _sort_by_depth(df, depth_col)
        df = _downsample_for_plot(df, MAX_PLOT_POINTS)

        if curves:
            plot_cols = [resolve_column(df, c) or (c if c in df.columns else "") for c in curves]
            plot_cols = [c for c in plot_cols if c][:8]
        else:
            rop = _resolve_rop_col(df)
            tg = _resolve_tg_col(df)
            comp = _auto_component_columns(df, depth_col, rop or "", tg or "")
            plot_cols = []
            for c in [rop, tg]:
                if c and c not in plot_cols:
                    plot_cols.append(c)
            for c in comp:
                if c not in plot_cols and len(plot_cols) < 8:
                    plot_cols.append(c)

        plot_cols = [c for c in plot_cols if c in df.columns]
        if len(plot_cols) < 1:
            return f"错误: 无可用数值道。可用列: {', '.join(map(str, df.columns.tolist()))}"

        n_curves = len(plot_cols)
        depth_min, depth_max = float(df[depth_col].min()), float(df[depth_col].max())
        depth_info = f"深度 {depth_min:.1f}–{depth_max:.1f} m | N={len(df)}"

        fig = make_subplots(
            rows=1,
            cols=n_curves,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            subplot_titles=plot_cols,
        )

        for i, curve_col in enumerate(plot_cols, start=1):
            vals = df[curve_col].replace([np.inf, -np.inf], np.nan)
            yv = df[depth_col].values
            xv = pd.to_numeric(vals, errors="coerce").values.astype(float)
            fig.add_trace(
                go.Scatter(
                    x=xv,
                    y=yv,
                    mode="lines",
                    name=curve_col,
                    line=dict(width=2.0),
                    hovertemplate=f"{curve_col}: %{{x:.6f}}<br>深度: %{{y:.2f}} m<extra></extra>",
                ),
                row=1,
                col=i,
            )
            fig.update_xaxes(title_text=curve_col, row=1, col=i, showgrid=True, gridcolor="#e0e0e0")
            fig.update_yaxes(autorange="reversed", row=1, col=i, showgrid=True, gridcolor="#e0e0e0")

        fig.update_yaxes(title_text="深度 (m)", row=1, col=1)
        fig.update_layout(
            title=dict(
                text=f"录井气测剖面<br><sup>{depth_info}</sup>",
                font=dict(size=16, family=FONT_FAMILY),
            ),
            height=900,
            width=min(380 * n_curves, 2400),
            font=dict(family=FONT_FAMILY, size=12),
            template="plotly_white",
            hovermode="closest",
            dragmode="zoom",
        )

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_mud_gas_profile.html")

        def _mpl_png(p: str) -> None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_m, axes = plt.subplots(1, n_curves, figsize=(2.8 * n_curves, 10), sharey=True)
            if n_curves == 1:
                axes = [axes]
            for ax, col in zip(axes, plot_cols):
                xv = pd.to_numeric(df[col], errors="coerce").values
                yv = df[depth_col].values
                ax.plot(xv, yv, lw=0.8)
                ax.set_xlabel(col, fontsize=8)
                ax.invert_yaxis()
            axes[0].set_ylabel("深度 (m)", fontsize=9)
            plt.tight_layout()
            fig_m.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig_m)

        _, png_ok, png_err = _write_plotly_outputs(fig, output_path, _mpl_png)
        return _result_msg("录井气测剖面图", output_path, png_ok, png_err)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.exception("录井剖面图失败")
        return f"错误: 录井剖面图异常 - {str(e)}"
