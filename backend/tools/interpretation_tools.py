"""
地质解释工具 - 岩性解释与储层识别

【工具】interpret_lithology：综合 GR、密度、中子、自然电位、声波时差、深电阻率等多曲线，
      按测井机理与阈值分层判别岩性（向量化 numpy，适配大数据）。
      identify_reservoir：基于孔隙度/渗透率分级储层。
输出 CSV 供 plot_lithology_distribution / plot_reservoir_profile 使用。
"""
import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from tools.data_loader import load_dataframe, resolve_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 报告用：深度分箱默认厚度（米），越大段数越少，越接近「地质单元」尺度
DEFAULT_SEGMENT_BIN_M = 15.0

# 解释方法参考表（写入工具输出，供 LLM 与用户对照）
METHODOLOGY_TABLE_MD = """
| 岩性 | GR (API) | SP | 电阻率(深) | 中子 | 密度 | 声波 |
|------|----------|-----|-----------|------|------|------|
| 泥岩 | 偏高(>70) | 近基线 | 常偏低 | 常偏高 | 2.2~2.6 | 较高 |
| 砂岩 | 中低(40~70) | 负异常常见 | 中—高 | 中等 | 2.0~2.4 | 中等 |
| 粉砂岩 | 中等(50~70) | 弱负异常 | 变化大 | 中等 | 2.1~2.5 | 中等 |
| 灰岩 | 常低(<40) | 不定 | 可高 | 常低 | >2.7 | 较低 |
| 含油气显示砂岩 | 中低 | 负异常 | 明显偏高 | 中—高 | 可偏低 | 中高 |
（表为经验区间；本工具逐点分类规则见输出说明，二者需对照理解）
"""


def _numeric_series(df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if not col or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    return s.values.astype(float)


def _first_resolved_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for cand in candidates:
        if not cand:
            continue
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c and c in df.columns:
            return c
    return ""


def _auto_resistivity_column(df: pd.DataFrame, prefer: str) -> str:
    if prefer:
        c = resolve_column(df, prefer) or (prefer if prefer in df.columns else "")
        if c:
            return c
    for cand in ("R90", "R60", "R30", "R20", "R10", "RT", "Rt", "ILD", "LLD", "Rx", "rx"):
        c = resolve_column(df, cand) or (cand if cand in df.columns else "")
        if c:
            return c
    return ""


def _resolve_depth_column(df: pd.DataFrame) -> str:
    return _first_resolved_column(
        df, ["Depth", "depth", "DEPTH", "MD", "TVD", "DEPT", "井深", "测深"]
    )


def _estimate_sample_interval(depth_arr: np.ndarray) -> float:
    """由深度列估计采样间隔（米）：取相邻深度差的中位数。"""
    d = np.asarray(depth_arr, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return 0.1
    dif = np.diff(np.sort(d))
    dif = dif[dif > 1e-6]
    if len(dif) == 0:
        return 0.1
    step = float(np.nanmedian(dif))
    if step <= 0 or not np.isfinite(step):
        return 0.1
    return min(max(step, 0.01), 50.0)


def _mode_ignore_nan(values: np.ndarray) -> str:
    s = pd.Series(values).dropna()
    if s.empty:
        return ""
    m = s.mode()
    return str(m.iloc[0]) if len(m) else str(s.iloc[0])


def build_lithology_segments_binned(
    df: pd.DataFrame,
    depth_col: str,
    lith_col: str = "Lithology",
    bin_m: float = DEFAULT_SEGMENT_BIN_M,
    gr_col: str = "",
    rt_col: str = "",
    neu_col: str = "",
    den_col: str = "",
    dt_col: str = "",
) -> pd.DataFrame:
    """
    按深度等间距分箱，每箱内取岩性众数及曲线均值/范围，用于「深度段划分」表（接近专业报告中的单元划分）。
    """
    if not depth_col or depth_col not in df.columns:
        return pd.DataFrame()

    depth = pd.to_numeric(df[depth_col], errors="coerce").values
    lith = df[lith_col].values
    valid = np.isfinite(depth) & pd.notna(lith) & (lith != "无效数据")
    if not valid.any():
        return pd.DataFrame()

    dmin = float(np.nanmin(depth[valid]))
    dmax = float(np.nanmax(depth[valid]))
    if dmax <= dmin:
        return pd.DataFrame()

    edges = np.arange(np.floor(dmin / bin_m) * bin_m, np.ceil(dmax / bin_m) * bin_m + bin_m * 1.001, bin_m)
    bin_id = np.digitize(depth, edges) - 1
    bin_id = np.clip(bin_id, 0, len(edges) - 2)

    rows = []
    gr_s = pd.to_numeric(df[gr_col], errors="coerce").values if gr_col and gr_col in df.columns else None
    rt_s = pd.to_numeric(df[rt_col], errors="coerce").values if rt_col and rt_col in df.columns else None
    neu_s = pd.to_numeric(df[neu_col], errors="coerce").values if neu_col and neu_col in df.columns else None
    den_s = pd.to_numeric(df[den_col], errors="coerce").values if den_col and den_col in df.columns else None
    dt_s = pd.to_numeric(df[dt_col], errors="coerce").values if dt_col and dt_col in df.columns else None

    for b in range(len(edges) - 1):
        m = valid & (bin_id == b)
        if not m.any():
            continue
        top = float(edges[b])
        bottom = float(edges[b + 1])
        lab = _mode_ignore_nan(lith[m])
        if not lab:
            continue
        thick = bottom - top
        row = {
            "depth_top_m": round(top, 2),
            "depth_bottom_m": round(bottom, 2),
            "thickness_m": round(thick, 2),
            "lithology_mode": lab,
            "n_samples": int(np.sum(m)),
        }
        if gr_s is not None:
            g = gr_s[m]
            if np.isfinite(g).any():
                row["gr_min"] = round(float(np.nanmin(g)), 2)
                row["gr_max"] = round(float(np.nanmax(g)), 2)
                row["gr_mean"] = round(float(np.nanmean(g)), 2)
        if rt_s is not None:
            r = rt_s[m]
            if np.isfinite(r).any():
                row["rt_mean"] = round(float(np.nanmean(r)), 2)
                row["rt_max"] = round(float(np.nanmax(r)), 2)
        if neu_s is not None:
            n = neu_s[m]
            if np.isfinite(n).any():
                mx = np.nanmax(n)
                scale = 100.0 if mx <= 1.5 else 1.0
                row["cn_mean_pct"] = round(float(np.nanmean(n)) * scale, 2)
        if den_s is not None:
            d = den_s[m]
            if np.isfinite(d).any():
                row["den_mean"] = round(float(np.nanmean(d)), 3)
        if dt_s is not None:
            t = dt_s[m]
            if np.isfinite(t).any():
                row["dt_mean"] = round(float(np.nanmean(t)), 2)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    seg = pd.DataFrame(rows)
    # 合并相邻且众数岩性相同的行（减少表碎片）
    merged = []
    i = 0
    while i < len(seg):
        j = i + 1
        while j < len(seg) and seg.iloc[j]["lithology_mode"] == seg.iloc[i]["lithology_mode"]:
            j += 1
        block = seg.iloc[i:j]
        row = {
            "depth_top_m": block["depth_top_m"].iloc[0],
            "depth_bottom_m": block["depth_bottom_m"].iloc[-1],
            "thickness_m": round(float(block["depth_bottom_m"].iloc[-1] - block["depth_top_m"].iloc[0]), 2),
            "lithology_mode": block["lithology_mode"].iloc[0],
            "n_samples": int(block["n_samples"].sum()),
        }
        if "gr_min" in block.columns:
            row["gr_min"] = round(float(block["gr_min"].min()), 2)
            row["gr_max"] = round(float(block["gr_max"].max()), 2)
            row["gr_mean"] = round(float(block["gr_mean"].mean()), 2)
        if "rt_mean" in block.columns:
            row["rt_mean"] = round(float(block["rt_mean"].mean()), 2)
        if "rt_max" in block.columns:
            row["rt_max"] = round(float(block["rt_max"].max()), 2)
        if "cn_mean_pct" in block.columns:
            row["cn_mean_pct"] = round(float(block["cn_mean_pct"].mean()), 2)
        if "den_mean" in block.columns:
            row["den_mean"] = round(float(block["den_mean"].mean()), 3)
        if "dt_mean" in block.columns:
            row["dt_mean"] = round(float(block["dt_mean"].mean()), 2)
        merged.append(row)
        i = j
    return pd.DataFrame(merged)


def _merge_segments_for_display(seg: pd.DataFrame, max_rows: int = 40) -> pd.DataFrame:
    """若段过多，将相邻小段合并为更大块（仅影响展示表，不修改原始点数据）。"""
    if seg.empty or len(seg) <= max_rows:
        return seg
    # 每 k 行合并为一块
    k = int(np.ceil(len(seg) / max_rows))
    merged = []
    for i in range(0, len(seg), k):
        block = seg.iloc[i : i + k]
        merged.append(
            {
                "depth_top_m": block["depth_top_m"].iloc[0],
                "depth_bottom_m": block["depth_bottom_m"].iloc[-1],
                "thickness_m": round(float(block["depth_bottom_m"].iloc[-1] - block["depth_top_m"].iloc[0]), 2),
                "lithology_mode": _mode_ignore_nan(block["lithology_mode"].values),
                "n_samples": int(block["n_samples"].sum()) if "n_samples" in block.columns else 0,
            }
        )
    return pd.DataFrame(merged)


def _format_segments_markdown(seg: pd.DataFrame) -> str:
    if seg.empty:
        return "（无深度段表：缺少深度列或有效数据过少）"
    lines = [
        "| 深度顶(m) | 深度底(m) | 厚度(m) | 主要岩性 | GR均值 | R90均值 | Cn均值% | Den均值 |",
        "|-----------|-----------|---------|----------|--------|---------|---------|---------|",
    ]
    for _, r in seg.iterrows():
        lines.append(
            f"| {r.get('depth_top_m', '')} | {r.get('depth_bottom_m', '')} | {r.get('thickness_m', '')} | "
            f"{r.get('lithology_mode', '')} | {r.get('gr_mean', '')} | {r.get('rt_mean', '')} | "
            f"{r.get('cn_mean_pct', '')} | {r.get('den_mean', '')} |"
        )
    return "\n".join(lines)


def _heuristic_reservoir_hints(seg: pd.DataFrame) -> str:
    """基于规则化岩性单元与电阻率，给出「储层提示」（非试油结论）。"""
    if seg.empty:
        return "（无）"
    hints = []
    for _, r in seg.iterrows():
        lab = str(r.get("lithology_mode", ""))
        rt_m = r.get("rt_mean")
        rt_max = r.get("rt_max")
        gr_m = r.get("gr_mean")
        top, bot = r.get("depth_top_m"), r.get("depth_bottom_m")
        rt_ok = rt_m is not None and np.isfinite(rt_m)
        if lab == "含油气显示砂岩" or (
            lab in ("砂岩", "粉砂岩") and rt_ok and float(rt_m) > 50
        ):
            rt_str = f"{float(rt_m):.1f}" if rt_ok else "—"
            gr_str = f"{float(gr_m):.1f}" if gr_m is not None and np.isfinite(gr_m) else "—"
            hints.append(
                f"- **{top}–{bot} m**：{lab}，R90 均值约 {rt_str} Ω·m（规则化高阻显示），"
                f"GR 均值约 {gr_str} API；**试油/含油性需结合录井、气测等验证**。"
            )
        if rt_max is not None and np.isfinite(rt_max) and float(rt_max) > 500 and lab not in ("泥岩", "无效数据"):
            hints.append(
                f"- **{top}–{bot} m**：局部极深电阻率约 {rt_max:.1f} Ω·m，需区分**油气、钙质胶结或致密**等；建议结合交会图与邻井。"
            )
    if not hints:
        return "（无显著高阻规则化段；可结合交会图与录井深化）"
    # 去重（近似）
    seen = set()
    out = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return "\n".join(out[:12])


def _format_data_quality_block(
    depth_arr: np.ndarray,
    gr_arr: np.ndarray,
) -> str:
    """数据概况：井段范围、有效段、采样间隔。"""
    d = np.asarray(depth_arr, dtype=float)
    g = np.asarray(gr_arr, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return "- 深度列无效或为空。\n"
    dmin, dmax = float(np.min(d)), float(np.max(d))
    step = _estimate_sample_interval(np.asarray(depth_arr, dtype=float))
    ok = np.isfinite(g)
    if np.any(ok):
        first_valid = np.where(ok)[0][0]
        last_valid = np.where(ok)[0][-1]
        depth_full = np.asarray(depth_arr, dtype=float)
        d_valid_top = float(depth_full[first_valid]) if first_valid < len(depth_full) else dmin
        d_valid_bot = float(depth_full[last_valid]) if last_valid < len(depth_full) else dmax
    else:
        d_valid_top, d_valid_bot = dmin, dmax
    invalid_early = ""
    if np.any(ok):
        first_ok_depth = float(np.asarray(depth_arr, dtype=float)[np.where(ok)[0][0]])
        if first_ok_depth > dmin + 1e-3:
            invalid_early = f"- **浅部无效段**：约 **{dmin:.1f} – {first_ok_depth:.1f} m** GR 为空或无效，解释以有效深度起算。\n"
    lines = [
        f"- **井段（深度列）**：{dmin:.1f} – {dmax:.1f} m；**估计采样间隔**：约 {step:.3f} m（由深度差分中位数）。\n",
        invalid_early,
        f"- **有效 GR 起点（近似）**：约 **{d_valid_top:.1f} m** 起至 **{d_valid_bot:.1f} m**（仍可能有局部缺失）。\n",
    ]
    return "".join(lines)


def _classify_lithology_multicurve(
    gr: np.ndarray,
    den: Optional[np.ndarray],
    neu: Optional[np.ndarray],
    rt: Optional[np.ndarray],
    sp: Optional[np.ndarray],
    dt: Optional[np.ndarray],
) -> np.ndarray:
    """
    多曲线岩性判别（与行业常用 GR-密度-中子-电阻率-SP-Dt 组合思路一致）。
    条件按优先级自上而下匹配（np.select 第一条命中为准）。
    """
    n = len(gr)
    den_a = den if den is not None else np.full(n, np.nan)
    neu_a = neu if neu is not None else np.full(n, np.nan)
    rt_a = rt if rt is not None else np.full(n, np.nan)
    sp_a = sp if sp is not None else np.full(n, np.nan)
    dt_a = dt if dt is not None else np.full(n, np.nan)

    valid_gr = ~np.isnan(gr)
    has_den = ~np.isnan(den_a)
    has_neu = ~np.isnan(neu_a)
    has_rt = ~np.isnan(rt_a)

    # 中子可能为小数形式（0–1），统一到近似“百分数”量级用于阈值
    neu_scaled = neu_a.copy()
    if has_neu.any():
        mx = np.nanmax(neu_scaled)
        if mx <= 1.5:
            neu_scaled = neu_scaled * 100.0

    # SP：负异常指示渗透性砂体（相对中位数）
    sp_neg = np.zeros(n, dtype=bool)
    if sp is not None and np.isfinite(sp_a).any():
        med = np.nanmedian(sp_a[np.isfinite(sp_a)])
        sp_neg = np.isfinite(sp_a) & (sp_a < med - 3.0)

    # 高孔指示：密度偏低、中子偏高、声波时差偏大（单位随数据一致，仅用相对组合）
    high_poro_hint = has_den & has_neu & (den_a < 2.35) & (neu_scaled > 35)

    condlist: List[np.ndarray] = []
    choicelist: List[str] = []

    # 1) GR 无效：不参与解释
    condlist.append(~valid_gr)
    choicelist.append("无效数据")

    # 2) 灰岩：低 GR + 高密度骨架
    condlist.append(valid_gr & has_den & (gr < 56) & (den_a > 2.64))
    choicelist.append("灰岩")

    # 3) 泥岩：高 GR，或 GR 偏高且高中子（泥质/高孔泥岩）
    condlist.append(
        valid_gr
        & (
            (gr > 86)
            | ((gr > 70) & has_neu & (neu_scaled > 50))
            | ((gr > 75) & has_den & (den_a > 2.45) & (gr <= 86))
        )
    )
    choicelist.append("泥岩")

    # 4) 含油气显示砂岩：中等 GR + 明显高阻（需电阻率列）
    condlist.append(
        valid_gr
        & has_rt
        & (rt_a > 80)
        & (gr > 35)
        & (gr < 72)
        & (den_a < 2.72)
        & ~(has_den & (den_a > 2.64) & (gr < 56))
    )
    choicelist.append("含油气显示砂岩")

    # 5) 砂岩：低 GR + 较低密度（较净砂）
    condlist.append(valid_gr & has_den & (gr < 48) & (den_a < 2.62) & (den_a > 1.15))
    choicelist.append("砂岩")

    # 6) 致密砂岩 / 粉砂岩：中等 GR + 中高密度
    condlist.append(valid_gr & has_den & (gr < 72) & (den_a >= 2.28) & (den_a <= 2.62) & (gr >= 40))
    choicelist.append("致密砂岩")

    # 7) 泥质粉砂岩：中等 GR + 高中子、偏低密度
    condlist.append(
        valid_gr & (gr >= 50) & (gr <= 76) & has_neu & (neu_scaled > 45) & has_den & (den_a < 2.45)
    )
    choicelist.append("泥质粉砂岩")

    # 8) SP 负异常 + 中低 GR → 偏砂岩储层特征
    condlist.append(valid_gr & sp_neg & (gr < 72) & (gr > 30))
    choicelist.append("砂岩")

    # 9) 声波时差辅助：高孔-软地层（在粉砂岩主带内细化）
    condlist.append(
        valid_gr & (gr >= 45) & (gr <= 78) & np.isfinite(dt_a) & (dt_a > 90) & high_poro_hint
    )
    choicelist.append("粉砂岩")

    # 10) 主带粉砂岩类
    condlist.append(valid_gr & (gr >= 45) & (gr <= 82))
    choicelist.append("粉砂岩")

    lith = np.select(condlist, choicelist, default="粉砂岩")
    return lith


@tool
def interpret_lithology(
    data_path: str,
    gr_column: str = "GR",
    density_column: str = "DEN",
    neutron_column: str = "CNL",
    sp_column: str = "",
    dt_column: str = "",
    resistivity_column: str = "",
    cal_column: str = "",
    segment_bin_m: float = DEFAULT_SEGMENT_BIN_M,
) -> str:
    """
    岩性解释 - 基于多测井曲线综合判别岩性（自然伽马、密度、中子、深电阻率、自然电位、声波时差等）。

    参数:
        data_path: 数据文件路径
        gr_column: 自然伽马列名，默认 GR（兼容 Gr）
        density_column: 密度列名，默认 DEN（兼容 Den/RHOB）
        neutron_column: 中子列名，默认 CNL（兼容 Cn/NPHI）
        sp_column: 自然电位列名，默认自动识别 SP/Sp
        dt_column: 声波时差列名，默认自动识别 DT/Dt/AC
        resistivity_column: 深电阻率优先列名；默认识别 R90→R60→…→Rx
        cal_column: 井径列名，预留（当前不参与分类，可扩展）
        segment_bin_m: 深度段划分分箱长度（米），默认 15；越大段越少、越接近「单元」尺度

    返回:
        岩性解释结果摘要（含数据概况、深度段表、储层提示、分类规则）
    """
    _ = cal_column  # 预留井径质量标记等扩展
    logger.info(f"开始岩性解释: {data_path}")
    try:
        df = load_dataframe(data_path)

        gr_col = _first_resolved_column(df, [gr_column, "GR", "Gr", "gr"])
        den_col = _first_resolved_column(df, [density_column, "DEN", "Den", "RHOB", "rhob"])
        neu_col = _first_resolved_column(
            df, [neutron_column, "CNL", "Cn", "NPHI", "nphi"]
        )
        sp_col = sp_column.strip()
        if not sp_col:
            sp_col = _first_resolved_column(df, ["SP", "Sp", "sp", "自然电位"])
        dt_col = dt_column.strip()
        if not dt_col:
            dt_col = _first_resolved_column(df, ["DT", "Dt", "AC", "ac", "DTCO", "声波时差"])
        rt_col = _auto_resistivity_column(df, resistivity_column.strip())

        if not gr_col:
            return f"错误: 未找到自然伽马列 '{gr_column}'。可用列: {', '.join(df.columns.tolist())}"

        gr = _numeric_series(df, gr_col)
        if gr is None:
            return "错误: 伽马列无法转为数值。"

        den = _numeric_series(df, den_col) if den_col else None
        neu = _numeric_series(df, neu_col) if neu_col else None
        rt = _numeric_series(df, rt_col) if rt_col else None
        sp = _numeric_series(df, sp_col) if sp_col else None
        dt = _numeric_series(df, dt_col) if dt_col else None

        lith_arr = _classify_lithology_multicurve(gr, den, neu, rt, sp, dt)
        df["Lithology"] = lith_arr

        lithology_stats = df["Lithology"].value_counts(normalize=True).to_dict()

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_lithology.csv"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        depth_col = _resolve_depth_column(df)
        seg_df = pd.DataFrame()
        seg_display = pd.DataFrame()
        segments_path = ""
        res_hints = "（无深度列，未生成深度段表与储层段提示；请检查数据中是否包含 Depth/MD 等深度列）"
        bin_m_out = DEFAULT_SEGMENT_BIN_M
        if depth_col:
            try:
                bin_m = float(segment_bin_m) if segment_bin_m and segment_bin_m > 0 else DEFAULT_SEGMENT_BIN_M
            except (TypeError, ValueError):
                bin_m = DEFAULT_SEGMENT_BIN_M
            bin_m_out = bin_m
            seg_df = build_lithology_segments_binned(
                df,
                depth_col,
                "Lithology",
                bin_m=bin_m,
                gr_col=gr_col,
                rt_col=rt_col or "",
                neu_col=neu_col or "",
                den_col=den_col or "",
                dt_col=dt_col or "",
            )
            res_hints = _heuristic_reservoir_hints(seg_df)
            seg_display = _merge_segments_for_display(seg_df, max_rows=40)
            segments_path = os.path.join(dir_name, f"{name}_lithology_segments.csv")
            seg_df.to_csv(segments_path, index=False, encoding="utf-8-sig")

        used = [
            f"GR={gr_col}",
            f"深度={depth_col or '无'}",
            f"密度={den_col or '无'}",
            f"中子={neu_col or '无'}",
            f"电阻率={rt_col or '无'}",
            f"SP={sp_col or '无'}",
            f"声波时差={dt_col or '无'}",
        ]

        depth_arr = df[depth_col].values if depth_col else np.arange(len(df), dtype=float)
        dq_block = _format_data_quality_block(depth_arr, gr)
        if depth_col and segments_path and not seg_display.empty:
            seg_md = _format_segments_markdown(seg_display)
        elif depth_col and not seg_df.empty:
            seg_md = _format_segments_markdown(seg_df)
        else:
            seg_md = _format_segments_markdown(pd.DataFrame())

        result = f"""
        === 岩性解释完成（多曲线综合 + 深度段汇总） ===
        输入文件: {data_path}
        逐点结果: {output_path}
        """
        if segments_path:
            result += f"深度段表: {segments_path}\n"
        result += f"""
        【参与字段】{'; '.join(used)}
        【深度分箱】{bin_m_out} m（可调工具参数 segment_bin_m；越大段越少）

        ## 一、数据概况
        {dq_block}

        ## 二、解释方法（经验表 + 本工具规则）
        {METHODOLOGY_TABLE_MD}
        - 交会图、纵向追踪等需在报告中结合 **plot_crossplot / plot_well_log_curves** 等进一步说明。

        ## 三、岩性占比（逐点统计）
        """
        for rock_type, percentage in lithology_stats.items():
            result += f"- {rock_type}: {percentage*100:.2f}%\n"

        result += f"""
        ## 四、深度段划分（分箱众数，供报告「单元划分」；非地质分层结论）
        {seg_md}

        ## 五、储层与含油气性提示（规则化，非试油）
        {res_hints}

        【逐点分类思路】
        - 无效值占位（如 -999.25）已在加载时置空；GR 缺失行记为「无效数据」。
        - 自然伽马(GR)、密度、中子、深电阻率、SP、声波时差组合判别，优先级见下。

        【分类优先级（自上而下命中）】
        无效数据 → 灰岩 → 泥岩 → 含油气显示砂岩 → 砂岩 → 致密砂岩 → 泥质粉砂岩 → SP/声波辅助 → 粉砂岩（默认）

        岩性解释已完成；综合报告可由监督智能体根据本结果撰写。
        """
        return result

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"岩性解释失败: {str(e)}")
        return f"错误: 岩性解释时发生异常 - {str(e)}"


@tool
def identify_reservoir(data_path: str, porosity_column: str = "Porosity", permeability_column: str = "Permeability", sample_interval: float = 0.125) -> str:
    """
    储层识别 - 识别和评价储层

    参数:
        data_path: 数据文件路径
        porosity_column: 孔隙度列名，默认"Porosity"
        permeability_column: 渗透率列名，默认"Permeability"
        sample_interval: 采样间隔，默认0.125米

    返回:
        储层识别结果
    """
    logger.info(f"开始储层识别: {data_path}")
    try:
        df = load_dataframe(data_path)

        def _first_existing_column(candidates: list) -> str:
            for cand in candidates:
                if not cand:
                    continue
                c = resolve_column(df, cand) or (cand if cand in df.columns else "")
                if c and c in df.columns:
                    return c
            return ""

        poro_col = _first_existing_column(
            [
                porosity_column,
                "Porosity",
                "PHIT",
                "PHIE",
                "POR",
                "porosity",
                "孔隙度",
            ]
        )
        perm_col = _first_existing_column(
            [
                permeability_column,
                "Permeability",
                "PERM",
                "perm",
                "渗透率",
                "K",
            ]
        )
        porosity = df.get(poro_col) if poro_col else None
        permeability = df.get(perm_col) if perm_col else None

        if porosity is None or permeability is None:
            return (
                "错误: 未找到孔隙度或渗透率列，无法完成储层识别。"
                f" 默认期望列名: {porosity_column}, {permeability_column}。当前文件列: {', '.join(df.columns.tolist())}。"
                " 若仅有测井曲线而无孔渗解释成果，请补充孔隙度/渗透率列后再试，或仅进行岩性解释与曲线类分析。"
            )

        # 向量化：避免逐行 Python 循环，大数据时显著加速
        poro_vals = np.where(pd.isna(porosity.values), 0, porosity.values)
        perm_vals = np.where(pd.isna(permeability.values), 0, permeability.values)
        reservoir_quality = np.where(
            (poro_vals > 15) & (perm_vals > 100), "优质储层",
            np.where((poro_vals > 10) & (perm_vals > 10), "中等储层",
            np.where(poro_vals > 5, "差储层", "非储层")),
        )
        df["Reservoir_Quality"] = reservoir_quality

        reservoir_zones = df[df['Reservoir_Quality'].isin(['优质储层', '中等储层'])]
        total_thickness = len(reservoir_zones) * sample_interval

        reservoir_stats = df['Reservoir_Quality'].value_counts().to_dict()

         # 生成输出路径（统一保存为 CSV，避免扩展名与格式不一致导致下游读取失败）
        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_reservoir.csv"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        result = f"""
        === 储层识别完成 ===
        输入文件: {data_path}
        输出文件: {output_path}

        【储层参数】
        孔隙度列: {poro_col}
        渗透率列: {perm_col}
        采样间隔: {sample_interval}米

        【储层统计】
        """
        for quality, count in reservoir_stats.items():
            result += f"- {quality}: {count}个采样点\n"

        result += f"""
        【储层评价】
        有效储层（优质+中等）总厚度: {total_thickness:.2f}米

        【储层分级标准】
        - 优质储层: 孔隙度 > 15%, 渗透率 > 100mD
        - 中等储层: 孔隙度 > 10%, 渗透率 > 10mD
        - 差储层: 孔隙度 > 5%
        - 非储层: 孔隙度 ≤ 5%

        储层识别已完成，结果已保存！
        """
        return result

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"储层识别失败: {str(e)}")
        return f"错误: 储层识别时发生异常 - {str(e)}"
