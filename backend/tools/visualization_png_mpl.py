"""
Matplotlib 静态 PNG 导出

与 Plotly HTML 配套：同名路径生成 PNG，供 Word 解释报告嵌入。网页交互仍以 Plotly 为准。
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tools.data_loader import load_dataframe, resolve_column

logger = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 160,
    }
)

CURVE_COLOR = "#1a5276"
LITHOLOGY_COLORS = {
    "Sandstone": "#E8D4A4", "砂岩": "#E8D4A4",
    "Shale": "#B8B8B8", "泥岩": "#B8B8B8",
    "Limestone": "#E8E8E8", "石灰岩": "#E8E8E8",
    "Siltstone": "#C8C4B8", "粉砂岩": "#C8C4B8",
    "Dolomite": "#D4D4C8", "白云岩": "#D4D4C8",
    "Unknown": "#F5F5F5", "未知": "#F5F5F5",
}
RESERVOIR_COLORS = {
    "优质储层": "#2E7D32", "优质": "#2E7D32", "好": "#2E7D32",
    "中等储层": "#66BB6A", "中等": "#66BB6A", "中": "#66BB6A",
    "差储层": "#FFA726", "差": "#FFA726", "劣": "#FFA726",
    "非储层": "#EEEEEE", "非储": "#EEEEEE", "无效": "#EEEEEE",
}
_MAX_POINTS = 1500
_CURVE_PRIORITY = ["GR", "DEN", "CNL", "AC", "RT", "SP", "CALI", "PHIT", "Porosity", "Permeability"]


def _downsample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= _MAX_POINTS:
        return df
    idx = np.unique(np.linspace(0, len(df) - 1, _MAX_POINTS, dtype=int))
    return df.iloc[idx].reset_index(drop=True)


def export_well_log_curves_png(
    data_path: str,
    png_path: str,
    curves: Optional[List[str]] = None,
    depth_column: str = "depth",
) -> None:
    df = load_dataframe(data_path)
    depth_col = resolve_column(df, depth_column) or resolve_column(df, "Depth") or resolve_column(df, "depth") or ""
    if not depth_col:
        raise ValueError("无深度列")
    df = df.sort_values(depth_col, ascending=True).reset_index(drop=True)
    df = _downsample(df)

    if curves is None:
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != depth_col]
        pl = [p.lower() for p in _CURVE_PRIORITY]

        def _order(c):
            cl = c.lower()
            for i, p in enumerate(pl):
                if cl == p or p in cl or cl in p:
                    return i
            return 999

        numeric.sort(key=_order)
        curves = numeric[:6]
    else:
        curves = [c for c in curves if resolve_column(df, c) or c in df.columns][:6]

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 9), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    for i, curve in enumerate(curves):
        cc = resolve_column(df, curve) or curve
        if cc not in df.columns:
            continue
        vals = df[cc].replace([np.inf, -np.inf], np.nan)
        axes[i].plot(vals, df[depth_col], color=CURVE_COLOR, linewidth=1.8)
        if cc.upper() in ("GR", "GR_CLEAN", "GR_NORM") and not vals.isna().all():
            vmin = float(np.nanmin(vals))
            axes[i].fill_betweenx(df[depth_col], vmin, vals, alpha=0.12, color=CURVE_COLOR)
        axes[i].invert_yaxis()
        axes[i].set_title(cc, fontsize=11)
        axes[i].grid(True, alpha=0.35)
    axes[0].set_ylabel("深度 (m)")
    fig.suptitle("测井曲线综合图（静态）", fontsize=13)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_lithology_distribution_png(data_path: str, png_path: str, lithology_column: str = "Lithology") -> None:
    df = load_dataframe(data_path)
    lith_col = resolve_column(df, lithology_column) or (
        lithology_column if lithology_column in df.columns else ""
    )
    if not lith_col:
        raise ValueError("无岩性列")
    counts = df[lith_col].value_counts()
    labels = counts.index.tolist()
    fallback = ["#1a5276", "#c0392b", "#2874a6", "#6c3483", "#1e8449"]
    colors = [LITHOLOGY_COLORS.get(str(l), fallback[i % len(fallback)]) for i, l in enumerate(labels)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("岩性分布比例")
    axes[1].bar(range(len(labels)), counts.values, color=colors, edgecolor="#333")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_title("岩性样本点统计")
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_crossplot_png(
    data_path: str,
    png_path: str,
    x_parameter: str,
    y_parameter: str,
    color_by: Optional[str] = None,
    depth_range: Optional[list] = None,
) -> None:
    df = load_dataframe(data_path)
    depth_col = resolve_column(df, "depth") or resolve_column(df, "Depth") or ""
    if depth_range and depth_col:
        df = df[(df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])]
    x_col = resolve_column(df, x_parameter) or x_parameter
    y_col = resolve_column(df, y_parameter) or y_parameter
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("交会图列不存在")

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    color_col = resolve_column(df, color_by) if color_by else None
    if color_col and color_col in df.columns:
        if df[color_col].dtype == object or str(df[color_col].dtype) == "category":
            cats = df[color_col].astype("category")
            for j, cat in enumerate(cats.cat.categories):
                m = cats == cat
                ax.scatter(df.loc[m, x_col], df.loc[m, y_col], s=18, alpha=0.75, label=str(cat))
            ax.legend(fontsize=8)
        else:
            sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap="viridis", s=18, alpha=0.75)
            plt.colorbar(sc, ax=ax, label=color_col)
    else:
        ax.scatter(df[x_col], df[y_col], c=CURVE_COLOR, s=20, alpha=0.75, edgecolors="white", linewidths=0.3)
        x_clean = df[x_col].dropna()
        y_clean = df[y_col].dropna()
        valid = ~(x_clean.isna() | y_clean.isna())
        xc = x_clean[valid].values.astype(float)
        yc = y_clean[valid].values.astype(float)
        if len(xc) > 2:
            z = np.polyfit(xc, yc, 1)
            p = np.poly1d(z)
            xl = np.linspace(xc.min(), xc.max(), 80)
            ax.plot(xl, p(xl), "r--", linewidth=1.5, alpha=0.85)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} 交会图（静态）", fontsize=12)
    ax.grid(True, alpha=0.35)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_heatmap_png(data_path: str, png_path: str, parameters: Optional[List[Any]] = None) -> None:
    df = load_dataframe(data_path)
    if parameters:
        valid = []
        for p in parameters:
            if not p:
                continue
            c = resolve_column(df, str(p)) or (str(p) if str(p) in df.columns else "")
            if c and c in df.columns:
                valid.append(c)
        numeric_df = df[valid].select_dtypes(include=[np.number]) if valid else df.select_dtypes(include=[np.number])
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    if corr.empty:
        raise ValueError("无数值列")

    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(6, n * 0.8)), constrained_layout=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("测井参数相关性热力图（静态）", fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _strip_segments_mpl(depth: np.ndarray, categories: np.ndarray):
    depth = np.asarray(depth, dtype=float)
    cats = [str(c) if pd.notna(c) and str(c).strip() else "Unknown" for c in categories]
    if len(depth) > 1:
        ds = np.sort(np.unique(depth))
        diffs = np.diff(ds)
        d_avg = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else (ds[-1] - ds[0]) / max(len(ds) - 1, 1)
    else:
        d_avg = 0.125
    segs = []
    i = 0
    while i < len(cats):
        cat = cats[i]
        j = i
        while j + 1 < len(cats) and cats[j + 1] == cat:
            j += 1
        d_top = float(depth[i])
        if j + 1 < len(depth):
            d_bot = (float(depth[j]) + float(depth[j + 1])) / 2
        else:
            d_bot = float(depth[j]) + d_avg
        segs.append((d_top, d_bot, cat))
        i = j + 1
    return segs


def _try_merge_lithology(df: pd.DataFrame, data_path: str, depth_col: str) -> pd.DataFrame:
    base = os.path.basename(data_path)
    name, _ = os.path.splitext(base)
    if "_reservoir" in name:
        lith_name = name.replace("_reservoir", "_lithology") + ".csv"
    else:
        lith_name = name + "_lithology.csv"
    lith_path = os.path.join(os.path.dirname(data_path), lith_name)
    if not os.path.isfile(lith_path):
        return df
    try:
        lith_df = load_dataframe(lith_path)
        lith_col = resolve_column(lith_df, "Lithology") or "Lithology"
        depth_lith = resolve_column(lith_df, depth_col) or resolve_column(lith_df, "depth") or depth_col
        if lith_col not in lith_df.columns or depth_lith not in lith_df.columns:
            return df
        if len(lith_df) != len(df):
            return df
        df = df.copy()
        df["Lithology"] = lith_df[lith_col].values
    except Exception as e:
        logger.warning("合并岩性文件失败: %s", e)
    return df


def export_reservoir_profile_png(
    data_path: str,
    png_path: str,
    depth_column: str = "depth",
    gr_column: Optional[str] = None,
    porosity_column: Optional[str] = None,
    lithology_column: str = "Lithology",
    reservoir_column: str = "Reservoir_Quality",
) -> None:
    df = load_dataframe(data_path)
    depth_col = resolve_column(df, depth_column) or resolve_column(df, "Depth") or resolve_column(df, "depth") or ""
    if not depth_col:
        raise ValueError("无深度列")
    df = df.sort_values(depth_col, ascending=True).reset_index(drop=True)
    df = _downsample(df)
    df = _try_merge_lithology(df, data_path, depth_col)
    depth = df[depth_col].values

    gr_col = resolve_column(df, gr_column or "GR")
    if not gr_col:
        for c in ["GR", "gr", "gamma"]:
            gr_col = resolve_column(df, c)
            if gr_col:
                break
    poro_col = resolve_column(df, porosity_column or "Porosity")
    if not poro_col:
        for c in ["Porosity", "PHIT", "phit"]:
            poro_col = resolve_column(df, c)
            if poro_col:
                break
    lith_col = resolve_column(df, lithology_column) or ("Lithology" if "Lithology" in df.columns else None)
    res_col = resolve_column(df, reservoir_column) or ("Reservoir_Quality" if "Reservoir_Quality" in df.columns else None)

    curve_specs = []
    if gr_col:
        curve_specs.append(("GR", gr_col, 0, 200))
    if poro_col:
        pv = df[poro_col].dropna()
        p_max = float(pv.max()) if len(pv) else 1
        xlim = (0, 40) if p_max > 1.5 else (0, 1)
        curve_specs.append(("孔隙度", poro_col, xlim[0], xlim[1]))
    if not curve_specs:
        nums = [c for c in df.select_dtypes(include=[np.number]).columns if c != depth_col]
        if nums:
            curve_specs.append((nums[0], nums[0], None, None))
        else:
            raise ValueError("无曲线列")

    n_tracks = len(curve_specs) + (1 if lith_col else 0) + (1 if res_col else 0)
    fig, axes = plt.subplots(1, n_tracks, figsize=(3.5 * n_tracks, 10), sharey=True, constrained_layout=True)
    if n_tracks == 1:
        axes = [axes]
    idx = 0
    for title, col, xmin, xmax in curve_specs:
        ax = axes[idx]
        vals = df[col].replace([np.inf, -np.inf], np.nan)
        ax.plot(vals, depth, color=CURVE_COLOR, linewidth=1.8)
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
        ax.invert_yaxis()
        ax.set_xlabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("深度 (m)")
        idx += 1

    if lith_col:
        ax = axes[idx]
        for d0, d1, cat in _strip_segments_mpl(depth, df[lith_col].values):
            c = LITHOLOGY_COLORS.get(cat, "#F5F5F5")
            ax.axhspan(d0, d1, xmin=0, xmax=1, color=c, clip_on=True)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title("岩性", fontsize=9)
        idx += 1

    if res_col:
        ax = axes[idx]
        for d0, d1, cat in _strip_segments_mpl(depth, df[res_col].values):
            c = RESERVOIR_COLORS.get(cat, "#EEEEEE")
            ax.axhspan(d0, d1, xmin=0, xmax=1, color=c, clip_on=True)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title("储层质量", fontsize=9)

    fig.suptitle("储层剖面综合图（静态）", fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)) or ".", exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
