"""
数据可视化工具 - 测井数据图表生成

【工具】测井曲线图、岩性分布图、交会图、相关性热力图、储层剖面图。统一使用 matplotlib 输出 PNG，
高 DPI、清晰字体、抗锯齿，保证图表精确易读。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from langchain_core.tools import tool
import logging

from tools.data_loader import load_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局绘图配置：字体、抗锯齿、清晰度
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 1.2,
    'lines.antialiased': True,
    'path.simplify': True,
    'path.simplify_threshold': 0.5,
})

# 曲线图最大点数（提高以保留更多细节，兼顾渲染速度）
MAX_PLOT_POINTS = 1200
# 输出 DPI（高清晰度）
OUTPUT_DPI = 150

# 岩性标准配色（测井/地质常用，支持中英文）
LITHOLOGY_COLORS = {
    "Sandstone": "#E8D4A4", "砂岩": "#E8D4A4",
    "Shale": "#B8B8B8", "泥岩": "#B8B8B8",
    "Limestone": "#E8E8E8", "石灰岩": "#E8E8E8",
    "Siltstone": "#C8C4B8", "粉砂岩": "#C8C4B8",
    "Dolomite": "#D4D4C8", "白云岩": "#D4D4C8",
    "Unknown": "#F5F5F5", "未知": "#F5F5F5",
}

# 储层质量配色（支持多种表述）
RESERVOIR_COLORS = {
    "优质储层": "#2E7D32", "优质": "#2E7D32", "好": "#2E7D32",
    "中等储层": "#66BB6A", "中等": "#66BB6A", "中": "#66BB6A",
    "差储层": "#FFA726", "差": "#FFA726", "劣": "#FFA726",
    "非储层": "#EEEEEE", "非储": "#EEEEEE", "无效": "#EEEEEE",
}


def _resolve_column(df: pd.DataFrame, name: str) -> str:
    """解析列名，支持大小写不敏感匹配。若未找到则返回空字符串。"""
    if name in df.columns:
        return name
    lower = name.lower()
    for col in df.columns:
        if col.lower() == lower:
            return col
    return ""


def _downsample_for_plot(df: pd.DataFrame, max_points: int = None) -> pd.DataFrame:
    """等间隔抽点保留首尾，确保深度范围和关键边界不丢失。"""
    max_points = max_points or MAX_PLOT_POINTS
    if len(df) <= max_points:
        return df
    indices = np.unique(np.linspace(0, len(df) - 1, max_points, dtype=int))
    return df.iloc[indices].reset_index(drop=True)


def _sort_by_depth(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    """按深度升序排序，确保曲线从上到下正确绘制。"""
    return df.sort_values(depth_col, ascending=True).reset_index(drop=True)


def _try_merge_lithology(df: pd.DataFrame, data_path: str, depth_col: str) -> pd.DataFrame:
    """
    尝试合并同目录下的岩性文件（*_lithology.csv），与 identify_reservoir 输出配套使用。
    路径规则：xxx_reservoir.csv -> xxx_lithology.csv
    """
    base = os.path.basename(data_path)
    name, _ = os.path.splitext(base)
    # 若当前是 reservoir 文件，推导 lithology 路径
    if "_reservoir" in name:
        lith_name = name.replace("_reservoir", "_lithology") + ".csv"
    else:
        lith_name = name + "_lithology.csv"
    lith_path = os.path.join(os.path.dirname(data_path), lith_name)
    if not os.path.isfile(lith_path):
        return df
    try:
        lith_df = load_dataframe(lith_path)
        lith_col = _resolve_column(lith_df, "Lithology") or "Lithology"
        depth_lith = _resolve_column(lith_df, depth_col) or _resolve_column(lith_df, "depth") or depth_col
        if lith_col not in lith_df.columns or depth_lith not in lith_df.columns:
            return df
        if len(lith_df) != len(df):
            return df
        # 按深度对齐合并（假定行序一致）
        df = df.copy()
        df["Lithology"] = lith_df[lith_col].values
        logger.info(f"已合并岩性数据: {lith_path}")
    except Exception as e:
        logger.warning(f"合并岩性文件失败: {e}")
    return df


def _draw_strip_log(ax, depth: np.ndarray, categories: np.ndarray, color_map: dict, y_label: str = "深度(m)") -> None:
    """
    绘制岩性道/储层道条带图。按连续相同类别绘制水平填色区间，深度边界精确对齐。
    """
    if len(depth) == 0 or len(categories) == 0:
        return
    depth = np.asarray(depth, dtype=float)
    cats = [str(c) if pd.notna(c) and str(c).strip() else "Unknown" for c in categories]
    # 计算典型采样间距（用于最后一段的底部延伸）
    if len(depth) > 1:
        d_sorted = np.sort(np.unique(depth))
        diffs = np.diff(d_sorted)
        d_avg = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else (d_sorted[-1] - d_sorted[0]) / max(len(d_sorted) - 1, 1)
    else:
        d_avg = 0.125
    i = 0
    while i < len(cats):
        cat = cats[i]
        color = color_map.get(cat, color_map.get("Unknown", "#F5F5F5"))
        j = i
        while j + 1 < len(cats) and cats[j + 1] == cat:
            j += 1
        d_top = float(depth[i])
        # 最后一段用 d_avg 延伸，否则用下一采样点与当前点的中点作为边界
        if j + 1 < len(depth):
            d_bot = (float(depth[j]) + float(depth[j + 1])) / 2
        else:
            d_bot = float(depth[j]) + d_avg
        ax.axhspan(d_top, d_bot, 0, 1, color=color, linewidth=0, edgecolor="none")
        i = j + 1
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel(y_label, fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.35, linestyle="-", linewidth=0.6)


# 测井曲线绘制优先级（自动选择时优先使用）
_CURVE_PRIORITY = ["GR", "DEN", "CNL", "AC", "RT", "SP", "CALI", "PHIT", "Porosity", "Permeability"]


@tool
def plot_well_log_curves(data_path: str, curves: list = None, depth_column: str = 'depth') -> str:
    """
    绘制测井曲线综合图。支持多道并列、按深度排序、网格线，自动优选 GR/DEN/CNL 等标准曲线。
    """
    logger.info(f"开始绘制测井曲线图: {data_path}")
    try:
        df = load_dataframe(data_path)

        depth_col = _resolve_column(df, depth_column) or _resolve_column(df, "Depth") or _resolve_column(df, "depth") or ""
        if not depth_col:
            return f"错误: 未找到深度列。可用列: {', '.join(df.columns.tolist())}"

        df = _sort_by_depth(df, depth_col)
        df = _downsample_for_plot(df)

        if curves is None:
            numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != depth_col]
            priority_lower = [p.lower() for p in _CURVE_PRIORITY]
            def _order(c):
                cl = c.lower()
                for i, pl in enumerate(priority_lower):
                    if cl == pl or pl in cl or cl in pl:
                        return i
                return 999
            numeric.sort(key=_order)
            curves = numeric[:6]
        else:
            curves = [c for c in curves if _resolve_column(df, c) or c in df.columns][:6]

        if not curves:
            return f"错误: 没有可绘制的数值曲线。可用列: {', '.join(df.columns.tolist())}"

        n_curves = len(curves)
        fig, axes = plt.subplots(1, n_curves, figsize=(5 * n_curves, 10), sharey=True, constrained_layout=True)
        if n_curves == 1:
            axes = [axes]
        for i, curve in enumerate(curves):
            curve_col = _resolve_column(df, curve) or (curve if curve in df.columns else "")
            if curve_col:
                vals = df[curve_col].replace([np.inf, -np.inf], np.nan)
                axes[i].plot(vals, df[depth_col], color="#1a5fb4", linewidth=2.0, antialiased=True)
                axes[i].invert_yaxis()
                axes[i].grid(True, axis="both", alpha=0.4, linestyle="--", linewidth=0.5)
                axes[i].set_axisbelow(True)
            axes[i].set_title(curve_col or curve, fontsize=12, fontweight="bold")
            axes[i].set_xlabel(curve_col or curve, fontsize=11)
            axes[i].tick_params(labelsize=10)
        axes[0].set_ylabel("深度 (m)", fontsize=11)
        fig.suptitle("测井曲线综合图", fontsize=15, fontweight="bold", y=1.02)
        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_curves_plot.png")
        os.makedirs(dir_name, exist_ok=True)
        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        return f"测井曲线图已生成，保存至: {output_path}"

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_lithology_distribution(data_path: str, lithology_column: str = 'Lithology') -> str:
    """
    绘制岩性分布图

    参数:
        data_path: 数据文件路径
        lithology_column: 岩性列名，默认'Lithology'

    返回:
        绘图结果信息
    """
    logger.info(f"开始绘制岩性分布图: {data_path}")
    try:
        df = load_dataframe(data_path)

        lith_col = _resolve_column(df, lithology_column) or (lithology_column if lithology_column in df.columns else "")
        if not lith_col:
            return f"错误: 未找到岩性列 '{lithology_column}'。可用列: {', '.join(df.columns.tolist())}"

        fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

        lithology_counts = df[lith_col].value_counts()
        labels = lithology_counts.index.tolist()
        fallback = ['#2563eb', '#ea580c', '#7c3aed', '#dc2626', '#059669',
                    '#4f46e5', '#ca8a04', '#0891b2', '#db2777', '#65a30d']
        colors = [LITHOLOGY_COLORS.get(str(l), fallback[i % len(fallback)]) for i, l in enumerate(labels)]

        wedges, texts, autotexts = axes[0].pie(
            lithology_counts.values, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90, explode=[0.02] * len(labels),
            textprops={"fontsize": 11}
        )
        for t in autotexts:
            t.set_fontsize(10)
        axes[0].set_title('岩性分布比例', fontsize=14, fontweight='bold')

        bars = axes[1].bar(range(len(labels)), lithology_counts.values, color=colors, alpha=0.9, edgecolor='white', linewidth=1)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45 if len(labels) > 4 else 0, ha='right', fontsize=11)
        axes[1].set_xlabel('岩性', fontsize=12)
        axes[1].set_ylabel('样本点数', fontsize=12)
        axes[1].set_title('岩性样本点统计', fontsize=14, fontweight='bold')
        axes[1].grid(True, axis='y', alpha=0.4, linestyle='--')

        for bar, value in zip(bars, lithology_counts.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(lithology_counts) * 0.02,
                        f'{int(value)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, ext = os.path.splitext(base_name)
        output_filename = f"{name}_lithology_distribution.png"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)

        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        return f"岩性分布图已生成，保存至: {output_path}"

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_crossplot(data_path: str, x_parameter: str, y_parameter: str,
                   color_by: str = None, depth_range: list = None) -> str:
    """
    绘制交会图

    参数:
        data_path: 数据文件路径
        x_parameter: x轴参数列名
        y_parameter: y轴参数列名
        color_by: 按该列着色（可选）
        depth_range: 深度范围 [min, max]（可选）

    返回:
        绘图结果信息
    """
    logger.info(f"开始绘制交会图: {data_path}")
    try:
        df = load_dataframe(data_path)

        depth_col = _resolve_column(df, "depth") or _resolve_column(df, "Depth") or ""
        if depth_range and depth_col:
            df = df[(df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])]

        x_col = _resolve_column(df, x_parameter) or (x_parameter if x_parameter in df.columns else "")
        y_col = _resolve_column(df, y_parameter) or (y_parameter if y_parameter in df.columns else "")
        if not x_col or not y_col:
            return f"错误: 未找到指定的参数列。x需为 {x_parameter}，y需为 {y_parameter}。可用列: {', '.join(df.columns.tolist())}"

        n_pts = len(df)
        point_size = max(8, min(60, 8000 / max(n_pts, 1)))  # 自适应点大小

        fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)

        color_col = _resolve_column(df, color_by) or (color_by if color_by and color_by in df.columns else None)
        if color_col:
            if df[color_col].dtype == 'object' or df[color_col].dtype.name == 'category':
                categories = df[color_col].astype('category')
                codes = categories.cat.codes
                unique_categories = list(categories.cat.categories)
                n_cat = max(len(unique_categories), 1)
                scatter = ax.scatter(df[x_col], df[y_col], c=codes, cmap='tab10',
                                    alpha=0.75, s=point_size, edgecolors='white', linewidths=0.5)
                handles = [
                    plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=plt.cm.tab10(i / max(n_cat - 1, 1)),
                               markersize=10, label=cat, markeredgewidth=0.5, markeredgecolor='white')
                    for i, cat in enumerate(unique_categories)
                ]
                ax.legend(handles=handles, title=color_col, fontsize=10, title_fontsize=11)
            else:
                scatter = ax.scatter(df[x_col], df[y_col], c=df[color_col],
                                    cmap='viridis', alpha=0.75, s=point_size, edgecolors='white', linewidths=0.5)
                cbar = plt.colorbar(scatter, ax=ax, label=color_col, shrink=0.85, pad=0.02)
                cbar.ax.tick_params(labelsize=9)
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.75, s=point_size, c='#2563eb', edgecolors='white', linewidths=0.5)

        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{x_col} vs {y_col} 交会图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=10)

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_{x_col}_{y_col}_crossplot.png"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)
        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        return f"交会图已生成，保存至: {output_path}"

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_heatmap(data_path: str, parameters: list = None) -> str:
    """
    绘制参数相关性热力图

    参数:
        data_path: 数据文件路径
        parameters: 需要分析相关性的参数列表，默认所有数值列

    返回:
        绘图结果信息
    """
    logger.info(f"开始绘制热力图: {data_path}")
    try:
        df = load_dataframe(data_path)

        if parameters:
            valid_params = []
            for p in (parameters or []):
                if not p:
                    continue
                c = _resolve_column(df, str(p)) or (str(p) if str(p) in df.columns else "")
                if c and c in df.columns:
                    valid_params.append(c)
            numeric_df = df[valid_params].select_dtypes(include=[np.number]) if valid_params else df.select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        corr_matrix = numeric_df.corr()
        n_params = len(corr_matrix)
        annot_font = max(7, min(12, 120 // max(n_params, 1)))  # 参数多时字号自动减小

        fig, ax = plt.subplots(figsize=(max(10, n_params * 1.1), max(8, n_params * 0.9)), constrained_layout=True)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.8, cbar_kws={"shrink": 0.75, "label": "相关系数", "pad": 0.02},
                    annot_kws={"size": annot_font, "weight": "bold"}, vmin=-1, vmax=1,
                    xticklabels=True, yticklabels=True)

        ax.set_title('测井参数相关性热力图', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_correlation_heatmap.png"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)
        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        return f"相关性热力图已生成，保存至: {output_path}"

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_reservoir_profile(data_path: str, depth_column: str = 'depth',
                           gr_column: str = None, porosity_column: str = None,
                           lithology_column: str = 'Lithology', reservoir_column: str = 'Reservoir_Quality') -> str:
    """
    绘制专业储层剖面图，含 GR 曲线、孔隙度曲线、岩性道、储层质量道。

    支持自动合并同目录 *_lithology.csv；储层质量来自 identify_reservoir 输出的 Reservoir_Quality 列。
    深度按升序排列，GR 采用 0–200 API 标准刻度，孔隙度采用 0–40% 刻度。
    """
    logger.info(f"开始绘制储层剖面图: {data_path}")
    try:
        df = load_dataframe(data_path)

        depth_col = _resolve_column(df, depth_column) or _resolve_column(df, "Depth") or _resolve_column(df, "depth") or ""
        if not depth_col:
            return f"错误: 未找到深度列 '{depth_column}'。可用列: {', '.join(df.columns.tolist())}"

        # 按深度排序，确保曲线自上而下正确
        df = _sort_by_depth(df, depth_col)
        df = _downsample_for_plot(df)

        # 尝试合并岩性文件（当输入为 reservoir CSV 时）
        df = _try_merge_lithology(df, data_path, depth_col)

        depth = df[depth_col].values

        # 优先选择 GR、孔隙度，支持常用列名
        gr_col = _resolve_column(df, gr_column or "GR") or (gr_column if gr_column and gr_column in df.columns else None)
        if not gr_col:
            for c in ["GR", "gr", "gamma", "Gamma"]:
                gr_col = _resolve_column(df, c)
                if gr_col:
                    break
        poro_col = _resolve_column(df, porosity_column or "Porosity") or (
            porosity_column if porosity_column and porosity_column in df.columns else None
        )
        if not poro_col:
            for c in ["Porosity", "porosity", "PHIT", "phit", "PHI"]:
                poro_col = _resolve_column(df, c)
                if poro_col:
                    break

        lith_col = _resolve_column(df, lithology_column) or _resolve_column(df, "Lithology") or ("Lithology" if "Lithology" in df.columns else None)
        res_col = _resolve_column(df, reservoir_column) or _resolve_column(df, "Reservoir_Quality") or ("Reservoir_Quality" if "Reservoir_Quality" in df.columns else None)

        # 确定子图顺序：曲线道 + 岩性道 + 储层道
        curve_cols = []
        if gr_col:
            curve_cols.append(("GR", gr_col, 0, 200))   # API 单位标准刻度
        if poro_col:
            poro_vals = df[poro_col].dropna()
            p_max = poro_vals.max() if len(poro_vals) else 1
            xlim = (0, 40) if p_max > 1.5 else (0, 1)
            curve_cols.append(("孔隙度" + ("(%)" if p_max > 1.5 else ""), poro_col, xlim[0], xlim[1]))
        if not curve_cols:
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != depth_col]
            if numeric_cols:
                curve_cols.append((numeric_cols[0], numeric_cols[0], None, None))

        n_tracks = len(curve_cols) + (1 if lith_col else 0) + (1 if res_col else 0)
        if n_tracks == 0:
            return f"错误: 需要至少一个曲线列。可用列: {', '.join(df.columns.tolist())}"

        fig, axes = plt.subplots(1, n_tracks, figsize=(4 * n_tracks, 12), sharey=True, constrained_layout=True)
        if n_tracks == 1:
            axes = [axes]

        idx = 0
        for title, col, xmin, xmax in curve_cols:
            ax = axes[idx]
            vals = df[col].replace([np.inf, -np.inf], np.nan)
            ax.plot(vals, depth, color="#1a5fb4", linewidth=2.0, antialiased=True)
            if xmin is not None and xmax is not None:
                ax.set_xlim(xmin, xmax)
            ax.invert_yaxis()
            ax.set_xlabel(title, fontsize=11)
            ax.grid(True, axis="both", alpha=0.4, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=10)
            if idx == 0:
                ax.set_ylabel("深度 (m)", fontsize=11)
            idx += 1

        if lith_col:
            ax = axes[idx]
            _draw_strip_log(ax, depth, df[lith_col].values, LITHOLOGY_COLORS, "深度 (m)" if idx == 0 else "")
            ax.set_title("岩性", fontsize=12, fontweight="bold")
            idx += 1

        if res_col:
            ax = axes[idx]
            _draw_strip_log(ax, depth, df[res_col].values, RESERVOIR_COLORS, "深度 (m)" if idx == 0 else "")
            ax.set_title("储层质量", fontsize=12, fontweight="bold")
            idx += 1

        fig.suptitle("储层剖面综合图", fontsize=15, fontweight="bold", y=1.02)

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_reservoir_profile.png")
        os.makedirs(dir_name, exist_ok=True)
        fig.patch.set_facecolor("white")
        plt.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close()

        return f"储层剖面图已生成，保存至: {output_path}"

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"储层剖面图绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"