"""
数据可视化工具 - 测井数据图表生成

【工具】测井曲线图、岩性分布图、交会图、相关性热力图、储层剖面图。
- 网页：Plotly 独立 HTML（交互）。
- 报告：Matplotlib 生成同名 PNG（供 Word 嵌入）。可选环境变量 SKIP_PLOTLY_PNG=1 跳过 PNG。
"""
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.tools import tool

from tools.data_loader import load_dataframe, resolve_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plotly 交互：滚轮缩放、工具栏、框选/套索（散点图等 2D 图）
PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["select2d", "lasso2d", "drawclosedpath", "eraseselect"],
}

# 全局字体（浏览器端常见中文字体）
FONT_FAMILY = "Microsoft YaHei, SimHei, PingFang SC, sans-serif"

# 曲线图最大点数
MAX_PLOT_POINTS = 1500

# 专业配色
CURVE_COLOR = "#1a5276"
CURVE_FILL_COLOR = "rgba(26, 82, 118, 0.12)"

# 岩性标准配色
LITHOLOGY_COLORS = {
    "Sandstone": "#E8D4A4", "砂岩": "#E8D4A4",
    "Shale": "#B8B8B8", "泥岩": "#B8B8B8",
    "Limestone": "#E8E8E8", "石灰岩": "#E8E8E8",
    "Siltstone": "#C8C4B8", "粉砂岩": "#C8C4B8",
    "Dolomite": "#D4D4C8", "白云岩": "#D4D4C8",
    "Unknown": "#F5F5F5", "未知": "#F5F5F5",
}

# 储层质量配色
RESERVOIR_COLORS = {
    "优质储层": "#2E7D32", "优质": "#2E7D32", "好": "#2E7D32",
    "中等储层": "#66BB6A", "中等": "#66BB6A", "中": "#66BB6A",
    "差储层": "#FFA726", "差": "#FFA726", "劣": "#FFA726",
    "非储层": "#EEEEEE", "非储": "#EEEEEE", "无效": "#EEEEEE",
}

_CURVE_PRIORITY = ["GR", "DEN", "CNL", "AC", "RT", "SP", "CALI", "PHIT", "Porosity", "Permeability"]


def _downsample_for_plot(df: pd.DataFrame, max_points: int = None) -> pd.DataFrame:
    max_points = max_points or MAX_PLOT_POINTS
    if len(df) <= max_points:
        return df
    indices = np.unique(np.linspace(0, len(df) - 1, max_points, dtype=int))
    return df.iloc[indices].reset_index(drop=True)


def _sort_by_depth(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    return df.sort_values(depth_col, ascending=True).reset_index(drop=True)


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
        logger.info(f"已合并岩性数据: {lith_path}")
    except Exception as e:
        logger.warning(f"合并岩性文件失败: {e}")
    return df


def _strip_segments(depth: np.ndarray, categories: np.ndarray) -> list:
    """返回 [(d_top, d_bot, cat_str), ...] 用于 Plotly shapes。"""
    if len(depth) == 0:
        return []
    depth = np.asarray(depth, dtype=float)
    cats = [str(c) if pd.notna(c) and str(c).strip() else "Unknown" for c in categories]
    if len(depth) > 1:
        d_sorted = np.sort(np.unique(depth))
        diffs = np.diff(d_sorted)
        d_avg = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else (d_sorted[-1] - d_sorted[0]) / max(len(d_sorted) - 1, 1)
    else:
        d_avg = 0.125
    segments = []
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
        segments.append((d_top, d_bot, cat))
        i = j + 1
    return segments


def _write_plotly_outputs(
    fig: go.Figure,
    html_path: str,
    mpl_png: Callable[[str], None],
) -> tuple[str, bool, str]:
    """
    写入 Plotly HTML，并用 Matplotlib 写入同名 PNG。
    返回 (html_path, png_ok, png_err)。
    """
    os.makedirs(os.path.dirname(html_path) or ".", exist_ok=True)
    fig.write_html(
        html_path,
        include_plotlyjs="cdn",
        config=PLOTLY_CONFIG,
        full_html=True,
    )
    base, _ = os.path.splitext(html_path)
    png_path = base + ".png"
    if os.environ.get("SKIP_PLOTLY_PNG", "").strip().lower() in ("1", "true", "yes"):
        return html_path, False, "已设置 SKIP_PLOTLY_PNG，跳过静态 PNG"
    try:
        mpl_png(png_path)
        if os.path.isfile(png_path) and os.path.getsize(png_path) > 0:
            return html_path, True, ""
        return html_path, False, "Matplotlib 未写出有效 PNG 文件"
    except Exception as e:
        logger.exception("Matplotlib 静态 PNG 失败: %s", png_path)
        return html_path, False, f"{type(e).__name__}: {e}"


def _result_msg(kind: str, html_path: str, png_ok: bool, png_err: str = "") -> str:
    """工具返回给 Agent 的说明文本。"""
    msg = f"{kind}已生成，交互图: {html_path}"
    if png_ok:
        base, _ = os.path.splitext(html_path)
        msg += f" | 静态图（可嵌入报告）: {base}.png"
    else:
        msg += " | 静态 PNG 未生成。"
        if png_err:
            msg += f" {png_err}"
    return msg


@tool
def plot_well_log_curves(data_path: str, curves: list = None, depth_column: str = "depth") -> str:
    """
    绘制测井曲线综合图（Plotly HTML：可缩放、平移、悬浮查看数值）。
    """
    logger.info(f"开始绘制测井曲线图: {data_path}")
    try:
        df = load_dataframe(data_path)

        depth_col = resolve_column(df, depth_column) or resolve_column(df, "Depth") or resolve_column(df, "depth") or ""
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
            curves = [c for c in curves if resolve_column(df, c) or c in df.columns][:6]

        if not curves:
            return f"错误: 没有可绘制的数值曲线。可用列: {', '.join(df.columns.tolist())}"

        n_curves = len(curves)
        depth_min, depth_max = float(df[depth_col].min()), float(df[depth_col].max())
        depth_info = f"深度 {depth_min:.1f}–{depth_max:.1f} m | 共 {len(df)} 点"

        fig = make_subplots(
            rows=1,
            cols=n_curves,
            shared_yaxes=True,
            horizontal_spacing=0.04,
            subplot_titles=[resolve_column(df, c) or c for c in curves],
        )

        for i, curve in enumerate(curves, start=1):
            curve_col = resolve_column(df, curve) or (curve if curve in df.columns else "")
            if not curve_col:
                continue
            vals = df[curve_col].replace([np.inf, -np.inf], np.nan)
            yv = df[depth_col].values
            xv = vals.values.astype(float)

            is_gr = curve_col.upper() in ("GR", "GR_CLEAN", "GR_NORM") and not vals.isna().all()
            if is_gr:
                v_min = float(np.nanmin(vals))
                fig.add_trace(
                    go.Scatter(
                        x=np.full(len(yv), v_min),
                        y=yv,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=i,
                )
                fig.add_trace(
                    go.Scatter(
                        x=xv,
                        y=yv,
                        mode="lines",
                        name=curve_col,
                        line=dict(color=CURVE_COLOR, width=2.2),
                        fill="tonextx",
                        fillcolor=CURVE_FILL_COLOR,
                        hovertemplate=f"{curve_col}: %{{x:.4f}}<br>深度: %{{y:.2f}} m<extra></extra>",
                    ),
                    row=1,
                    col=i,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=xv,
                        y=yv,
                        mode="lines",
                        name=curve_col,
                        line=dict(color=CURVE_COLOR, width=2.2),
                        hovertemplate=f"{curve_col}: %{{x:.4f}}<br>深度: %{{y:.2f}} m<extra></extra>",
                    ),
                    row=1,
                    col=i,
                )

            fig.update_xaxes(title_text=curve_col, row=1, col=i, showgrid=True, gridcolor="#e0e0e0")
            fig.update_yaxes(autorange="reversed", row=1, col=i, showgrid=True, gridcolor="#e0e0e0")

        fig.update_yaxes(title_text="深度 (m)", row=1, col=1)
        fig.update_layout(
            title=dict(text=f"测井曲线综合图<br><sup>{depth_info}</sup>", font=dict(size=16, family=FONT_FAMILY)),
            height=900,
            width=min(400 * n_curves, 2400),
            font=dict(family=FONT_FAMILY, size=12),
            template="plotly_white",
            hovermode="closest",
            dragmode="zoom",
        )

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_curves_plot.html")

        def _mpl_png(p: str) -> None:
            from tools.visualization_png_mpl import export_well_log_curves_png

            export_well_log_curves_png(data_path, p, curves=curves, depth_column=depth_column)

        _, png_ok, png_err = _write_plotly_outputs(fig, output_path, _mpl_png)

        return _result_msg("测井曲线图", output_path, png_ok, png_err)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_lithology_distribution(data_path: str, lithology_column: str = "Lithology") -> str:
    """绘制岩性分布图（Plotly：饼图 + 柱状图，可缩放与悬浮）。"""
    logger.info(f"开始绘制岩性分布图: {data_path}")
    try:
        df = load_dataframe(data_path)

        lith_col = resolve_column(df, lithology_column) or (lithology_column if lithology_column in df.columns else "")
        if not lith_col:
            return f"错误: 未找到岩性列 '{lithology_column}'。可用列: {', '.join(df.columns.tolist())}"

        lithology_counts = df[lith_col].value_counts()
        labels = lithology_counts.index.tolist()
        total = int(lithology_counts.sum())
        fallback = ["#1a5276", "#c0392b", "#2874a6", "#6c3483", "#1e8449",
                    "#d35400", "#2980b9", "#7d3c98", "#27ae60", "#8e44ad"]
        colors = [LITHOLOGY_COLORS.get(str(l), fallback[i % len(fallback)]) for i, l in enumerate(labels)]

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=(f"岩性分布比例 | N={total}", "岩性样本点统计"),
        )

        fig.add_trace(
            go.Pie(
                labels=labels,
                values=lithology_counts.values,
                marker=dict(colors=colors, line=dict(color="#333", width=1)),
                textinfo="label+percent",
                hovertemplate="%{label}<br>占比: %{percent}<br>数量: %{value}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=labels,
                y=lithology_counts.values,
                marker=dict(color=colors, line=dict(color="#333", width=1)),
                text=[str(int(v)) for v in lithology_counts.values],
                textposition="outside",
                hovertemplate="%{x}<br>点数: %{y}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="岩性", row=1, col=2, tickangle=-45 if len(labels) > 4 else 0)
        fig.update_yaxes(title_text="样本点数", row=1, col=2)
        fig.update_layout(
            height=700,
            width=1200,
            font=dict(family=FONT_FAMILY, size=12),
            template="plotly_white",
            showlegend=False,
            dragmode="zoom",
        )

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_lithology_distribution.html")

        def _mpl_png(p: str) -> None:
            from tools.visualization_png_mpl import export_lithology_distribution_png

            export_lithology_distribution_png(data_path, p, lithology_column=lithology_column)

        _, png_ok, png_err = _write_plotly_outputs(fig, output_path, _mpl_png)

        return _result_msg("岩性分布图", output_path, png_ok, png_err)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_crossplot(
    data_path: str,
    x_parameter: str,
    y_parameter: str,
    color_by: str = None,
    depth_range: list = None,
) -> str:
    """绘制交会图（Plotly：支持框选、套索选点、滚轮缩放）。"""
    logger.info(f"开始绘制交会图: {data_path}")
    try:
        df = load_dataframe(data_path)

        depth_col = resolve_column(df, "depth") or resolve_column(df, "Depth") or ""
        if depth_range and depth_col:
            df = df[(df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])]

        x_col = resolve_column(df, x_parameter) or (x_parameter if x_parameter in df.columns else "")
        y_col = resolve_column(df, y_parameter) or (y_parameter if y_parameter in df.columns else "")
        if not x_col or not y_col:
            return f"错误: 未找到指定的参数列。x需为 {x_parameter}，y需为 {y_parameter}。可用列: {', '.join(df.columns.tolist())}"

        n_pts = len(df)
        point_size = max(6, min(16, int(8000 / max(n_pts, 1)) ** 0.5))

        color_col = resolve_column(df, color_by) or (color_by if color_by and color_by in df.columns else None)

        fig = go.Figure()

        if color_col:
            if df[color_col].dtype == "object" or str(df[color_col].dtype) == "category":
                categories = df[color_col].astype("category")
                unique_categories = list(categories.cat.categories)
                palette = [
                    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
                ]
                for i, cat in enumerate(unique_categories):
                    mask = categories == cat
                    sub = df.loc[mask]
                    fig.add_trace(
                        go.Scatter(
                            x=sub[x_col],
                            y=sub[y_col],
                            mode="markers",
                            name=str(cat),
                            marker=dict(
                                size=point_size,
                                color=palette[i % len(palette)],
                                line=dict(width=0.5, color="white"),
                            ),
                            hovertemplate=f"{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>{color_col}: {cat}<extra></extra>",
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode="markers",
                        marker=dict(
                            size=point_size,
                            color=df[color_col],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title=color_col),
                            line=dict(width=0.5, color="white"),
                        ),
                        hovertemplate=f"{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<br>{color_col}: %{{marker.color:.4f}}<extra></extra>",
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[y_col],
                    mode="markers",
                    name="数据点",
                    marker=dict(size=point_size, color=CURVE_COLOR, line=dict(width=0.8, color="white")),
                    hovertemplate=f"{x_col}: %{{x:.4f}}<br>{y_col}: %{{y:.4f}}<extra></extra>",
                )
            )

        if not color_col:
            x_vals = df[x_col].dropna()
            y_vals = df[y_col].dropna()
            valid = ~(x_vals.isna() | y_vals.isna())
            x_clean = x_vals[valid].values.astype(float)
            y_clean = y_vals[valid].values.astype(float)
            if len(x_clean) > 2:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
                r2 = np.corrcoef(x_clean, y_clean)[0, 1] ** 2
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=p(x_line),
                        mode="lines",
                        name=f"线性拟合 R²={r2:.3f}",
                        line=dict(color="#c0392b", width=2, dash="dash"),
                    )
                )

        fig.update_layout(
            title=dict(text=f"{x_col} vs {y_col} 交会图 | N={n_pts}", font=dict(size=16, family=FONT_FAMILY)),
            xaxis=dict(title=x_col, gridcolor="#e0e0e0"),
            yaxis=dict(title=y_col, gridcolor="#e0e0e0"),
            height=820,
            width=1000,
            font=dict(family=FONT_FAMILY, size=12),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            dragmode="select",
            selectdirection="any",
        )

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_{x_col}_{y_col}_crossplot.html"
        output_path = os.path.join(dir_name, output_filename)

        def _mpl_png(p: str) -> None:
            from tools.visualization_png_mpl import export_crossplot_png

            export_crossplot_png(
                data_path,
                p,
                x_parameter=x_parameter,
                y_parameter=y_parameter,
                color_by=color_by,
                depth_range=depth_range,
            )

        _, png_ok, png_err = _write_plotly_outputs(fig, output_path, _mpl_png)

        return _result_msg("交会图", output_path, png_ok, png_err)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_heatmap(data_path: str, parameters: list = None) -> str:
    """绘制参数相关性热力图（Plotly：可缩放、悬浮显示相关系数）。"""
    logger.info(f"开始绘制热力图: {data_path}")
    try:
        df = load_dataframe(data_path)

        if parameters:
            valid_params = []
            for p in (parameters or []):
                if not p:
                    continue
                c = resolve_column(df, str(p)) or (str(p) if str(p) in df.columns else "")
                if c and c in df.columns:
                    valid_params.append(c)
            numeric_df = df[valid_params].select_dtypes(include=[np.number]) if valid_params else df.select_dtypes(include=[np.number])
        else:
            numeric_df = df.select_dtypes(include=[np.number])

        corr_matrix = numeric_df.corr()
        n_params = len(corr_matrix)
        if n_params == 0:
            return "错误: 没有可用的数值列用于相关性分析。"

        labels = list(corr_matrix.columns)
        z = corr_matrix.values

        text = [[f"{v:.2f}" for v in row] for row in z]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=max(8, min(12, 140 // max(n_params, 1)))),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="相关系数"),
                hovertemplate="%{y} vs %{x}<br>r = %{z:.4f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(text=f"测井参数相关性热力图 (共 {n_params} 个参数)", font=dict(size=16, family=FONT_FAMILY)),
            height=max(500, n_params * 55),
            width=max(600, n_params * 55),
            font=dict(family=FONT_FAMILY, size=11),
            template="plotly_white",
            xaxis=dict(side="bottom", tickangle=-45),
            yaxis=dict(autorange="reversed"),
            dragmode="zoom",
        )

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_correlation_heatmap.html")

        def _mpl_png(p: str) -> None:
            from tools.visualization_png_mpl import export_heatmap_png

            export_heatmap_png(data_path, p, parameters=parameters)

        _, png_ok, png_err = _write_plotly_outputs(fig, output_path, _mpl_png)

        return _result_msg("相关性热力图", output_path, png_ok, png_err)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"


@tool
def plot_reservoir_profile(
    data_path: str,
    depth_column: str = "depth",
    gr_column: str = None,
    porosity_column: str = None,
    lithology_column: str = "Lithology",
    reservoir_column: str = "Reservoir_Quality",
) -> str:
    """绘制储层剖面图（Plotly HTML：曲线道可缩放平移，岩性/储层道为色块填充）。"""
    logger.info(f"开始绘制储层剖面图: {data_path}")
    try:
        df = load_dataframe(data_path)

        depth_col = resolve_column(df, depth_column) or resolve_column(df, "Depth") or resolve_column(df, "depth") or ""
        if not depth_col:
            return f"错误: 未找到深度列 '{depth_column}'。可用列: {', '.join(df.columns.tolist())}"

        df = _sort_by_depth(df, depth_col)
        df = _downsample_for_plot(df)
        df = _try_merge_lithology(df, data_path, depth_col)

        depth = df[depth_col].values

        gr_col = resolve_column(df, gr_column or "GR") or (gr_column if gr_column and gr_column in df.columns else None)
        if not gr_col:
            for c in ["GR", "gr", "gamma", "Gamma"]:
                gr_col = resolve_column(df, c)
                if gr_col:
                    break
        poro_col = resolve_column(df, porosity_column or "Porosity") or (
            porosity_column if porosity_column and porosity_column in df.columns else None
        )
        if not poro_col:
            for c in ["Porosity", "porosity", "PHIT", "phit", "PHI"]:
                poro_col = resolve_column(df, c)
                if poro_col:
                    break

        lith_col = resolve_column(df, lithology_column) or resolve_column(df, "Lithology") or ("Lithology" if "Lithology" in df.columns else None)
        res_col = resolve_column(df, reservoir_column) or resolve_column(df, "Reservoir_Quality") or ("Reservoir_Quality" if "Reservoir_Quality" in df.columns else None)

        curve_specs = []
        if gr_col:
            curve_specs.append(("GR", gr_col, 0, 200))
        if poro_col:
            poro_vals = df[poro_col].dropna()
            p_max = poro_vals.max() if len(poro_vals) else 1
            xlim = (0, 40) if p_max > 1.5 else (0, 1)
            curve_specs.append(("孔隙度" + ("(%)" if p_max > 1.5 else ""), poro_col, xlim[0], xlim[1]))
        if not curve_specs:
            numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != depth_col]
            if numeric_cols:
                curve_specs.append((numeric_cols[0], numeric_cols[0], None, None))

        n_tracks = len(curve_specs) + (1 if lith_col else 0) + (1 if res_col else 0)
        if n_tracks == 0:
            return f"错误: 需要至少一个曲线列。可用列: {', '.join(df.columns.tolist())}"

        fig = make_subplots(
            rows=1,
            cols=n_tracks,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            subplot_titles=[t[0] for t in curve_specs] + (["岩性"] if lith_col else []) + (["储层质量"] if res_col else []),
        )

        col_idx = 1
        for title, col, xmin, xmax in curve_specs:
            vals = df[col].replace([np.inf, -np.inf], np.nan)
            yv = df[depth_col].values
            xv = vals.values.astype(float)

            if title == "GR" and not vals.isna().all():
                v_min = float(np.nanmin(vals))
                fig.add_trace(
                    go.Scatter(
                        x=np.full(len(yv), v_min),
                        y=yv,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=xv,
                        y=yv,
                        mode="lines",
                        line=dict(color=CURVE_COLOR, width=2.2),
                        fill="tonextx",
                        fillcolor=CURVE_FILL_COLOR,
                        name=title,
                        hovertemplate=f"{title}: %{{x:.4f}}<br>深度: %{{y:.2f}} m<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=xv,
                        y=yv,
                        mode="lines",
                        line=dict(color=CURVE_COLOR, width=2.2),
                        name=title,
                        hovertemplate=f"{title}: %{{x:.4f}}<br>深度: %{{y:.2f}} m<extra></extra>",
                    ),
                    row=1,
                    col=col_idx,
                )

            if xmin is not None and xmax is not None:
                fig.update_xaxes(range=[xmin, xmax], row=1, col=col_idx)
            fig.update_xaxes(title_text=title, row=1, col=col_idx, showgrid=True, gridcolor="#e0e0e0")
            fig.update_yaxes(autorange="reversed", row=1, col=col_idx, showgrid=True, gridcolor="#e0e0e0")
            col_idx += 1

        if lith_col:
            segs = _strip_segments(depth, df[lith_col].values)
            xi = col_idx
            for d_top, d_bot, cat in segs:
                color = LITHOLOGY_COLORS.get(cat, LITHOLOGY_COLORS.get("Unknown", "#F5F5F5"))
                fig.add_shape(
                    type="rect",
                    x0=0,
                    x1=1,
                    y0=d_top,
                    y1=d_bot,
                    xref=f"x{xi} domain",
                    yref="y1",
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                )
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[(float(depth.min()) + float(depth.max())) / 2],
                    mode="markers",
                    marker=dict(size=0.01, opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(range=[0, 1], showticklabels=False, row=1, col=col_idx)
            fig.update_yaxes(autorange="reversed", row=1, col=col_idx)
            col_idx += 1

        if res_col:
            segs = _strip_segments(depth, df[res_col].values)
            xi = col_idx
            for d_top, d_bot, cat in segs:
                color = RESERVOIR_COLORS.get(cat, "#EEEEEE")
                fig.add_shape(
                    type="rect",
                    x0=0,
                    x1=1,
                    y0=d_top,
                    y1=d_bot,
                    xref=f"x{xi} domain",
                    yref="y1",
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below",
                )
            fig.add_trace(
                go.Scatter(
                    x=[0.5],
                    y=[(float(depth.min()) + float(depth.max())) / 2],
                    mode="markers",
                    marker=dict(size=0.01, opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.update_xaxes(range=[0, 1], showticklabels=False, row=1, col=col_idx)
            fig.update_yaxes(autorange="reversed", row=1, col=col_idx)

        depth_min, depth_max = float(depth.min()), float(depth.max())
        depth_info = f"深度 {depth_min:.1f}–{depth_max:.1f} m"

        fig.update_yaxes(title_text="深度 (m)", row=1, col=1)
        fig.update_layout(
            title=dict(text=f"储层剖面综合图 | {depth_info}", font=dict(size=16, family=FONT_FAMILY)),
            height=1000,
            width=min(420 * n_tracks, 2800),
            font=dict(family=FONT_FAMILY, size=12),
            template="plotly_white",
            hovermode="closest",
            dragmode="zoom",
        )

        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(dir_name, f"{name}_reservoir_profile.html")

        def _mpl_png(p: str) -> None:
            from tools.visualization_png_mpl import export_reservoir_profile_png

            export_reservoir_profile_png(
                data_path,
                p,
                depth_column=depth_column,
                gr_column=gr_column,
                porosity_column=porosity_column,
                lithology_column=lithology_column,
                reservoir_column=reservoir_column,
            )

        _, png_ok, png_err = _write_plotly_outputs(fig, output_path, _mpl_png)

        return _result_msg("储层剖面图", output_path, png_ok, png_err)

    except FileNotFoundError:
        return f"错误: 文件未找到 - {data_path}"
    except Exception as e:
        logger.error(f"储层剖面图绘图失败: {str(e)}")
        return f"错误: 绘图时发生异常 - {str(e)}"
