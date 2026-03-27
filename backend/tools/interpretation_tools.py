"""
地质解释工具 - 岩性解释与储层识别

【工具】interpret_lithology（基于 GR/密度/中子划分岩性）、identify_reservoir（基于孔隙度/渗透率分级储层）。
均使用向量化 numpy 运算，大数据时性能优于逐行 Python 循环。输出 CSV 供 plot_lithology_distribution/plot_reservoir_profile 使用。
"""
import os
import pandas as pd
import numpy as np
from langchain_core.tools import tool
import logging

from tools.data_loader import load_dataframe, resolve_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@tool
def interpret_lithology(data_path: str, gr_column: str = "GR", density_column: str = "DEN", neutron_column: str = "CNL") -> str:
    """
    岩性解释 - 基于测井曲线进行岩性分类

    参数:
        data_path: 数据文件路径
        gr_column: 伽马射线曲线列名，默认"GR"
        density_column: 密度曲线列名，默认"DEN"
        neutron_column: 中子曲线列名，默认"CNL"

    返回:
        岩性解释结果
    """
    logger.info(f"开始岩性解释: {data_path}")
    try:
        df = load_dataframe(data_path)

        gr_col = resolve_column(df, gr_column) or (gr_column if gr_column in df.columns else "")
        den_col = resolve_column(df, density_column) or (density_column if density_column in df.columns else "")
        neu_col = resolve_column(df, neutron_column) or (neutron_column if neutron_column in df.columns else "")
        if not gr_col:
            return f"错误: 未找到伽马射线列 '{gr_column}'。可用列: {', '.join(df.columns.tolist())}"
        gr = df[gr_col]
        density = df[den_col] if den_col else None
        neutron = df[neu_col] if neu_col else None

        # 向量化：避免逐行 Python 循环，大数据时显著加速
        gr_vals = gr.values
        mask_na = pd.isna(gr_vals)
        mask_gr_low = ~mask_na & (gr_vals < 50)
        mask_gr_high = ~mask_na & (gr_vals > 100)
        if density is not None:
            den_vals = np.where(pd.isna(density.values), 0, density.values)
            mask_limestone = mask_gr_low & (den_vals > 2.65)
            mask_sandstone = mask_gr_low & ~mask_limestone
        else:
            mask_limestone = np.zeros(len(df), dtype=bool)
            mask_sandstone = mask_gr_low
        lithology = np.where(mask_na, "Unknown",
            np.where(mask_limestone, "Limestone",
            np.where(mask_sandstone, "Sandstone",
            np.where(mask_gr_high, "Shale", "Siltstone"))))
        df["Lithology"] = lithology

        lithology_stats = df['Lithology'].value_counts(normalize=True).to_dict()

        # 生成输出路径（统一 CSV）
        dir_name = os.path.dirname(data_path)
        base_name = os.path.basename(data_path)
        name, _ = os.path.splitext(base_name)
        output_filename = f"{name}_lithology.csv"
        output_path = os.path.join(dir_name, output_filename)
        os.makedirs(dir_name, exist_ok=True)

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        result = f"""
        === 岩性解释完成 ===
        输入文件: {data_path}
        输出文件: {output_path}

        【岩性统计】
        """
        for rock_type, percentage in lithology_stats.items():
            result += f"- {rock_type}: {percentage*100:.2f}%\n"

        result += """
        【曲线物性依据（供报告撰写引用）】
        - 自然伽马(GR)：主要反映泥质含量与放射性矿物，高值多对应泥质岩类，低值多对应净砂岩或碳酸盐岩（需结合密度等）
        - 体积密度(DEN)：反映岩石骨架与孔隙流体综合效应，碳酸盐岩骨架密度通常较高，可与 GR、中子交会区分岩性
        - 中子(CNL)：对含氢指数敏感，在岩性识别中常与密度交会；本工具若未用中子参与阈值，报告中可说明留作交会或质量监控

        【岩性分类规则（本算法阈值）】
        - GR < 50 且 密度 > 2.65 g/cm³: 石灰岩 (Limestone)；无密度列时灰岩判识退化为不启用
        - GR < 50 且 密度 ≤ 2.65 或无密度: 砂岩 (Sandstone)
        - GR > 100: 泥岩 (Shale)
        - 50 ≤ GR ≤ 100: 粉砂岩 (Siltstone)

        岩性解释已完成，结果已保存！
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

        poro_col = resolve_column(df, porosity_column) or porosity_column
        perm_col = resolve_column(df, permeability_column) or permeability_column
        porosity = df.get(poro_col) if poro_col else None
        permeability = df.get(perm_col) if perm_col else None

        if porosity is None or permeability is None:
            return f"错误: 未找到孔隙度或渗透率列。请检查列名：{porosity_column}, {permeability_column}。可用列: {', '.join(df.columns.tolist())}"

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

         # 生成输出路径（统一 CSV）
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
        孔隙度列: {porosity_column}
        渗透率列: {permeability_column}
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