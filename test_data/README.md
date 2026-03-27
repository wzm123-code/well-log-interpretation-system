# 测试数据说明

本目录包含多种测井数据文件，用于全面检验系统应对不同数据场景的能力。

## 文件列表与测试场景

| 文件名 | 测试目的 | 预期行为 |
|--------|----------|----------|
| **sample_well_data.csv** | 标准完整测井数据（原有） | 全流程可正常执行：清洗→解释→储层→绘图→报告 |
| **well_minimal_curves.csv** | 最小可绘图数据：Depth + GR | 可绘制测井曲线图，岩性解释需 GR，储层识别缺孔隙度/渗透率 |
| **well_two_numeric_only.csv** | 仅深度 + 单曲线 | 同上，测试最简曲线图 |
| **well_depth_only.csv** | 仅深度列，无数值曲线 | 数据校验拒绝（数值列过少），提示非测井数据 |
| **well_no_depth.csv** | 无深度列，有 GR/DEN/CNL 等 | 数据校验拒绝（未检测到深度列） |
| **well_with_missing.csv** | 含缺失值 (NaN) | 测试 clean_data 的缺失值处理（drop/fill） |
| **well_with_outliers.csv** | 含明显异常值 | 测试 clean_data 的 IQR 异常值移除 |
| **well_chinese_columns.csv** | 中文列名（深度、密度、中子、孔隙度、渗透率） | 测试 resolve_column 与中文列名兼容；任务规划需传入正确列名 |
| **well_alternate_names.csv** | 小写/别名列名（depth, gr, PHIT, CALI, AC, RT） | 测试大小写不敏感与常用别名解析 |
| **well_no_reservoir_columns.csv** | 无孔隙度、渗透率列 | 岩性解释可执行；储层识别报错并提示缺失列；储层剖面图可能降级 |
| **well_large.csv** | 约 800 行数据 | 测试性能、抽点绘图、超时等 |
| **well_business_like.csv** | 类业务表格（姓名、部门、销售额等） | 数据校验拒绝（数值列占比低、无深度、非测井结构） |
| **test_customer_list.csv** | 客户名单（原有） | 数据校验拒绝 |
| **test_random_columns.csv** | 随机列（原有） | 数据校验拒绝 |

## 快速验证建议

1. **正常流程**：上传 `sample_well_data.csv` 或 `well_chinese_columns.csv`，请求「请进行岩性解释、储层识别并生成报告」。
2. **清洗能力**：上传 `well_with_missing.csv` 或 `well_with_outliers.csv`，请求「先清洗数据再分析」。
3. **校验与跳过**：上传 `well_depth_only.csv`、`well_no_depth.csv` 或 `well_business_like.csv`，观察系统拒绝或友好提示。
4. **绘图可行性**：上传 `well_no_reservoir_columns.csv` 请求储层剖面图，或 `well_minimal_curves.csv` 请求交会图（需指定存在列）。
5. **大数性能**：上传 `well_large.csv`，请求全流程，观察响应时间与绘图效果。
