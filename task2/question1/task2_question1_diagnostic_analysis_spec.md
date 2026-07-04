# TASK2 问题1基础诊断分析 Spec

## 1. 任务背景

本 spec 针对 TASK2 第一个问题：

> 对数据进行基础的诊断分析，检查缺失值，计算描述性统计量。

拟使用 TASK1 已获取的中芯国际近三年每日交易数据，对 A 股与港股两份数据分别进行基础诊断分析，并在执行后形成可审阅的统计结果与文字解析。

本文件仅作为执行前计划，暂不实际运行统计计算。

## 2. 数据范围

计划使用以下两份 TASK1 数据文件：

- A 股数据：`task1/smic_a_daily.csv`
- 港股数据：`task1/smic_hk_daily.csv`

两份数据当前可识别字段一致：

- `ts_code`：证券代码
- `trade_date`：交易日期
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `pre_close`：前收盘价
- `change`：涨跌额
- `pct_chg`：涨跌幅
- `vol`：成交量
- `amount`：成交额

## 3. 分析目标

本题计划完成以下目标：

1. 检查 A 股与港股每日数据是否存在缺失值。
2. 对数值型字段计算描述性统计量。
3. 对日期、样本数量、字段完整性进行基础诊断。
4. 对 A 股与港股的统计特征进行解释，包括价格、涨跌幅、成交量和成交额的分布情况。
5. 输出适合写入作业报告的结论性文字。

## 4. 诊断分析口径

### 4.1 基础结构检查

对每个数据集分别检查：

- 数据总行数和总列数。
- 字段名称是否符合预期。
- `trade_date` 是否可以正确解析为日期。
- 日期范围，即最早交易日和最晚交易日。
- `trade_date` 是否存在重复记录。
- `ts_code` 是否只有单一证券代码。

### 4.2 缺失值检查

对每个字段分别统计：

- 缺失值数量。
- 缺失值比例。
- 是否存在全空列。
- 是否存在关键字段缺失。

关键字段定义为：

- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `change`
- `pct_chg`
- `vol`
- `amount`

如果发现缺失值，后续报告中应说明：

- 缺失字段。
- 缺失数量和比例。
- 是否影响描述性统计和后续建模。
- 是否需要删除、填补或保留。

### 4.3 异常值和逻辑一致性检查

在基础诊断中增加轻量逻辑检查，但不进行复杂异常值处理：

- `high` 是否小于 `low`。
- `open`、`high`、`low`、`close` 是否存在小于等于 0 的记录。
- `vol`、`amount` 是否存在负值。
- `change` 是否大致等于 `close - pre_close`。
- `pct_chg` 是否大致等于 `change / pre_close * 100`。

上述检查的目的是发现明显数据质量问题，不在本题中直接修正原始数据。

## 5. 描述性统计指标

计划对以下数值字段分别计算描述性统计量：

- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `change`
- `pct_chg`
- `vol`
- `amount`

每个字段至少计算：

- 样本数 `count`
- 均值 `mean`
- 标准差 `std`
- 最小值 `min`
- 最大值 `max`
- 极差 `range`
- 中位数 `median`
- 偏度 `skew`
- 峰度 `kurtosis`
- 1% 分位数 `q01`
- 5% 分位数 `q05`
- 10% 分位数 `q10`
- 25% 分位数 `q25`
- 50% 分位数 `q50`
- 75% 分位数 `q75`
- 90% 分位数 `q90`
- 95% 分位数 `q95`
- 99% 分位数 `q99`
- 缺失值数量 `missing_count`
- 缺失值比例 `missing_ratio`

## 6. 衍生指标

除原始字段外，执行时应增加衍生指标，并对衍生指标同样计算描述性统计：

- `log_return`：对数收益率。
- `amplitude_pct`：日内振幅。
- `intraday_return_pct`：日内收益率。
- `gap_pct`：开盘跳空幅度。
- `upper_shadow_pct`：上影线比例。
- `lower_shadow_pct`：下影线比例。
- `vwap_proxy`：成交额除以成交量得到的近似成交均价。
- `close_ma5_gap_pct`：收盘价相对 5 日均线偏离。
- `close_ma20_gap_pct`：收盘价相对 20 日均线偏离。
- `pct_chg_5d`：5 日收益率。
- `pct_chg_20d`：20 日收益率。
- `volatility_20d`：20 日滚动日涨跌幅标准差。
- `amount_ma20_ratio`：成交额相对 20 日均值倍数。

## 7. 结果解析框架

执行后计划从以下角度进行文字解析。

### 6.1 数据完整性解析

说明：

- A 股数据是否存在缺失值。
- 港股数据是否存在缺失值。
- 两份数据的日期范围是否覆盖近三年。
- 是否存在重复交易日或关键字段异常。

### 6.2 价格指标解析

重点解释：

- `open`、`high`、`low`、`close` 的均值、中位数和分位数。
- 均值与中位数是否接近，用于判断价格分布是否明显偏斜。
- 最大值和最小值反映的价格波动区间。
- A 股与港股价格水平不能直接按数值大小比较，因为交易币种和市场制度不同。

### 6.3 收益波动解析

重点解释：

- `change` 和 `pct_chg` 的均值是否接近 0。
- `pct_chg` 的标准差反映日度涨跌幅波动水平。
- 最大涨幅和最大跌幅是否较极端。
- 分位数用于观察多数交易日的涨跌幅集中区间。

### 6.4 成交活跃度解析

重点解释：

- `vol` 与 `amount` 的均值、中位数和分位数。
- 均值是否显著高于中位数，用于判断成交量或成交额是否受少数高成交日拉高。
- 最大成交量和最大成交额是否体现阶段性交易活跃。
- A 股和港股成交量单位可能不同，比较时应关注各自市场内的分布特征，而不是直接比较绝对数。

### 7.5 策略含义解析

分析不只堆砌统计值，还应解释这些统计特征对后续策略构造的影响：

- 波动率和振幅对止损、仓位和调仓频率的影响。
- 跳空幅度对隔夜风险和盘中交易策略的影响。
- 成交额偏度和成交额相对均值倍数对放量突破、风险释放和流动性过滤的影响。
- A 股与港股指标差异对 A/H 联动、跨市场比较和标准化处理的影响。
- 分位数和尾部统计对极端行情风险控制的意义。

## 8. 建议输出文件

执行本 spec 后，统一在小写 `task2/question1/` 下生成文件。以后任务目录强制使用小写，并且每个问题单独放入对应 question 目录。

- `task2_question1_diagnostic_analysis.ipynb`：完整分析 notebook。
- `task2_question1_missing_summary.csv`：集合版缺失值统计表。
- `task2_question1_descriptive_stats.csv`：集合版描述性统计汇总表。
- `task2_question1_ah_comparison.csv`：A 股与港股对比表。
- `task2_question1_diagnostic_report.md`：文字版诊断分析报告。
- `task2_question1_diagnostic_report.html`：HTML 版诊断分析报告。
- `task2_question1_field_dictionary.csv`：字段英文名、中文名和解释。
- `task2_question1_stat_dictionary.csv`：统计指标英文名和中文名。
- `task2_question1_a_missing_summary.csv`：A 股缺失值统计表。
- `task2_question1_hk_missing_summary.csv`：港股缺失值统计表。
- `task2_question1_a_raw_descriptive_stats.csv`：A 股原始字段描述性统计。
- `task2_question1_hk_raw_descriptive_stats.csv`：港股原始字段描述性统计。
- `task2_question1_a_derived_descriptive_stats.csv`：A 股衍生指标描述性统计。
- `task2_question1_hk_derived_descriptive_stats.csv`：港股衍生指标描述性统计。

报告中应包含可直接复制到 Word 版本中的描述总结。Markdown 报告、HTML 报告、notebook 和主要 CSV 表中应保留英文字段名，并增加中文字段名列，便于阅读和后续代码追溯。

## 9. 建议执行步骤

1. 读取 `task1/smic_a_daily.csv` 和 `task1/smic_hk_daily.csv`。
2. 清理字段名中可能存在的 BOM 或不可见字符。
3. 将 `trade_date` 转换为日期格式。
4. 分别检查 A 股和港股的数据维度、日期范围、重复记录和字段类型。
5. 分别统计每列缺失值数量和缺失值比例。
6. 对数值字段计算描述性统计量。
7. 执行轻量逻辑一致性检查。
8. 计算衍生指标，并对衍生指标进行同样的描述性统计。
9. 合并输出 A 股与港股的缺失值表、描述性统计表和对比表。
10. 根据统计结果撰写中文解析，包含策略含义。
11. 保存结果文件到 `task2/question1/`。

## 10. 暂定技术实现

建议使用 Python notebook 实现，依赖：

- `pandas`
- `numpy`

核心实现思路：

- 使用 `pandas.read_csv()` 读取 CSV。
- 使用 `df.columns.str.replace()` 或重命名方式处理 BOM。
- 使用 `pd.to_datetime(df["trade_date"], format="%Y%m%d")` 解析日期。
- 使用 `df.isna().sum()` 和 `df.isna().mean()` 统计缺失。
- 使用 `df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])` 计算描述性统计。
- 使用 `median()`、`skew()`、`kurt()` 补充中位数、偏度和峰度。

## 11. 审查确认点

执行前建议确认以下事项：

- 任务目录已确认统一使用小写 `task2/`。
- 已确认需要 notebook、分散 CSV、集合 CSV、Markdown 报告和 HTML 报告。
- 已确认需要衍生指标。
- 已确认需要扩展分位数。
- 已确认需要 A 股与港股对比表。

## 12. 当前状态

当前 spec 已根据执行要求更新，后续可运行 `task2/question1/run_question1_analysis.py` 生成全部结果。
