# Task 6 模型概率因子交易策略

本任务使用 `model_data.csv` 中的季度股票指标训练分类模型，并将模型输出的上涨概率作为选股因子，构建季度换仓交易策略。

## 文件说明

- `model_data.csv`：本任务原始建模数据，包含 `Date`、`Code`、多列季度财务/估值指标和 `Next_Ret`。
- `task6_model_strategy_spec.md`：建模、策略和看板实现规格说明。
- `run_task6_model_strategy.py`：一键生成 Notebook、HTML 看板和中间结果文件的脚本。
- `task6_model_strategy.ipynb`：可分步骤执行的 Notebook。
- `task6_model_strategy_dashboard.html`：ECharts 交互式结果看板。
- `outputs/data_overview.csv`：数据概览。
- `outputs/data_quality_summary.csv`：字段类型、缺失值和缺失比例。
- `outputs/label_distribution.csv`：由 `Next_Ret > 0` 生成的标签分布。
- `outputs/feature_summary.csv`：入模特征、特征类型、缺失比例和训练集中位数填充值。
- `outputs/train_test_summary.csv`：按季度时间顺序划分训练集/测试集的摘要。
- `outputs/model_parameters.csv`：三个模型的关键默认超参数。
- `outputs/model_metrics.csv`：Accuracy、Precision、Recall、F1-score、AUC 和混淆矩阵结果。
- `outputs/feature_importance.csv`：逻辑回归系数、决策树和随机森林特征重要性。
- `outputs/strategy_quarterly_returns.csv`：各策略每季度收益、换手率、交易成本和资金曲线。
- `outputs/strategy_holdings.csv`：各策略每季度选股、概率、权重、收益和过滤指标。
- `outputs/strategy_metrics.csv`：各策略核心回测指标。

## 建模方法

应变量由 `Next_Ret` 生成：

- `Next_Ret > 0`：上涨，标签为 `1`
- `Next_Ret <= 0`：未上涨或下跌，标签为 `0`

特征处理规则：

- `Date`、`Code`、`Next_Ret` 和标签 `Y` 不进入模型特征。
- CSV 中可转换为数值型的季度指标作为基础特征。
- 对基础特征生成季度截面分位特征。
- 使用每只股票历史季度收益生成 `ret_lag_1q`、`ret_mean_4q`、`ret_vol_4q`、`ret_cum_4q`。
- 使用历史季度收益生成 RSI 和趋势过滤指标。
- 所有历史收益衍生指标均先对 `Next_Ret` 做滞后处理，避免未来数据泄漏。
- 缺失值只使用训练集的中位数填充。

训练/测试划分：

- 严格按 `Date` 升序划分。
- 前 70% 季度作为训练集。
- 后 30% 季度作为测试集。
- 同一季度内所有股票样本放在同一侧，避免时间穿越。

## 模型

实现并对比三个分类模型：

1. 逻辑回归
   - `penalty="elasticnet"`
   - `solver="saga"`
   - `l1_ratio=0.5`
   - `C=1.0`
   - `max_iter=5000`
   - 使用标准化

2. 决策树
   - `criterion="gini"`
   - `splitter="best"`
   - `max_depth=10`
   - `min_samples_split=2`
   - `min_samples_leaf=20`
   - `class_weight="balanced"`

3. 随机森林
   - `n_estimators=100`
   - `criterion="gini"`
   - `max_depth=10`
   - `min_samples_split=2`
   - `min_samples_leaf=10`
   - `max_features="sqrt"`
   - `class_weight="balanced"`

模型评价指标包括：

- Accuracy
- Precision
- Recall
- F1-score
- AUC
- ROC 曲线
- 混淆矩阵

## 交易策略

每个模型都使用测试集中的上涨概率作为选股因子。

默认环境参数：

- 初始资金：100000
- 无风险年化收益率：0.02
- 手续费率：0.0003
- 卖出印花税率：0.0005
- 滑点率：0.0002

默认交易规则：

- 每季度换仓一次。
- 每个模型每季度最多选择 3 只股票。
- 买入概率阈值：`0.55`
- 卖出/规避概率阈值：`0.45`
- 止盈阈值：`0.20`
- 止损阈值：`-0.10`
- RSI 过滤：默认保留 `30 <= RSI <= 80`
- 趋势过滤：默认要求过去 4 个季度累计收益不低于 `0`

仓位方式：

- 等权：入选股票平均分配仓位。
- 概率加权：按模型上涨概率比例分配仓位。

输出策略：

- 逻辑回归-等权
- 逻辑回归-概率加权
- 决策树-等权
- 决策树-概率加权
- 随机森林-等权
- 随机森林-概率加权
- 全市场等权基准
- 随机选 3 只基准

说明：当前数据为季度频率，没有日内或日度价格路径，因此止盈止损按单只股票季度收益近似触发，不等同于真实盘中止盈止损。

## 结果看板

打开以下文件查看交互式结果：

```text
task6/task6_model_strategy_dashboard.html
```

看板包含：

- 数据质量检查
- 入模特征清单
- 训练集/测试集时间顺序划分
- 三个模型关键参数
- 模型 Accuracy、Precision、Recall、F1-score、AUC 对比
- ROC 曲线对比，Legend 可点击显示/隐藏
- 混淆矩阵
- 逻辑回归系数、决策树和随机森林特征重要性
- 交易规则参数展示
- 策略核心指标对比
- 策略净值曲线
- 回撤曲线
- 每季度收益率曲线
- 每季度选股明细
- 中文结论和风险提示

## 复现方式

在项目根目录运行：

```bash
python3 -B task6/run_task6_model_strategy.py
```

运行后会刷新：

- `task6/task6_model_strategy.ipynb`
- `task6/task6_model_strategy_dashboard.html`
- `task6/outputs/*.csv`

如当前 Python 环境缺少依赖，需要先安装：

```bash
python3 -m pip install pandas numpy scikit-learn
```

## 当前结果摘要

当前测试集分类表现：

| 模型 | Accuracy | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| 逻辑回归 | 0.5562 | 0.3669 | 0.7085 | 0.4835 | 0.6390 |
| 决策树 | 0.5068 | 0.3286 | 0.6541 | 0.4374 | 0.5621 |
| 随机森林 | 0.5469 | 0.3524 | 0.6511 | 0.4573 | 0.6052 |

当前回测期末资金排名靠前：

| 策略 | 期末资金 | 累计收益率 | 夏普比率 | 最大回撤 |
|---|---:|---:|---:|---:|
| 随机森林-等权 | 133311.10 | 33.31% | 5.4327 | 0.00% |
| 随机森林-概率加权 | 132962.29 | 32.96% | 5.4744 | 0.00% |
| 逻辑回归-概率加权 | 117156.56 | 17.16% | 2.1031 | 0.00% |

按 AUC 看，逻辑回归当前分类效果最好；按回测期末资金看，随机森林-等权当前表现最好；按夏普比率看，随机森林-概率加权当前表现最好。

历史回测不代表未来收益，策略仍需要结合样本外检验、参数稳健性、流动性、停牌、涨跌停和真实交易执行约束进一步验证。
