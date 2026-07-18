# Task 5 股票涨跌分类模型

本任务使用 `model_data_stock.csv` 中已有的估值、规模、成长类因子，构建股票是否上涨的二分类模型，并输出可分步骤执行的 Notebook 与 ECharts 交互式 HTML 看板。

## 文件说明

- `model_data_stock.csv`：本任务原始建模数据，包含 `Date`、`Code`、`Y` 和多列因子。
- `task5_stock_classification_spec.md`：建模与看板实现规格说明。
- `run_task5_analysis.py`：一键生成 Notebook、HTML 看板和中间结果文件的脚本。
- `task5_stock_classification.ipynb`：可分步骤执行的 Notebook。
- `task5_stock_classification_dashboard.html`：ECharts 交互式结果看板。
- `outputs/model_metrics.csv`：三个模型的 Accuracy、Precision、Recall、F1-score、AUC 和混淆矩阵结果。
- `outputs/model_parameters.csv`：三个模型的关键超参数。
- `outputs/data_quality_summary.csv`：字段缺失值与数据质量检查结果。
- `outputs/train_test_summary.csv`：按时间顺序划分训练集/测试集的摘要。
- `outputs/feature_summary.csv`：入模因子清单和缺失值比例。

## 建模方法

应变量为 `Y`：

- `True` / `1`：上涨
- `False` / `0`：未上涨或下跌

特征选择规则：

- 排除 `Date`、`Code`、`Y`
- 其余可转换为数值型的已有字段全部作为候选因子
- 不生成文件中不存在的技术指标
- 缺失值只使用训练集的中位数填充，避免测试集信息泄漏

训练/测试划分：

- 严格按 `Date` 升序划分
- 前 70% 日期作为训练集
- 后 30% 日期作为测试集
- 同一天的所有股票样本放在同一侧

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

## 结果看板

打开以下文件查看交互式结果：

```text
task5/task5_stock_classification_dashboard.html
```

看板包含：

- 数据质量检查
- 入模因子清单
- 训练集/测试集时间顺序划分
- 三个模型关键参数
- Accuracy、Precision、Recall、F1-score、AUC 对比
- ROC 曲线对比，Legend 可点击显示/隐藏模型
- 混淆矩阵
- 决策树和随机森林特征重要性
- 逻辑回归系数 Top 20
- 三个模型的中文评价和总体结论

## 复现方式

在项目根目录运行：

```bash
python3 -B task5/run_task5_analysis.py
```

如当前 Python 环境缺少依赖，需要先安装：

```bash
python3 -m pip install pandas numpy scikit-learn
```

运行后会刷新：

- `task5/task5_stock_classification.ipynb`
- `task5/task5_stock_classification_dashboard.html`
- `task5/outputs/*.csv`

## 当前结果摘要

当前测试集表现：

| 模型 | Accuracy | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| 随机森林 | 0.5496 | 0.3471 | 0.4805 | 0.4031 | 0.5437 |
| 决策树 | 0.5239 | 0.3308 | 0.4927 | 0.3958 | 0.5163 |
| 逻辑回归 | 0.4532 | 0.3148 | 0.6182 | 0.4171 | 0.5032 |

按 AUC 看，随机森林当前表现最好；按 F1-score 看，逻辑回归当前最高。该结果是分类预测评估，不等同于可直接交易策略，若用于交易需要进一步做回测和稳健性检验。
