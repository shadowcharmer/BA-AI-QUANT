# TASK5 股票涨跌分类模型 Notebook 与 HTML 看板 Spec

## 1. 任务目标

使用 `task5/model_data_stock.csv` 构建股票是否上涨的二分类模型，输出：

1. 可分步骤运行和查看图形的 Jupyter Notebook：
   - `task5/task5_stock_classification.ipynb`
2. 可直接打开查看结果的 HTML 看板：
   - `task5/task5_stock_classification_dashboard.html`

模型包括：

1. 逻辑回归
2. 决策树
3. 随机森林

核心评估指标包括：

1. Accuracy
2. Precision
3. Recall
4. F1-score
5. AUC
6. 混淆矩阵
7. ROC 曲线

HTML 看板图形使用可交互图表，ROC 曲线等对比图的 Legend 支持点击显示/隐藏不同模型。

## 2. 当前数据说明

输入文件：

`task5/model_data_stock.csv`

当前已检查到的字段结构：

- 时间字段：`Date`
- 股票代码字段：`Code`
- 标签字段：`Y`
- 因子字段：企业倍数、市净率、市现率、市盈率、市销率、股息率、市值、利润/资产/收入增长率等

本次实现完全按照 `model_data_stock.csv` 文件中已有字段建模，不额外生成或补充文件中不存在的技术指标。

特征选择规则：

1. 排除 `Date`、`Code`、`Y`
2. 其余可转换为数值型的列全部作为候选因子
3. 非数值或无法稳定转换的字段不进入模型
4. 不使用任何未来时间的信息生成特征或填充值

本次 `model_data_stock.csv` 已经有 `Y` 字段，作为应变量。会将其转换为二分类标签：

- `True` / `1` 表示上涨
- `False` / `0` 表示未上涨或下跌

## 3. Notebook 结构

Notebook 按以下步骤组织，便于分步执行和查看图形。

### Step 1：导入依赖与参数配置

主要内容：

- 导入 `pandas`、`numpy`、`scikit-learn`、`plotly` 等库
- 设置输入/输出路径
- 设置训练测试划分参数

可调参数：

```python
DATA_PATH = "model_data_stock.csv"
TRAIN_RATIO = 0.70
RANDOM_STATE = 42
LR_PENALTY = "elasticnet"
LR_SOLVER = "saga"
LR_L1_RATIO = 0.5
LR_C = 1.0
LR_MAX_ITER = 5000
DT_MAX_DEPTH = 10
DT_MIN_SAMPLES_SPLIT = 2
DT_MIN_SAMPLES_LEAF = 20
RF_MAX_DEPTH = 10
RF_N_ESTIMATORS = 100
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 10
```

### Step 2：加载数据

主要内容：

- 读取 CSV
- 解析 `Date`
- 按 `Date`、`Code` 排序
- 显示数据前几行
- 显示数据规模：
  - 行数
  - 列数
  - 股票数量
  - 日期范围
  - 标签分布

### Step 3：数据质量检查

展示内容：

- 字段类型
- 缺失值数量和缺失比例
- 重复行数量
- 重复的 `Date + Code` 数量
- 数值字段的基本描述统计
- 标签 `Y` 的取值检查
- 正负样本比例
- 是否存在无穷大值

输出表：

- 数据概览表
- 缺失值检查表
- 标签分布表
- 重复记录检查结果

### Step 4：特征工程

本次只使用当前文件已有因子。使用当前 CSV 中除以下字段外的数值列作为特征：

- `Date`
- `Code`
- `Y`

当前会使用的候选特征包括：

- 企业倍数 EV/EBITDA
- 市净率 PB
- 市现率 PCF
- 市盈率 PE
- 市销率 PS
- 股息率
- MV
- 净利润同比增长率
- 净资产同比增长率
- 利润总额同比增长率
- 基本每股收益同比增长率
- 总资产同比增长率
- 现金净流量同比增长率
- 营业利润同比增长率
- 营业总收入同比增长率

缺失值处理：

- 数值特征使用训练集的中位数填充
- 无穷大值先替换为缺失值，再填充
- 不使用测试集信息计算填充值，避免数据泄漏

标准化处理：

- 逻辑回归使用标准化
- 决策树不要求标准化
- 随机森林不要求标准化

特征清单会在 Notebook 和 HTML 看板中展示，包括：

- 入模特征名称
- 特征数量
- 每个特征的缺失值比例
- 每个特征在训练集上的填充值，中位数

## 4. 时间顺序训练/测试划分

为避免未来数据泄漏，训练集和测试集按时间顺序划分。

处理方式：

1. 按 `Date` 升序排序
2. 取前 70% 日期作为训练集
3. 取后 30% 日期作为测试集
4. 同一个日期的所有股票样本放在同一侧，避免同一天数据被拆散

看板展示：

- 训练集日期范围
- 测试集日期范围
- 训练集样本数
- 测试集样本数
- 训练/测试比例
- 训练集与测试集标签分布

## 5. 模型训练步骤

### 5.1 逻辑回归

训练流程：

1. 选择训练集特征 `X_train` 和标签 `y_train`
2. 使用训练集中位数填充缺失值
3. 使用 `StandardScaler` 对训练集特征标准化
4. 使用同一个填充器和标准化器处理测试集
5. 训练逻辑回归模型
6. 输出预测类别
7. 输出预测为上涨的概率
8. 计算 Accuracy、Precision、Recall、F1-score、AUC
9. 绘制混淆矩阵和 ROC 曲线

初始参数：

```python
LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    l1_ratio=0.5,
    C=1.0,
    max_iter=5000,
    tol=0.0001,
    fit_intercept=True,
    class_weight="balanced",
    random_state=42
)
```

看板显示的关键参数字段：

- 模型名称：逻辑回归
- 是否标准化：是
- 正则化方式：elasticnet
- 求解器：saga
- L1 比例：0.5
- 正则化强度参数 C：1.0
- 最大迭代次数：5000
- 收敛阈值：0.0001
- 是否拟合截距：是
- 类别权重：balanced
- 训练样本数
- 测试样本数

### 5.2 决策树

训练流程：

1. 选择训练集特征 `X_train` 和标签 `y_train`
2. 使用训练集中位数填充缺失值
3. 不进行标准化
4. 训练决策树分类模型
5. 输出预测类别
6. 输出预测为上涨的概率
7. 计算 Accuracy、Precision、Recall、F1-score、AUC
8. 绘制混淆矩阵和 ROC 曲线
9. 输出特征重要性

初始参数：

```python
DecisionTreeClassifier(
    criterion="gini",
    splitter="best",
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=20,
    max_features=None,
    class_weight="balanced",
    random_state=42
)
```

看板显示的关键参数字段：

- 模型名称：决策树
- 划分标准：gini
- 划分策略：best
- 最大深度：10
- 内部节点再划分最小样本数：2
- 叶子节点最小样本数：20
- 最大特征数：None
- 类别权重：balanced
- 是否标准化：否
- 训练样本数
- 测试样本数

### 5.3 随机森林

训练流程：

1. 选择训练集特征 `X_train` 和标签 `y_train`
2. 使用训练集中位数填充缺失值
3. 不进行标准化
4. 训练随机森林分类模型
5. 输出预测类别
6. 输出预测为上涨的概率
7. 计算 Accuracy、Precision、Recall、F1-score、AUC
8. 绘制混淆矩阵和 ROC 曲线
9. 输出特征重要性

初始参数：

```python
RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=10,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

看板显示的关键参数字段：

- 模型名称：随机森林
- 树数量：100
- 划分标准：gini
- 最大深度：10
- 内部节点再划分最小样本数：2
- 叶子节点最小样本数：10
- 最大特征数：sqrt
- 是否 Bootstrap 抽样：是
- 类别权重：balanced
- 是否标准化：否
- 并行任务数：-1
- 训练样本数
- 测试样本数

## 6. 模型评估输出

每个模型输出：

- 混淆矩阵：
  - TN
  - FP
  - FN
  - TP
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- 分类报告
- ROC 曲线数据

模型对比输出：

- 指标对比表
- Accuracy / Precision / Recall / F1-score / AUC 柱状图
- 三个模型同图 ROC 曲线
- 三个模型混淆矩阵并排展示
- 决策树和随机森林的特征重要性图
- 三个模型的文字评价：
  - 哪个模型 AUC 最高
  - 哪个模型 F1-score 最高
  - 是否存在 Precision 高但 Recall 低的情况
  - 是否存在 Recall 高但 Precision 低的情况
  - 结合股票上涨预测场景给出简短评价，例如偏稳健、偏激进、综合表现较好

## 7. HTML 看板设计

HTML 看板包含以下区域。

### 7.1 顶部摘要

展示：

- 数据文件名
- 样本数量
- 特征数量
- 股票数量
- 日期范围
- 训练集比例
- 测试集比例
- 标签正负样本比例

### 7.2 数据质量检查

展示：

- 缺失值 Top 20
- 数据类型概览
- 重复记录检查
- 标签分布
- 入模因子清单
- 因子缺失值比例

### 7.3 训练/测试划分

展示：

- 训练集和测试集日期范围
- 训练集和测试集样本数
- 训练集和测试集标签分布
- 按时间划分说明：使用历史数据训练，未来时间段测试

### 7.4 模型参数

用中文字段展示三个模型的关键参数。

### 7.5 模型表现对比

展示：

- 指标对比表
- 指标对比柱状图
- ROC 曲线对比图
- AUC 排名
- 三个模型综合评价结论

ROC 曲线要求：

- 三个模型显示在同一张图
- Legend 可点击隐藏或显示模型曲线
- Hover 显示 FPR、TPR、模型名称、AUC

### 7.6 混淆矩阵

展示：

- 三个模型各自的混淆矩阵热力图
- 每个混淆矩阵旁显示 Accuracy、Precision、Recall、F1-score

### 7.7 特征重要性

展示：

- 决策树 Top 20 特征重要性
- 随机森林 Top 20 特征重要性
- 如逻辑回归可解释，则展示标准化后的系数 Top 正向/负向特征

### 7.8 三个模型评价

看板中需要对三个模型分别给出评价，评价内容包括：

- 逻辑回归：
  - 说明其作为线性模型，便于解释标准化系数方向
  - 结合 AUC、F1-score、Precision、Recall 判断是否适合作为基准模型
- 决策树：
  - 说明其能捕捉非线性关系，但单棵树可能不稳定
  - 结合混淆矩阵判断错判上涨和漏判上涨的情况
- 随机森林：
  - 说明其通过多棵树集成提升稳定性
  - 结合 AUC、F1-score 和特征重要性判断综合表现

同时输出一个总体结论：

- 最佳 AUC 模型
- 最佳 F1-score 模型
- 股票上涨预测场景下推荐优先参考的模型
- 主要风险提示：该任务是分类预测，不等同于可直接交易策略；需要进一步回测验证

## 8. 结果文件与中间文件

计划输出：

- `task5/task5_stock_classification.ipynb`
- `task5/task5_stock_classification_dashboard.html`
- `task5/outputs/model_metrics.csv`
- `task5/outputs/model_parameters.csv`
- `task5/outputs/data_quality_summary.csv`
- `task5/outputs/train_test_summary.csv`
- `task5/outputs/feature_summary.csv`

## 9. 验证方式

完成后会检查：

1. Notebook 能够从头运行到尾
2. HTML 文件可以打开
3. HTML 中包含数据质量、训练/测试划分、模型参数、模型指标、ROC、混淆矩阵
4. 训练/测试划分严格按照时间顺序
5. AUC 能正常计算
6. ROC 图中三个模型可通过 Legend 交互显示/隐藏
7. 看板中有对逻辑回归、决策树、随机森林三个模型的中文评价
8. 所有入模特征均来自 `model_data_stock.csv` 已有字段

## 10. 需要你确认的点

本 spec 已按你的修改要求调整为：

1. 完全使用 `model_data_stock.csv` 文件已有因子
2. 不处理文件中不存在的技术指标
3. 三个模型均显示较完整的关键默认超参数
4. 逻辑回归默认使用 `penalty="elasticnet"`、`solver="saga"`
5. 严格按 `Date` 时间顺序划分训练集和测试集
6. 看板中对三个模型分别做中文评价，并给出总体推荐

如果你同意，我将按这个 spec 生成 Notebook 和 HTML 看板。
