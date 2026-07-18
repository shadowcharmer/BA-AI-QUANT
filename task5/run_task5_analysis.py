from __future__ import annotations

import json
import math
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_data_stock.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
DASHBOARD_PATH = BASE_DIR / "task5_stock_classification_dashboard.html"
NOTEBOOK_PATH = BASE_DIR / "task5_stock_classification.ipynb"

TRAIN_RATIO = 0.70
RANDOM_STATE = 42

LR_PENALTY = "elasticnet"
LR_SOLVER = "saga"
LR_L1_RATIO = 0.5
LR_C = 1.0
LR_MAX_ITER = 5000
LR_TOL = 0.0001

DT_CRITERION = "gini"
DT_SPLITTER = "best"
DT_MAX_DEPTH = 10
DT_MIN_SAMPLES_SPLIT = 2
DT_MIN_SAMPLES_LEAF = 20
DT_MAX_FEATURES = None

RF_N_ESTIMATORS = 100
RF_CRITERION = "gini"
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 10
RF_MAX_FEATURES = "sqrt"
RF_BOOTSTRAP = True


def to_json(value):
    return json.dumps(value, ensure_ascii=False)


def pct(value, digits=2):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path, dtype={"Code": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)
    return df


def normalize_label(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    )
    if mapped.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = numeric.where(numeric.isin([0, 1]), mapped)
    return mapped.astype("Int64")


def build_quality_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing = pd.DataFrame(
        {
            "字段": df.columns,
            "缺失值数量": df.isna().sum().values,
            "缺失比例": df.isna().mean().values,
            "字段类型": [str(dtype) for dtype in df.dtypes],
        }
    ).sort_values(["缺失值数量", "字段"], ascending=[False, True])

    overview = pd.DataFrame(
        [
            ["数据文件", DATA_PATH.name],
            ["样本行数", len(df)],
            ["字段数量", len(df.columns)],
            ["股票数量", df["Code"].nunique()],
            ["日期数量", df["Date"].nunique()],
            ["开始日期", df["Date"].min().date().isoformat()],
            ["结束日期", df["Date"].max().date().isoformat()],
            ["重复行数量", int(df.duplicated().sum())],
            ["重复 Date+Code 数量", int(df.duplicated(["Date", "Code"]).sum())],
        ],
        columns=["项目", "值"],
    )

    label = normalize_label(df["Y"])
    label_dist = (
        label.value_counts(dropna=False)
        .rename_axis("标签")
        .reset_index(name="样本数")
        .assign(占比=lambda x: x["样本数"] / len(df))
    )
    label_dist["标签含义"] = label_dist["标签"].map({0: "未上涨/下跌", 1: "上涨"}).fillna("无效/缺失")

    numeric_df = df.drop(columns=["Date", "Code", "Y"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    infinite_counts = pd.DataFrame(
        {
            "字段": numeric_df.columns,
            "无穷大数量": np.isinf(numeric_df.to_numpy(dtype=float)).sum(axis=0),
        }
    ).sort_values(["无穷大数量", "字段"], ascending=[False, True])

    return {
        "overview": overview,
        "missing": missing,
        "label_dist": label_dist,
        "infinite_counts": infinite_counts,
    }


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    y = normalize_label(df["Y"])
    feature_df = df.drop(columns=["Date", "Code", "Y"], errors="ignore").copy()
    numeric_features = []
    converted = pd.DataFrame(index=df.index)
    for col in feature_df.columns:
        values = pd.to_numeric(feature_df[col], errors="coerce")
        if values.notna().sum() > 0:
            converted[col] = values.replace([np.inf, -np.inf], np.nan)
            numeric_features.append(col)

    valid = df["Date"].notna() & y.notna()
    X = converted.loc[valid, numeric_features].copy()
    y_valid = y.loc[valid].astype(int)
    feature_summary = pd.DataFrame(
        {
            "入模因子": numeric_features,
            "缺失值数量": X[numeric_features].isna().sum().values,
            "缺失比例": X[numeric_features].isna().mean().values,
        }
    )
    return X, y_valid, numeric_features, feature_summary


def split_by_time(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, train_ratio: float = TRAIN_RATIO):
    work = df.loc[X.index, ["Date", "Code"]].copy()
    dates = np.array(sorted(work["Date"].dropna().unique()))
    cutoff = int(np.floor(len(dates) * train_ratio))
    cutoff = min(max(cutoff, 1), len(dates) - 1)
    train_dates = set(dates[:cutoff])
    is_train = work["Date"].isin(train_dates)
    X_train, X_test = X.loc[is_train].copy(), X.loc[~is_train].copy()
    y_train, y_test = y.loc[is_train].copy(), y.loc[~is_train].copy()
    meta_train, meta_test = work.loc[is_train].copy(), work.loc[~is_train].copy()

    summary = pd.DataFrame(
        [
            ["训练集比例参数", train_ratio],
            ["训练集样本数", len(X_train)],
            ["测试集样本数", len(X_test)],
            ["实际训练集占比", len(X_train) / (len(X_train) + len(X_test))],
            ["训练集开始日期", meta_train["Date"].min().date().isoformat()],
            ["训练集结束日期", meta_train["Date"].max().date().isoformat()],
            ["测试集开始日期", meta_test["Date"].min().date().isoformat()],
            ["测试集结束日期", meta_test["Date"].max().date().isoformat()],
            ["训练集上涨样本占比", y_train.mean()],
            ["测试集上涨样本占比", y_test.mean()],
        ],
        columns=["项目", "值"],
    )
    return X_train, X_test, y_train, y_test, meta_train, meta_test, summary


def build_models(feature_names: list[str]) -> dict[str, Pipeline]:
    numeric_preprocess_scaled = ColumnTransformer(
        [
            (
                "数值特征",
                Pipeline(
                    [
                        ("缺失值填充", SimpleImputer(strategy="median")),
                        ("标准化", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ],
        remainder="drop",
    )
    numeric_preprocess = ColumnTransformer(
        [("数值特征", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )

    return {
        "逻辑回归": Pipeline(
            [
                ("预处理", numeric_preprocess_scaled),
                (
                    "模型",
                    LogisticRegression(
                        penalty=LR_PENALTY,
                        solver=LR_SOLVER,
                        l1_ratio=LR_L1_RATIO,
                        C=LR_C,
                        max_iter=LR_MAX_ITER,
                        tol=LR_TOL,
                        fit_intercept=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "决策树": Pipeline(
            [
                ("预处理", numeric_preprocess),
                (
                    "模型",
                    DecisionTreeClassifier(
                        criterion=DT_CRITERION,
                        splitter=DT_SPLITTER,
                        max_depth=DT_MAX_DEPTH,
                        min_samples_split=DT_MIN_SAMPLES_SPLIT,
                        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
                        max_features=DT_MAX_FEATURES,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "随机森林": Pipeline(
            [
                ("预处理", numeric_preprocess),
                (
                    "模型",
                    RandomForestClassifier(
                        n_estimators=RF_N_ESTIMATORS,
                        criterion=RF_CRITERION,
                        max_depth=RF_MAX_DEPTH,
                        min_samples_split=RF_MIN_SAMPLES_SPLIT,
                        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                        max_features=RF_MAX_FEATURES,
                        bootstrap=RF_BOOTSTRAP,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def model_parameters() -> pd.DataFrame:
    rows = [
        ["逻辑回归", "是否标准化", "是"],
        ["逻辑回归", "正则化方式 penalty", LR_PENALTY],
        ["逻辑回归", "求解器 solver", LR_SOLVER],
        ["逻辑回归", "L1 比例 l1_ratio", LR_L1_RATIO],
        ["逻辑回归", "正则化强度 C", LR_C],
        ["逻辑回归", "最大迭代次数 max_iter", LR_MAX_ITER],
        ["逻辑回归", "收敛阈值 tol", LR_TOL],
        ["逻辑回归", "是否拟合截距 fit_intercept", True],
        ["逻辑回归", "类别权重 class_weight", "balanced"],
        ["决策树", "是否标准化", "否"],
        ["决策树", "划分标准 criterion", DT_CRITERION],
        ["决策树", "划分策略 splitter", DT_SPLITTER],
        ["决策树", "最大深度 max_depth", DT_MAX_DEPTH],
        ["决策树", "内部节点再划分最小样本数 min_samples_split", DT_MIN_SAMPLES_SPLIT],
        ["决策树", "叶子节点最小样本数 min_samples_leaf", DT_MIN_SAMPLES_LEAF],
        ["决策树", "最大特征数 max_features", "None"],
        ["决策树", "类别权重 class_weight", "balanced"],
        ["随机森林", "是否标准化", "否"],
        ["随机森林", "树数量 n_estimators", RF_N_ESTIMATORS],
        ["随机森林", "划分标准 criterion", RF_CRITERION],
        ["随机森林", "最大深度 max_depth", RF_MAX_DEPTH],
        ["随机森林", "内部节点再划分最小样本数 min_samples_split", RF_MIN_SAMPLES_SPLIT],
        ["随机森林", "叶子节点最小样本数 min_samples_leaf", RF_MIN_SAMPLES_LEAF],
        ["随机森林", "最大特征数 max_features", RF_MAX_FEATURES],
        ["随机森林", "是否 Bootstrap 抽样", RF_BOOTSTRAP],
        ["随机森林", "类别权重 class_weight", "balanced"],
        ["随机森林", "并行任务数 n_jobs", -1],
    ]
    return pd.DataFrame(rows, columns=["模型", "参数", "默认值"])


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names):
    models = build_models(feature_names)
    metrics = []
    roc_data = {}
    confusion_data = {}
    feature_importance = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        auc = roc_auc_score(y_test, prob) if y_test.nunique() == 2 else np.nan
        fpr, tpr, thresholds = roc_curve(y_test, prob) if y_test.nunique() == 2 else ([], [], [])

        metrics.append(
            {
                "模型": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-score": f1_score(y_test, pred, zero_division=0),
                "AUC": auc,
                "TN": int(cm[0, 0]),
                "FP": int(cm[0, 1]),
                "FN": int(cm[1, 0]),
                "TP": int(cm[1, 1]),
            }
        )
        roc_data[name] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in thresholds],
            "auc": float(auc) if not np.isnan(auc) else None,
        }
        confusion_data[name] = cm.tolist()

        estimator = pipeline.named_steps["模型"]
        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
            feature_importance[name] = (
                pd.DataFrame({"因子": feature_names, "重要性": values})
                .sort_values("重要性", ascending=False)
                .head(20)
            )
        elif hasattr(estimator, "coef_"):
            values = estimator.coef_[0]
            feature_importance[name] = (
                pd.DataFrame({"因子": feature_names, "系数": values, "系数绝对值": np.abs(values)})
                .sort_values("系数绝对值", ascending=False)
                .head(20)
            )

    return pd.DataFrame(metrics).sort_values("AUC", ascending=False), roc_data, confusion_data, feature_importance, models


def evaluate_models_text(metrics_df: pd.DataFrame) -> dict[str, str]:
    best_auc = metrics_df.sort_values("AUC", ascending=False).iloc[0]
    best_f1 = metrics_df.sort_values("F1-score", ascending=False).iloc[0]
    comments = {}
    for _, row in metrics_df.iterrows():
        name = row["模型"]
        if name == "逻辑回归":
            base = "线性基准模型，可通过标准化系数解释因子方向。"
        elif name == "决策树":
            base = "单棵树能捕捉非线性关系，但稳定性通常弱于集成模型。"
        else:
            base = "多棵树集成模型，通常比单棵树更稳定，并可输出因子重要性。"
        balance = "Precision 和 Recall 较均衡。"
        if row["Precision"] - row["Recall"] > 0.08:
            balance = "Precision 明显高于 Recall，预测为上涨时更谨慎，但可能漏掉一部分上涨样本。"
        elif row["Recall"] - row["Precision"] > 0.08:
            balance = "Recall 明显高于 Precision，覆盖上涨样本更积极，但误判上涨的风险更高。"
        comments[name] = (
            f"{base} 测试集 AUC={fmt(row['AUC'])}，F1-score={fmt(row['F1-score'])}，"
            f"Accuracy={fmt(row['Accuracy'])}。{balance}"
        )

    comments["总体结论"] = (
        f"按 AUC 排名，当前最佳模型是 {best_auc['模型']}（AUC={fmt(best_auc['AUC'])}）；"
        f"按 F1-score 排名，当前最佳模型是 {best_f1['模型']}（F1-score={fmt(best_f1['F1-score'])}）。"
        "股票上涨预测仍需结合回测验证，本结果不能直接等同于可交易策略。"
    )
    return comments


def table_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    return view.to_html(index=False, classes="data-table", border=0, escape=False)


def chart_div(chart_id: str, title: str) -> str:
    return f"""
    <section class="panel">
      <div class="panel-title">{escape(title)}</div>
      <div id="{chart_id}" class="chart"></div>
    </section>
    """


def generate_dashboard(analysis: dict) -> None:
    metrics_df = analysis["metrics"]
    quality = analysis["quality"]
    split_summary = analysis["split_summary"]
    feature_summary = analysis["feature_summary"]
    params = analysis["parameters"]
    roc_data = analysis["roc_data"]
    confusion_data = analysis["confusion_data"]
    feature_importance = analysis["feature_importance"]
    comments = analysis["comments"]
    df = analysis["df"]

    metrics_records = metrics_df.to_dict("records")
    metric_names = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    metrics_chart = {
        "models": metrics_df["模型"].tolist(),
        "metrics": metric_names,
        "series": [
            {"name": metric, "type": "bar", "data": [round(float(v), 4) for v in metrics_df[metric]]}
            for metric in metric_names
        ],
    }
    roc_series = [
        {
            "name": f"{name} AUC={fmt(data['auc'])}",
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "data": [[round(x, 5), round(y, 5)] for x, y in zip(data["fpr"], data["tpr"])],
        }
        for name, data in roc_data.items()
    ]
    train_test_rows = dict(zip(split_summary["项目"], split_summary["值"]))
    split_chart = [
        {"name": "训练集", "value": int(train_test_rows["训练集样本数"])},
        {"name": "测试集", "value": int(train_test_rows["测试集样本数"])},
    ]
    label_dist = quality["label_dist"].copy()
    label_chart = [
        {"name": str(row["标签含义"]), "value": int(row["样本数"])}
        for _, row in label_dist.iterrows()
    ]

    confusion_charts = {}
    for name, cm in confusion_data.items():
        confusion_charts[name] = [
            [0, 0, int(cm[0][0])],
            [1, 0, int(cm[0][1])],
            [0, 1, int(cm[1][0])],
            [1, 1, int(cm[1][1])],
        ]

    fi_payload = {}
    for name, fi in feature_importance.items():
        value_col = "重要性" if "重要性" in fi.columns else "系数"
        ordered = fi.sort_values(value_col, ascending=True)
        fi_payload[name] = {
            "factors": ordered["因子"].tolist(),
            "values": [round(float(x), 6) for x in ordered[value_col]],
        }

    summary_cards = [
        ("样本数量", f"{len(df):,}"),
        ("入模因子", f"{len(analysis['feature_names'])}"),
        ("股票数量", f"{df['Code'].nunique():,}"),
        ("日期范围", f"{df['Date'].min().date()} 至 {df['Date'].max().date()}"),
        ("训练集", f"{int(train_test_rows['训练集样本数']):,}"),
        ("测试集", f"{int(train_test_rows['测试集样本数']):,}"),
    ]
    cards_html = "".join(
        f"<div class='metric-card'><span>{escape(k)}</span><strong>{escape(v)}</strong></div>"
        for k, v in summary_cards
    )

    comment_html = "".join(
        f"<article class='comment-card'><h3>{escape(k)}</h3><p>{escape(v)}</p></article>"
        for k, v in comments.items()
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TASK5 股票涨跌分类模型看板</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #18212f;
      --muted: #687284;
      --line: #dde3ee;
      --blue: #2563eb;
      --green: #059669;
      --orange: #d97706;
      --red: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.55;
    }}
    header {{
      padding: 28px 36px 20px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); margin: 0; }}
    main {{ padding: 24px 36px 44px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .metric-card, .panel, .comment-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(18, 32, 57, 0.06);
    }}
    .metric-card {{ padding: 14px 16px; min-height: 88px; }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .metric-card strong {{ font-size: 20px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}
    .grid-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}
    .panel {{ padding: 18px; margin-bottom: 16px; overflow: hidden; }}
    .panel-title {{ font-size: 17px; font-weight: 700; margin-bottom: 12px; }}
    .chart {{ width: 100%; height: 390px; }}
    .chart.small {{ height: 300px; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: #fff;
    }}
    .data-table th {{
      text-align: left;
      background: #eef2f7;
      color: #253142;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}
    .data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid #edf0f5;
      vertical-align: top;
    }}
    .table-wrap {{ overflow-x: auto; }}
    .comments {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 16px; }}
    .comment-card {{ padding: 16px; }}
    .comment-card h3 {{ margin: 0 0 8px; font-size: 16px; }}
    .comment-card p {{ margin: 0; color: #3f4b5f; }}
    @media (max-width: 1100px) {{
      .cards {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .grid-3 {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 760px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .cards, .grid-2, .comments {{ grid-template-columns: 1fr; }}
      .chart {{ height: 330px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TASK5 股票涨跌分类模型看板</h1>
    <p class="sub">严格按 Date 时间顺序划分训练集/测试集；所有入模特征均来自 model_data_stock.csv 已有因子。</p>
  </header>
  <main>
    <section class="cards">{cards_html}</section>

    <section class="grid-2">
      {chart_div("splitChart", "训练集 / 测试集样本划分")}
      {chart_div("labelChart", "标签分布")}
    </section>

    <section class="panel">
      <div class="panel-title">模型指标对比</div>
      <div id="metricChart" class="chart"></div>
      <div class="table-wrap">{table_html(metrics_df)}</div>
    </section>

    <section class="panel">
      <div class="panel-title">ROC 曲线与 AUC 对比</div>
      <div id="rocChart" class="chart"></div>
    </section>

    <section class="comments">{comment_html}</section>

    <section class="grid-3">
      {chart_div("cm_逻辑回归", "逻辑回归混淆矩阵")}
      {chart_div("cm_决策树", "决策树混淆矩阵")}
      {chart_div("cm_随机森林", "随机森林混淆矩阵")}
    </section>

    <section class="grid-3">
      {chart_div("fi_逻辑回归", "逻辑回归系数 Top 20")}
      {chart_div("fi_决策树", "决策树特征重要性 Top 20")}
      {chart_div("fi_随机森林", "随机森林特征重要性 Top 20")}
    </section>

    <section class="grid-2">
      <section class="panel">
        <div class="panel-title">训练 / 测试时间顺序划分</div>
        <div class="table-wrap">{table_html(split_summary)}</div>
      </section>
      <section class="panel">
        <div class="panel-title">模型关键超参数</div>
        <div class="table-wrap">{table_html(params)}</div>
      </section>
    </section>

    <section class="grid-2">
      <section class="panel">
        <div class="panel-title">数据质量概览</div>
        <div class="table-wrap">{table_html(quality["overview"])}</div>
      </section>
      <section class="panel">
        <div class="panel-title">缺失值 Top 20</div>
        <div class="table-wrap">{table_html(quality["missing"], 20)}</div>
      </section>
    </section>

    <section class="panel">
      <div class="panel-title">入模因子清单</div>
      <div class="table-wrap">{table_html(feature_summary)}</div>
    </section>
  </main>

  <script>
    const metricsChart = {to_json(metrics_chart)};
    const rocSeries = {to_json(roc_series)};
    const splitData = {to_json(split_chart)};
    const labelData = {to_json(label_chart)};
    const confusionCharts = {to_json(confusion_charts)};
    const fiPayload = {to_json(fi_payload)};

    const palette = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed"];

    function initChart(id, option) {{
      const dom = document.getElementById(id);
      if (!dom || !window.echarts) return;
      const chart = echarts.init(dom);
      chart.setOption(option);
      window.addEventListener("resize", () => chart.resize());
    }}

    initChart("splitChart", {{
      color: ["#2563eb", "#d97706"],
      tooltip: {{ trigger: "item" }},
      legend: {{ bottom: 0 }},
      series: [{{ type: "pie", radius: ["48%", "72%"], center: ["50%", "45%"], data: splitData }}]
    }});

    initChart("labelChart", {{
      color: ["#059669", "#dc2626", "#687284"],
      tooltip: {{ trigger: "item" }},
      legend: {{ bottom: 0 }},
      series: [{{ type: "pie", radius: "68%", center: ["50%", "45%"], data: labelData }}]
    }});

    initChart("metricChart", {{
      color: palette,
      tooltip: {{ trigger: "axis", axisPointer: {{ type: "shadow" }} }},
      legend: {{ top: 0 }},
      grid: {{ left: 44, right: 20, top: 48, bottom: 40 }},
      xAxis: {{ type: "category", data: metricsChart.models }},
      yAxis: {{ type: "value", min: 0, max: 1 }},
      series: metricsChart.series
    }});

    initChart("rocChart", {{
      color: palette,
      tooltip: {{
        trigger: "axis",
        formatter: function(params) {{
          return params.map(p => `${{p.seriesName}}<br>FPR: ${{p.data[0]}}，TPR: ${{p.data[1]}}`).join("<br><br>");
        }}
      }},
      legend: {{ top: 0, type: "scroll" }},
      grid: {{ left: 52, right: 24, top: 52, bottom: 48 }},
      xAxis: {{ type: "value", name: "FPR", min: 0, max: 1 }},
      yAxis: {{ type: "value", name: "TPR", min: 0, max: 1 }},
      series: [
        {{ name: "随机猜测", type: "line", data: [[0, 0], [1, 1]], lineStyle: {{ type: "dashed", color: "#94a3b8" }}, showSymbol: false }},
        ...rocSeries
      ]
    }});

    Object.entries(confusionCharts).forEach(([name, data]) => {{
      initChart(`cm_${{name}}`, {{
        tooltip: {{ position: "top" }},
        grid: {{ left: 52, right: 16, top: 16, bottom: 48 }},
        xAxis: {{ type: "category", data: ["预测未上涨", "预测上涨"], splitArea: {{ show: true }} }},
        yAxis: {{ type: "category", data: ["实际未上涨", "实际上涨"], splitArea: {{ show: true }} }},
        visualMap: {{ min: 0, max: Math.max(...data.map(d => d[2])), calculable: true, orient: "horizontal", left: "center", bottom: 0, inRange: {{ color: ["#eef2ff", "#2563eb"] }} }},
        series: [{{ type: "heatmap", data: data, label: {{ show: true, fontWeight: 700 }} }}]
      }});
    }});

    Object.entries(fiPayload).forEach(([name, payload]) => {{
      initChart(`fi_${{name}}`, {{
        color: [name === "逻辑回归" ? "#7c3aed" : "#059669"],
        tooltip: {{ trigger: "axis", axisPointer: {{ type: "shadow" }} }},
        grid: {{ left: 150, right: 18, top: 14, bottom: 28 }},
        xAxis: {{ type: "value" }},
        yAxis: {{ type: "category", data: payload.factors, axisLabel: {{ width: 135, overflow: "truncate" }} }},
        series: [{{ type: "bar", data: payload.values, barWidth: 12 }}]
      }});
    }});
  </script>
</body>
</html>"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def write_outputs(analysis: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    analysis["metrics"].to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    analysis["parameters"].to_csv(OUTPUT_DIR / "model_parameters.csv", index=False, encoding="utf-8-sig")
    analysis["feature_summary"].to_csv(OUTPUT_DIR / "feature_summary.csv", index=False, encoding="utf-8-sig")
    analysis["split_summary"].to_csv(OUTPUT_DIR / "train_test_summary.csv", index=False, encoding="utf-8-sig")
    analysis["quality"]["missing"].to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig")


def make_cell(cell_type: str, source: str) -> dict:
    cell = {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def generate_notebook() -> None:
    cells = [
        make_cell(
            "markdown",
            "# TASK5 股票涨跌分类模型\n\n"
            "本 Notebook 使用 `model_data_stock.csv` 文件已有因子，严格按 `Date` 时间顺序划分训练集和测试集，训练逻辑回归、决策树、随机森林三个分类模型，并查看 AUC、ROC、混淆矩阵等结果。",
        ),
        make_cell(
            "markdown",
            "## Step 1：导入依赖与参数配置\n\n"
            "逻辑回归默认使用 `penalty='elasticnet'`、`solver='saga'`；随机森林默认 `max_depth=10`、`n_estimators=100`。",
        ),
        make_cell(
            "code",
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import display, HTML, IFrame\n"
            "NOTEBOOK_DIR = Path.cwd()\n"
            f"if not (NOTEBOOK_DIR / 'run_task5_analysis.py').exists():\n"
            f"    NOTEBOOK_DIR = Path({str(BASE_DIR)!r})\n"
            "sys.path.insert(0, str(NOTEBOOK_DIR))\n"
            "import run_task5_analysis as task5\n\n"
            "task5.TRAIN_RATIO = 0.70\n"
            "task5.RANDOM_STATE = 42\n"
            "task5.LR_PENALTY = 'elasticnet'\n"
            "task5.LR_SOLVER = 'saga'\n"
            "task5.LR_L1_RATIO = 0.5\n"
            "task5.LR_C = 1.0\n"
            "task5.LR_MAX_ITER = 5000\n"
            "task5.DT_MAX_DEPTH = 10\n"
            "task5.RF_MAX_DEPTH = 10\n"
            "task5.RF_N_ESTIMATORS = 100",
        ),
        make_cell(
            "markdown",
            "## Step 2：加载数据\n\n读取 CSV，解析日期，并按 `Date`、`Code` 排序。",
        ),
        make_cell(
            "code",
            "df = task5.load_data()\n"
            "display(df.head())\n"
            "print('数据规模:', df.shape)\n"
            "print('日期范围:', df['Date'].min().date(), '至', df['Date'].max().date())\n"
            "print('股票数量:', df['Code'].nunique())",
        ),
        make_cell(
            "markdown",
            "## Step 3：数据质量检查\n\n检查字段类型、缺失值、重复记录、标签分布和无穷大值。",
        ),
        make_cell(
            "code",
            "quality = task5.build_quality_tables(df)\n"
            "display(quality['overview'])\n"
            "display(quality['label_dist'])\n"
            "display(quality['missing'].head(20))\n"
            "display(quality['infinite_counts'].head(20))",
        ),
        make_cell(
            "markdown",
            "## Step 4：特征工程\n\n完全使用文件已有因子：排除 `Date`、`Code`、`Y` 后，其余可转换为数值型的列进入模型。",
        ),
        make_cell(
            "code",
            "X, y, feature_names, feature_summary = task5.prepare_features(df)\n"
            "print('入模因子数量:', len(feature_names))\n"
            "display(feature_summary)\n"
            "display(X.head())",
        ),
        make_cell(
            "markdown",
            "## Step 5：严格按时间顺序划分训练集/测试集\n\n前 70% 日期作为训练集，后 30% 日期作为测试集。同一个日期的全部股票样本放在同一侧，避免未来数据泄漏。",
        ),
        make_cell(
            "code",
            "X_train, X_test, y_train, y_test, meta_train, meta_test, split_summary = task5.split_by_time(df, X, y, task5.TRAIN_RATIO)\n"
            "display(split_summary)\n"
            "display(pd.DataFrame({'训练集标签分布': y_train.value_counts(normalize=True), '测试集标签分布': y_test.value_counts(normalize=True)}))",
        ),
        make_cell(
            "markdown",
            "## Step 6：训练三个模型\n\n训练步骤：缺失值只用训练集中位数填充；逻辑回归额外做标准化；决策树和随机森林不做标准化。",
        ),
        make_cell(
            "code",
            "metrics, roc_data, confusion_data, feature_importance, models = task5.train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)\n"
            "parameters = task5.model_parameters()\n"
            "display(parameters)\n"
            "display(metrics)",
        ),
        make_cell(
            "markdown",
            "## Step 7：查看模型评价\n\n根据 AUC、F1-score、Precision、Recall 和混淆矩阵，对三个模型分别评价。",
        ),
        make_cell(
            "code",
            "comments = task5.evaluate_models_text(metrics)\n"
            "for name, comment in comments.items():\n"
            "    display(HTML(f'<h3>{name}</h3><p>{comment}</p>'))",
        ),
        make_cell(
            "markdown",
            "## Step 8：生成 ECharts HTML 看板\n\n看板包含可交互 ROC 曲线、指标对比、混淆矩阵、特征重要性和模型评价。",
        ),
        make_cell(
            "code",
            "analysis = {\n"
            "    'df': df,\n"
            "    'quality': quality,\n"
            "    'feature_names': feature_names,\n"
            "    'feature_summary': feature_summary,\n"
            "    'split_summary': split_summary,\n"
            "    'metrics': metrics,\n"
            "    'roc_data': roc_data,\n"
            "    'confusion_data': confusion_data,\n"
            "    'feature_importance': feature_importance,\n"
            "    'parameters': parameters,\n"
            "    'comments': comments,\n"
            "}\n"
            "task5.write_outputs(analysis)\n"
            "task5.generate_dashboard(analysis)\n"
            "print('HTML 看板已生成:', task5.DASHBOARD_PATH)\n"
            "IFrame(src=str(task5.DASHBOARD_PATH), width='100%', height=820)",
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def run_analysis() -> dict:
    df = load_data()
    quality = build_quality_tables(df)
    X, y, feature_names, feature_summary = prepare_features(df)
    X_train, X_test, y_train, y_test, meta_train, meta_test, split_summary = split_by_time(df, X, y)
    metrics, roc_data, confusion_data, feature_importance, models = train_and_evaluate(
        X_train, X_test, y_train, y_test, feature_names
    )
    parameters = model_parameters()
    comments = evaluate_models_text(metrics)
    analysis = {
        "df": df,
        "quality": quality,
        "feature_names": feature_names,
        "feature_summary": feature_summary,
        "split_summary": split_summary,
        "metrics": metrics,
        "roc_data": roc_data,
        "confusion_data": confusion_data,
        "feature_importance": feature_importance,
        "parameters": parameters,
        "comments": comments,
        "models": models,
    }
    write_outputs(analysis)
    generate_dashboard(analysis)
    generate_notebook()
    return analysis


if __name__ == "__main__":
    result = run_analysis()
    print("生成完成")
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"HTML 看板: {DASHBOARD_PATH}")
    print(result["metrics"].to_string(index=False))
