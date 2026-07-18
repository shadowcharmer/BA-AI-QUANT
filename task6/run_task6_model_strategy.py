from __future__ import annotations

import json
import math
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
DASHBOARD_PATH = BASE_DIR / "task6_model_strategy_dashboard.html"
NOTEBOOK_PATH = BASE_DIR / "task6_model_strategy.ipynb"

INITIAL_CAPITAL = 100000.0
TRAIN_RATIO = 0.70
RANDOM_STATE = 42

RISK_FREE_RATE_ANNUAL = 0.02
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005
SLIPPAGE_RATE = 0.0002

TOP_N = 3
BUY_PROB_THRESHOLD = 0.55
SELL_PROB_THRESHOLD = 0.45
TAKE_PROFIT = 0.20
STOP_LOSS = -0.10

ENABLE_RSI_FILTER = True
RSI_WINDOW = 4
RSI_MIN = 30
RSI_MAX = 80

ENABLE_TREND_FILTER = True
TREND_WINDOW = 4
TREND_MIN_RETURN = 0.00

LR_PARAMS = {
    "penalty": "elasticnet",
    "solver": "saga",
    "l1_ratio": 0.5,
    "C": 1.0,
    "max_iter": 5000,
    "tol": 0.0001,
    "fit_intercept": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

DT_PARAMS = {
    "criterion": "gini",
    "splitter": "best",
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 20,
    "max_features": None,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": 10,
    "min_samples_split": 2,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def to_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def fmt_num(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def date_str(value) -> str:
    if pd.isna(value):
        return "-"
    return pd.Timestamp(value).date().isoformat()


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, dtype={"Code": str})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df = df.sort_values(["Date", "Code"]).reset_index(drop=True)
    return df


def build_quality_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    missing = pd.DataFrame(
        {
            "字段": df.columns,
            "字段类型": [str(dtype) for dtype in df.dtypes],
            "缺失值数量": df.isna().sum().values,
            "缺失比例": df.isna().mean().values,
        }
    ).sort_values(["缺失值数量", "字段"], ascending=[False, True])

    next_ret = pd.to_numeric(df["Next_Ret"], errors="coerce")
    y = (next_ret > 0).astype("Int64").where(next_ret.notna())
    label_dist = (
        y.value_counts(dropna=False)
        .rename_axis("标签")
        .reset_index(name="样本数")
        .assign(占比=lambda x: x["样本数"] / len(df))
    )
    label_dist["标签含义"] = label_dist["标签"].map({0: "未上涨/下跌", 1: "上涨"}).fillna("缺失")

    numeric_df = df.drop(columns=["Date", "Code"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    arr = numeric_df.to_numpy(dtype=float)
    infinite_counts = pd.DataFrame(
        {
            "字段": numeric_df.columns,
            "无穷大数量": np.isinf(arr).sum(axis=0),
        }
    ).sort_values(["无穷大数量", "字段"], ascending=[False, True])

    overview = pd.DataFrame(
        [
            ["数据文件", DATA_PATH.name],
            ["样本行数", len(df)],
            ["字段数量", len(df.columns)],
            ["股票数量", df["Code"].nunique()],
            ["季度数量", df["Date"].nunique()],
            ["开始日期", date_str(df["Date"].min())],
            ["结束日期", date_str(df["Date"].max())],
            ["重复行数量", int(df.duplicated().sum())],
            ["重复 Date+Code 数量", int(df.duplicated(["Date", "Code"]).sum())],
            ["Next_Ret 缺失数量", int(next_ret.isna().sum())],
            ["Next_Ret 均值", float(next_ret.mean())],
            ["Next_Ret 中位数", float(next_ret.median())],
            ["上涨样本占比", float((next_ret > 0).mean())],
        ],
        columns=["项目", "值"],
    )
    desc = numeric_df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.reset_index()
    desc = desc.rename(columns={"index": "字段"})
    return {
        "overview": overview,
        "missing": missing,
        "label_dist": label_dist,
        "infinite_counts": infinite_counts,
        "describe": desc,
    }


def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    work = df.copy()
    work["Next_Ret"] = pd.to_numeric(work["Next_Ret"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work["Y"] = (work["Next_Ret"] > 0).astype("Int64").where(work["Next_Ret"].notna())

    base_cols = [c for c in work.columns if c not in {"Date", "Code", "Next_Ret", "Y"}]
    converted = pd.DataFrame(index=work.index)
    base_features = []
    for col in base_cols:
        values = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() > 0:
            converted[col] = values
            base_features.append(col)

    engineered = pd.DataFrame(index=work.index)
    if "MV" in converted.columns:
        engineered["log_mv"] = np.log(converted["MV"].where(converted["MV"] > 0))

    rank_features = []
    for col in base_features:
        rank_col = f"{col}_截面分位"
        engineered[rank_col] = converted.groupby(work["Date"])[col].rank(pct=True)
        rank_features.append(rank_col)

    ret_hist = work.groupby("Code", sort=False)["Next_Ret"].shift(1)
    engineered["ret_lag_1q"] = ret_hist
    engineered["ret_mean_4q"] = ret_hist.groupby(work["Code"], sort=False).transform(
        lambda s: s.rolling(4, min_periods=2).mean()
    )
    engineered["ret_vol_4q"] = ret_hist.groupby(work["Code"], sort=False).transform(
        lambda s: s.rolling(4, min_periods=2).std()
    )
    engineered["ret_cum_4q"] = ret_hist.groupby(work["Code"], sort=False).transform(
        lambda s: (1 + s).rolling(4, min_periods=2).apply(np.prod, raw=True) - 1
    )

    gain = ret_hist.clip(lower=0)
    loss = (-ret_hist.clip(upper=0))
    avg_gain = gain.groupby(work["Code"], sort=False).transform(
        lambda s: s.rolling(RSI_WINDOW, min_periods=2).mean()
    )
    avg_loss = loss.groupby(work["Code"], sort=False).transform(
        lambda s: s.rolling(RSI_WINDOW, min_periods=2).mean()
    )
    rs = avg_gain / avg_loss.replace(0, np.nan)
    engineered["RSI"] = 100 - 100 / (1 + rs)
    engineered.loc[(avg_loss == 0) & (avg_gain > 0), "RSI"] = 100
    engineered.loc[(avg_loss == 0) & (avg_gain == 0), "RSI"] = 50

    engineered["trend_return"] = ret_hist.groupby(work["Code"], sort=False).transform(
        lambda s: (1 + s).rolling(TREND_WINDOW, min_periods=2).apply(np.prod, raw=True) - 1
    )

    X_all = pd.concat([converted, engineered], axis=1)
    feature_names = list(X_all.columns)
    valid = work["Date"].notna() & work["Y"].notna()
    X = X_all.loc[valid, feature_names].copy()
    y = work.loc[valid, "Y"].astype(int)

    feature_types = []
    for name in feature_names:
        if name in base_features:
            feature_types.append("原始季度指标")
        elif name in rank_features:
            feature_types.append("季度截面分位衍生")
        elif name in {"RSI", "trend_return"}:
            feature_types.append("交易过滤衍生")
        else:
            feature_types.append("历史收益衍生")

    feature_summary = pd.DataFrame(
        {
            "入模特征": feature_names,
            "特征类型": feature_types,
            "缺失值数量": X.isna().sum().values,
            "缺失比例": X.isna().mean().values,
        }
    )
    meta = work.loc[valid, ["Date", "Code", "Next_Ret"]].copy()
    meta["RSI"] = engineered.loc[valid, "RSI"]
    meta["trend_return"] = engineered.loc[valid, "trend_return"]
    return X, y, feature_names, feature_summary, meta


def split_by_quarter(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame):
    dates = np.array(sorted(meta["Date"].dropna().unique()))
    cutoff = int(np.floor(len(dates) * TRAIN_RATIO))
    cutoff = min(max(cutoff, 1), len(dates) - 1)
    train_dates = set(dates[:cutoff])
    is_train = meta["Date"].isin(train_dates)
    X_train, X_test = X.loc[is_train].copy(), X.loc[~is_train].copy()
    y_train, y_test = y.loc[is_train].copy(), y.loc[~is_train].copy()
    meta_train, meta_test = meta.loc[is_train].copy(), meta.loc[~is_train].copy()
    summary = pd.DataFrame(
        [
            ["训练集比例参数", TRAIN_RATIO],
            ["训练集季度数", len(dates[:cutoff])],
            ["测试集季度数", len(dates[cutoff:])],
            ["训练集样本数", len(X_train)],
            ["测试集样本数", len(X_test)],
            ["实际训练集占比", len(X_train) / (len(X_train) + len(X_test))],
            ["训练集开始日期", date_str(meta_train["Date"].min())],
            ["训练集结束日期", date_str(meta_train["Date"].max())],
            ["测试集开始日期", date_str(meta_test["Date"].min())],
            ["测试集结束日期", date_str(meta_test["Date"].max())],
            ["训练集上涨样本占比", float(y_train.mean())],
            ["测试集上涨样本占比", float(y_test.mean())],
        ],
        columns=["项目", "值"],
    )
    return X_train, X_test, y_train, y_test, meta_train, meta_test, summary


def build_models(feature_names: list[str]) -> dict[str, Pipeline]:
    scaled = ColumnTransformer(
        [
            (
                "数值特征",
                Pipeline([("缺失值填充", SimpleImputer(strategy="median")), ("标准化", StandardScaler())]),
                feature_names,
            )
        ],
        remainder="drop",
    )
    tree_preprocess = ColumnTransformer(
        [("数值特征", SimpleImputer(strategy="median"), feature_names)],
        remainder="drop",
    )
    return {
        "逻辑回归": Pipeline([("预处理", scaled), ("模型", LogisticRegression(**LR_PARAMS))]),
        "决策树": Pipeline([("预处理", tree_preprocess), ("模型", DecisionTreeClassifier(**DT_PARAMS))]),
        "随机森林": Pipeline([("预处理", tree_preprocess), ("模型", RandomForestClassifier(**RF_PARAMS))]),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test, meta_test, feature_names):
    metrics = []
    roc_data = {}
    prediction_frames = []
    importances = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        auc = roc_auc_score(y_test, prob) if y_test.nunique() == 2 else np.nan
        fpr, tpr, thresholds = roc_curve(y_test, prob) if y_test.nunique() == 2 else ([], [], [])
        roc_data[name] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) if np.isfinite(x) else None for x in thresholds],
            "auc": None if pd.isna(auc) else float(auc),
        }
        metrics.append(
            {
                "模型": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-score": f1_score(y_test, pred, zero_division=0),
                "AUC": auc,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
                "训练样本数": len(X_train),
                "测试样本数": len(X_test),
            }
        )
        pred_df = meta_test.copy()
        pred_df["模型"] = name
        pred_df["实际标签"] = y_test.values
        pred_df["预测标签"] = pred
        pred_df["上涨概率"] = prob
        prediction_frames.append(pred_df)

        estimator = model.named_steps["模型"]
        if name == "逻辑回归":
            coefs = estimator.coef_[0]
            for feat, coef in zip(feature_names, coefs):
                importances.append({"模型": name, "特征": feat, "重要性或系数": float(coef), "类型": "标准化系数"})
        elif hasattr(estimator, "feature_importances_"):
            for feat, imp in zip(feature_names, estimator.feature_importances_):
                importances.append({"模型": name, "特征": feat, "重要性或系数": float(imp), "类型": "特征重要性"})

    return pd.DataFrame(metrics), roc_data, pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(importances)


def build_parameter_table() -> pd.DataFrame:
    rows = []
    rows.extend(
        [
            ["逻辑回归", "正则化方式", LR_PARAMS["penalty"]],
            ["逻辑回归", "L1 比例", LR_PARAMS["l1_ratio"]],
            ["逻辑回归", "正则化强度 C", LR_PARAMS["C"]],
            ["逻辑回归", "求解器", LR_PARAMS["solver"]],
            ["逻辑回归", "最大迭代次数", LR_PARAMS["max_iter"]],
            ["逻辑回归", "收敛阈值", LR_PARAMS["tol"]],
            ["逻辑回归", "类别权重", LR_PARAMS["class_weight"]],
            ["逻辑回归", "是否标准化", "是"],
            ["决策树", "划分标准", DT_PARAMS["criterion"]],
            ["决策树", "划分策略", DT_PARAMS["splitter"]],
            ["决策树", "最大深度", DT_PARAMS["max_depth"]],
            ["决策树", "内部节点再划分最小样本数", DT_PARAMS["min_samples_split"]],
            ["决策树", "叶子节点最小样本数", DT_PARAMS["min_samples_leaf"]],
            ["决策树", "最大特征数", DT_PARAMS["max_features"]],
            ["决策树", "类别权重", DT_PARAMS["class_weight"]],
            ["决策树", "是否标准化", "否"],
            ["随机森林", "树数量", RF_PARAMS["n_estimators"]],
            ["随机森林", "划分标准", RF_PARAMS["criterion"]],
            ["随机森林", "最大深度", RF_PARAMS["max_depth"]],
            ["随机森林", "内部节点再划分最小样本数", RF_PARAMS["min_samples_split"]],
            ["随机森林", "叶子节点最小样本数", RF_PARAMS["min_samples_leaf"]],
            ["随机森林", "最大特征数", RF_PARAMS["max_features"]],
            ["随机森林", "是否 Bootstrap 抽样", "是" if RF_PARAMS["bootstrap"] else "否"],
            ["随机森林", "类别权重", RF_PARAMS["class_weight"]],
            ["随机森林", "是否标准化", "否"],
            ["随机森林", "并行任务数", RF_PARAMS["n_jobs"]],
        ]
    )
    return pd.DataFrame(rows, columns=["模型", "参数", "默认值"])


def allocation(selected: pd.DataFrame, method: str) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    if method == "等权":
        return pd.Series(1 / len(selected), index=selected.index)
    probs = selected["上涨概率"].clip(lower=0)
    total = probs.sum()
    if total <= 0:
        return pd.Series(1 / len(selected), index=selected.index)
    return probs / total


def transaction_cost(prev_weights: dict[str, float], new_weights: dict[str, float]) -> tuple[float, float]:
    codes = set(prev_weights) | set(new_weights)
    buy_turnover = sum(max(new_weights.get(c, 0.0) - prev_weights.get(c, 0.0), 0.0) for c in codes)
    sell_turnover = sum(max(prev_weights.get(c, 0.0) - new_weights.get(c, 0.0), 0.0) for c in codes)
    cost = buy_turnover * (COMMISSION_RATE + SLIPPAGE_RATE) + sell_turnover * (
        COMMISSION_RATE + STAMP_TAX_RATE + SLIPPAGE_RATE
    )
    return buy_turnover + sell_turnover, cost


def run_model_strategy(predictions: pd.DataFrame, model_name: str, weight_method: str):
    capital = INITIAL_CAPITAL
    prev_weights: dict[str, float] = {}
    quarter_rows = []
    holdings_rows = []

    model_df = predictions[predictions["模型"] == model_name].sort_values(["Date", "上涨概率"], ascending=[True, False])
    for date, qdf in model_df.groupby("Date", sort=True):
        candidates = qdf[qdf["上涨概率"] >= BUY_PROB_THRESHOLD].copy()
        candidates = candidates[candidates["上涨概率"] > SELL_PROB_THRESHOLD]
        if ENABLE_RSI_FILTER:
            candidates = candidates[candidates["RSI"].isna() | candidates["RSI"].between(RSI_MIN, RSI_MAX)]
        if ENABLE_TREND_FILTER:
            candidates = candidates[candidates["trend_return"].isna() | (candidates["trend_return"] >= TREND_MIN_RETURN)]
        selected = candidates.sort_values("上涨概率", ascending=False).head(TOP_N).copy()
        weights = allocation(selected, weight_method)
        new_weights = {code: float(w) for code, w in zip(selected["Code"], weights)}
        turnover, cost_rate = transaction_cost(prev_weights, new_weights)

        adjusted_ret = selected["Next_Ret"].clip(lower=STOP_LOSS, upper=TAKE_PROFIT) if not selected.empty else pd.Series(dtype=float)
        gross_ret = float((weights.values * adjusted_ret.values).sum()) if not selected.empty else 0.0
        net_ret = gross_ret - cost_rate
        capital = capital * (1 + net_ret)
        draw_selected = ", ".join(selected["Code"].tolist()) if not selected.empty else "现金"

        quarter_rows.append(
            {
                "日期": date,
                "策略": f"{model_name}-{weight_method}",
                "模型": model_name,
                "仓位方式": weight_method,
                "持仓股票": draw_selected,
                "持仓数量": len(selected),
                "组合季度毛收益": gross_ret,
                "换手率": turnover,
                "交易成本率": cost_rate,
                "组合季度净收益": net_ret,
                "期末资金": capital,
            }
        )
        for idx, row in selected.iterrows():
            holdings_rows.append(
                {
                    "日期": date,
                    "策略": f"{model_name}-{weight_method}",
                    "模型": model_name,
                    "仓位方式": weight_method,
                    "股票代码": row["Code"],
                    "上涨概率": row["上涨概率"],
                    "权重": float(weights.loc[idx]),
                    "原始季度收益": row["Next_Ret"],
                    "止盈止损后收益": float(adjusted_ret.loc[idx]),
                    "RSI": row["RSI"],
                    "趋势收益": row["trend_return"],
                }
            )
        prev_weights = new_weights

    return pd.DataFrame(quarter_rows), pd.DataFrame(holdings_rows)


def run_benchmark(predictions: pd.DataFrame, name: str):
    base = predictions[predictions["模型"] == "逻辑回归"].copy()
    capital = INITIAL_CAPITAL
    prev_weights: dict[str, float] = {}
    rng = np.random.default_rng(RANDOM_STATE)
    quarter_rows = []
    holdings_rows = []

    for date, qdf in base.groupby("Date", sort=True):
        if name == "全市场等权基准":
            selected = qdf.copy()
        else:
            selected = qdf.sample(n=min(TOP_N, len(qdf)), random_state=int(rng.integers(0, 1_000_000))).copy()
        weight = 1 / len(selected) if len(selected) else 0
        new_weights = {code: float(weight) for code in selected["Code"]}
        turnover, cost_rate = transaction_cost(prev_weights, new_weights)
        gross_ret = float((selected["Next_Ret"].clip(lower=STOP_LOSS, upper=TAKE_PROFIT) * weight).sum()) if len(selected) else 0.0
        net_ret = gross_ret - cost_rate
        capital = capital * (1 + net_ret)
        quarter_rows.append(
            {
                "日期": date,
                "策略": name,
                "模型": "基准",
                "仓位方式": "等权",
                "持仓股票": "全市场" if name == "全市场等权基准" else ", ".join(selected["Code"].tolist()),
                "持仓数量": len(selected),
                "组合季度毛收益": gross_ret,
                "换手率": turnover,
                "交易成本率": cost_rate,
                "组合季度净收益": net_ret,
                "期末资金": capital,
            }
        )
        for _, row in selected.iterrows():
            holdings_rows.append(
                {
                    "日期": date,
                    "策略": name,
                    "模型": "基准",
                    "仓位方式": "等权",
                    "股票代码": row["Code"],
                    "上涨概率": np.nan,
                    "权重": weight,
                    "原始季度收益": row["Next_Ret"],
                    "止盈止损后收益": float(np.clip(row["Next_Ret"], STOP_LOSS, TAKE_PROFIT)),
                    "RSI": row["RSI"],
                    "趋势收益": row["trend_return"],
                }
            )
        prev_weights = new_weights
    return pd.DataFrame(quarter_rows), pd.DataFrame(holdings_rows)


def max_drawdown(nav: pd.Series) -> float:
    running_max = nav.cummax()
    dd = nav / running_max - 1
    return float(dd.min())


def compute_strategy_metrics(quarterly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, sdf in quarterly.groupby("策略"):
        sdf = sdf.sort_values("日期")
        returns = sdf["组合季度净收益"].astype(float)
        nav = sdf["期末资金"].astype(float) / INITIAL_CAPITAL
        n = len(returns)
        final_capital = float(sdf["期末资金"].iloc[-1])
        cumulative = final_capital / INITIAL_CAPITAL - 1
        annual = (1 + cumulative) ** (4 / n) - 1 if n > 0 and cumulative > -1 else np.nan
        vol = returns.std(ddof=1) * math.sqrt(4) if n > 1 else np.nan
        sharpe = (annual - RISK_FREE_RATE_ANNUAL) / vol if vol and vol > 0 else np.nan
        pos = returns[returns > 0]
        neg = returns[returns < 0]
        rows.append(
            {
                "策略": strategy,
                "初始资金": INITIAL_CAPITAL,
                "期末资金": final_capital,
                "累计收益率": cumulative,
                "年化收益率": annual,
                "年化波动率": vol,
                "夏普比率": sharpe,
                "最大回撤": max_drawdown(nav),
                "胜率": float((returns > 0).mean()),
                "盈亏比": float(pos.mean() / abs(neg.mean())) if len(pos) and len(neg) else np.nan,
                "最佳季度收益": float(returns.max()),
                "最差季度收益": float(returns.min()),
                "平均季度收益": float(returns.mean()),
                "季度收益标准差": float(returns.std(ddof=1)) if n > 1 else np.nan,
                "平均换手率": float(sdf["换手率"].mean()),
                "总交易成本": float((sdf["交易成本率"] * sdf["期末资金"].shift(1).fillna(INITIAL_CAPITAL)).sum()),
                "持仓季度数": n,
            }
        )
    return pd.DataFrame(rows).sort_values("期末资金", ascending=False)


def table_html(df: pd.DataFrame, max_rows: int = 20, pct_cols=None, num_cols=None) -> str:
    pct_cols = set(pct_cols or [])
    num_cols = set(num_cols or [])
    show = df.head(max_rows).copy()
    for col in show.columns:
        if col in pct_cols:
            show[col] = show[col].map(fmt_pct)
        elif col in num_cols:
            show[col] = show[col].map(fmt_num)
        elif pd.api.types.is_datetime64_any_dtype(show[col]):
            show[col] = show[col].map(date_str)
    return show.to_html(index=False, classes="data-table", border=0, escape=True)


def echarts_line_series(df: pd.DataFrame, value_col: str):
    series = []
    for name, sdf in df.groupby("策略"):
        sdf = sdf.sort_values("日期")
        series.append(
            {
                "name": name,
                "type": "line",
                "showSymbol": False,
                "data": [[date_str(r["日期"]), round(float(r[value_col]), 6)] for _, r in sdf.iterrows()],
            }
        )
    return series


def echarts_bar_metrics(metrics: pd.DataFrame):
    cols = ["累计收益率", "年化收益率", "夏普比率", "最大回撤"]
    return [
        {
            "name": col,
            "type": "bar",
            "data": [None if pd.isna(v) else round(float(v), 6) for v in metrics[col]],
        }
        for col in cols
    ]


def make_dashboard(context: dict):
    quality = context["quality"]
    model_metrics = context["model_metrics"]
    strategy_metrics = context["strategy_metrics"]
    quarterly = context["quarterly"]
    holdings = context["holdings"]
    roc_data = context["roc_data"]
    importances = context["importances"]
    feature_summary = context["feature_summary"]
    train_test_summary = context["train_test_summary"]
    params = context["params"]

    best_auc = model_metrics.sort_values("AUC", ascending=False).iloc[0]["模型"]
    best_f1 = model_metrics.sort_values("F1-score", ascending=False).iloc[0]["模型"]
    best_strategy = strategy_metrics.iloc[0]["策略"]
    best_sharpe = strategy_metrics.sort_values("夏普比率", ascending=False, na_position="last").iloc[0]["策略"]

    roc_series = []
    for model, values in roc_data.items():
        roc_series.append(
            {
                "name": f"{model} AUC={values['auc']:.4f}" if values["auc"] is not None else model,
                "type": "line",
                "showSymbol": False,
                "data": [[round(x, 6), round(y, 6)] for x, y in zip(values["fpr"], values["tpr"])],
            }
        )
    roc_series.append({"name": "随机参考线", "type": "line", "showSymbol": False, "data": [[0, 0], [1, 1]], "lineStyle": {"type": "dashed"}})

    nav_df = quarterly.copy()
    nav_df["净值"] = nav_df["期末资金"] / INITIAL_CAPITAL
    drawdown_parts = []
    for strategy, sdf in nav_df.groupby("策略"):
        sdf = sdf.sort_values("日期").copy()
        sdf["回撤"] = sdf["净值"] / sdf["净值"].cummax() - 1
        drawdown_parts.append(sdf)
    drawdown_df = pd.concat(drawdown_parts, ignore_index=True)

    model_bar_series = [
        {"name": metric, "type": "bar", "data": [round(float(v), 6) for v in model_metrics[metric]]}
        for metric in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    ]

    top_importance = (
        importances.assign(abs_value=lambda x: x["重要性或系数"].abs())
        .sort_values(["模型", "abs_value"], ascending=[True, False])
        .groupby("模型")
        .head(12)
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TASK6 模型概率因子交易策略看板</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f6f7f9; }}
    header {{ padding: 28px 32px; background: #16213e; color: #fff; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; font-weight: 600; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; font-weight: 600; }}
    h3 {{ margin: 18px 0 10px; font-size: 16px; font-weight: 600; }}
    main {{ padding: 24px 32px 48px; }}
    section {{ background: #fff; border: 1px solid #e4e7eb; border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #e4e7eb; border-radius: 8px; padding: 12px; background: #fbfcfd; }}
    .metric .label {{ color: #52606d; font-size: 13px; }}
    .metric .value {{ margin-top: 5px; font-size: 20px; font-weight: 600; }}
    .chart {{ height: 420px; margin-top: 12px; }}
    .chart-small {{ height: 340px; margin-top: 12px; }}
    .table-wrap {{ overflow-x: auto; }}
    table.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    table.data-table th, table.data-table td {{ border-bottom: 1px solid #e4e7eb; padding: 8px 10px; text-align: left; white-space: nowrap; }}
    table.data-table th {{ background: #f1f3f5; font-weight: 600; }}
    .note {{ color: #52606d; font-size: 13px; line-height: 1.6; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} main {{ padding: 16px; }} }}
    @media (max-width: 560px) {{ .grid {{ grid-template-columns: 1fr; }} header {{ padding: 20px; }} }}
  </style>
</head>
<body>
<header>
  <h1>TASK6 模型概率因子交易策略看板</h1>
  <div>逻辑回归、决策树、随机森林分类预测与季度换仓回测</div>
</header>
<main>
  <section>
    <h2>顶部摘要</h2>
    <div class="grid">
      <div class="metric"><div class="label">样本数量</div><div class="value">{fmt_num(context['sample_count'], 0)}</div></div>
      <div class="metric"><div class="label">股票数量</div><div class="value">{fmt_num(context['stock_count'], 0)}</div></div>
      <div class="metric"><div class="label">季度范围</div><div class="value">{context['date_range']}</div></div>
      <div class="metric"><div class="label">初始资金</div><div class="value">{fmt_num(INITIAL_CAPITAL, 0)}</div></div>
      <div class="metric"><div class="label">最佳 AUC 模型</div><div class="value">{escape(str(best_auc))}</div></div>
      <div class="metric"><div class="label">最佳 F1 模型</div><div class="value">{escape(str(best_f1))}</div></div>
      <div class="metric"><div class="label">期末资金最高策略</div><div class="value">{escape(str(best_strategy))}</div></div>
      <div class="metric"><div class="label">夏普最高策略</div><div class="value">{escape(str(best_sharpe))}</div></div>
    </div>
  </section>

  <section>
    <h2>数据质量检查</h2>
    <h3>数据概览</h3><div class="table-wrap">{table_html(quality['overview'], 30)}</div>
    <h3>缺失值 Top 20</h3><div class="table-wrap">{table_html(quality['missing'], 20, pct_cols=['缺失比例'])}</div>
    <h3>标签分布</h3><div class="table-wrap">{table_html(quality['label_dist'], 10, pct_cols=['占比'])}</div>
    <h3>无穷大检查</h3><div class="table-wrap">{table_html(quality['infinite_counts'], 20)}</div>
  </section>

  <section>
    <h2>特征工程与训练测试划分</h2>
    <p class="note">`Next_Ret` 仅用于生成标签和测试期真实收益，不进入模型特征。历史收益、RSI、趋势指标均使用滞后季度收益计算。</p>
    <h3>训练测试划分</h3><div class="table-wrap">{table_html(train_test_summary, 20, pct_cols=['值'])}</div>
    <h3>入模特征 Top 30</h3><div class="table-wrap">{table_html(feature_summary, 30, pct_cols=['缺失比例'])}</div>
  </section>

  <section>
    <h2>模型参数</h2>
    <div class="table-wrap">{table_html(params, 60)}</div>
  </section>

  <section>
    <h2>模型分类效果</h2>
    <div class="table-wrap">{table_html(model_metrics, 10, pct_cols=['Accuracy','Precision','Recall','F1-score'], num_cols=['AUC'])}</div>
    <div id="modelMetricChart" class="chart-small"></div>
    <div id="rocChart" class="chart"></div>
    <h3>混淆矩阵</h3><div class="table-wrap">{table_html(model_metrics[['模型','TN','FP','FN','TP','Accuracy','Precision','Recall','F1-score','AUC']], 10, pct_cols=['Accuracy','Precision','Recall','F1-score'], num_cols=['AUC'])}</div>
    <h3>系数与特征重要性 Top 12</h3><div class="table-wrap">{table_html(top_importance[['模型','特征','类型','重要性或系数']], 40, num_cols=['重要性或系数'])}</div>
  </section>

  <section>
    <h2>交易规则</h2>
    <div class="grid">
      <div class="metric"><div class="label">每季度持仓数</div><div class="value">{TOP_N}</div></div>
      <div class="metric"><div class="label">买入概率阈值</div><div class="value">{fmt_pct(BUY_PROB_THRESHOLD)}</div></div>
      <div class="metric"><div class="label">卖出概率阈值</div><div class="value">{fmt_pct(SELL_PROB_THRESHOLD)}</div></div>
      <div class="metric"><div class="label">止盈 / 止损</div><div class="value">{fmt_pct(TAKE_PROFIT)} / {fmt_pct(STOP_LOSS)}</div></div>
      <div class="metric"><div class="label">RSI 过滤</div><div class="value">{RSI_MIN}-{RSI_MAX}</div></div>
      <div class="metric"><div class="label">趋势窗口</div><div class="value">{TREND_WINDOW} 季度</div></div>
      <div class="metric"><div class="label">手续费率</div><div class="value">{fmt_pct(COMMISSION_RATE, 3)}</div></div>
      <div class="metric"><div class="label">印花税 / 滑点</div><div class="value">{fmt_pct(STAMP_TAX_RATE, 3)} / {fmt_pct(SLIPPAGE_RATE, 3)}</div></div>
    </div>
    <p class="note">当前数据为季度频率，止盈止损按单只股票季度收益近似触发，无法模拟真实盘中触发路径。</p>
  </section>

  <section>
    <h2>策略回测结果</h2>
    <div class="table-wrap">{table_html(strategy_metrics, 20, pct_cols=['累计收益率','年化收益率','年化波动率','最大回撤','胜率','最佳季度收益','最差季度收益','平均季度收益','季度收益标准差','平均换手率'], num_cols=['初始资金','期末资金','夏普比率','盈亏比','总交易成本'])}</div>
    <div id="strategyMetricChart" class="chart"></div>
    <div id="navChart" class="chart"></div>
    <div id="drawdownChart" class="chart"></div>
    <div id="quarterRetChart" class="chart"></div>
    <h3>季度选股明细 Top 60</h3><div class="table-wrap">{table_html(holdings.sort_values(['日期','策略']).head(60), 60, pct_cols=['上涨概率','权重','原始季度收益','止盈止损后收益','趋势收益'], num_cols=['RSI'])}</div>
  </section>

  <section>
    <h2>结论</h2>
    <p class="note">分类维度：AUC 最高的是 {escape(str(best_auc))}，F1-score 最高的是 {escape(str(best_f1))}。交易维度：期末资金最高的是 {escape(str(best_strategy))}，夏普比率最高的是 {escape(str(best_sharpe))}。分类效果和交易效果不一定一致，最终策略选择应优先结合净值、回撤、换手率和交易成本综合判断。</p>
  </section>
</main>
<script>
const modelNames = {to_json(model_metrics['模型'].tolist())};
const modelMetricSeries = {to_json(model_bar_series)};
const rocSeries = {to_json(roc_series)};
const strategyNames = {to_json(strategy_metrics['策略'].tolist())};
const strategyMetricSeries = {to_json(echarts_bar_metrics(strategy_metrics))};
const navSeries = {to_json(echarts_line_series(nav_df, '净值'))};
const drawdownSeries = {to_json(echarts_line_series(drawdown_df, '回撤'))};
const quarterRetSeries = {to_json(echarts_line_series(quarterly, '组合季度净收益'))};

function makeChart(id, option) {{
  const el = document.getElementById(id);
  const chart = echarts.init(el);
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
}}

makeChart('modelMetricChart', {{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ top: 0 }},
  grid: {{ left: 48, right: 24, top: 70, bottom: 50 }},
  xAxis: {{ type: 'category', data: modelNames }},
  yAxis: {{ type: 'value' }},
  series: modelMetricSeries
}});
makeChart('rocChart', {{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ top: 0, type: 'scroll' }},
  grid: {{ left: 48, right: 24, top: 70, bottom: 50 }},
  xAxis: {{ type: 'value', name: 'FPR', min: 0, max: 1 }},
  yAxis: {{ type: 'value', name: 'TPR', min: 0, max: 1 }},
  series: rocSeries
}});
makeChart('strategyMetricChart', {{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ top: 0 }},
  grid: {{ left: 60, right: 24, top: 70, bottom: 110 }},
  xAxis: {{ type: 'category', data: strategyNames, axisLabel: {{ rotate: 35 }} }},
  yAxis: {{ type: 'value' }},
  series: strategyMetricSeries
}});
makeChart('navChart', {{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ top: 0, type: 'scroll' }},
  grid: {{ left: 60, right: 24, top: 80, bottom: 50 }},
  xAxis: {{ type: 'time' }},
  yAxis: {{ type: 'value', name: '净值' }},
  series: navSeries
}});
makeChart('drawdownChart', {{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ top: 0, type: 'scroll' }},
  grid: {{ left: 60, right: 24, top: 80, bottom: 50 }},
  xAxis: {{ type: 'time' }},
  yAxis: {{ type: 'value', name: '回撤' }},
  series: drawdownSeries
}});
makeChart('quarterRetChart', {{
  tooltip: {{ trigger: 'axis' }},
  legend: {{ top: 0, type: 'scroll' }},
  grid: {{ left: 60, right: 24, top: 80, bottom: 50 }},
  xAxis: {{ type: 'time' }},
  yAxis: {{ type: 'value', name: '季度收益' }},
  series: quarterRetSeries
}});
</script>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def make_notebook():
    cells = []

    def md(text: str):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})

    def code(text: str):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)})

    md("# TASK6 模型概率因子交易策略\n\n本 Notebook 可分步骤执行：数据检查、特征工程、时间顺序划分、三个模型训练、分类评估、季度策略回测、结果对比。")
    code(
        """from pathlib import Path
import pandas as pd

BASE_DIR = Path.cwd()
if BASE_DIR.name != 'task6':
    BASE_DIR = BASE_DIR / 'task6'
DATA_PATH = BASE_DIR / 'model_data.csv'
OUTPUT_DIR = BASE_DIR / 'outputs'

INITIAL_CAPITAL = 100000
TRAIN_RATIO = 0.70
RISK_FREE_RATE_ANNUAL = 0.02
COMMISSION_RATE = 0.0003
STAMP_TAX_RATE = 0.0005
SLIPPAGE_RATE = 0.0002
BUY_PROB_THRESHOLD = 0.55
SELL_PROB_THRESHOLD = 0.45
TAKE_PROFIT = 0.20
STOP_LOSS = -0.10
RSI_WINDOW = 4
RSI_MIN = 30
RSI_MAX = 80
TREND_WINDOW = 4
TOP_N = 3

params = pd.DataFrame([
    ['初始资金', INITIAL_CAPITAL],
    ['训练集比例', TRAIN_RATIO],
    ['无风险年化收益率', RISK_FREE_RATE_ANNUAL],
    ['手续费率', COMMISSION_RATE],
    ['卖出印花税率', STAMP_TAX_RATE],
    ['滑点率', SLIPPAGE_RATE],
    ['买入概率阈值', BUY_PROB_THRESHOLD],
    ['卖出概率阈值', SELL_PROB_THRESHOLD],
    ['止盈阈值', TAKE_PROFIT],
    ['止损阈值', STOP_LOSS],
    ['RSI窗口', RSI_WINDOW],
    ['RSI下限', RSI_MIN],
    ['RSI上限', RSI_MAX],
    ['趋势窗口', TREND_WINDOW],
    ['每季度持仓股票数', TOP_N],
], columns=['参数', '默认值'])
params
"""
    )
    md("## Step 1：导入数据并检查质量")
    code(
        """df = pd.read_csv(DATA_PATH, dtype={'Code': str})
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Code'] = df['Code'].astype(str).str.zfill(6)
df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
display(df.head())
display(pd.DataFrame([
    ['样本行数', len(df)],
    ['字段数量', len(df.columns)],
    ['股票数量', df['Code'].nunique()],
    ['季度数量', df['Date'].nunique()],
    ['开始日期', df['Date'].min()],
    ['结束日期', df['Date'].max()],
    ['重复行数量', df.duplicated().sum()],
    ['重复 Date+Code 数量', df.duplicated(['Date', 'Code']).sum()],
], columns=['项目', '值']))
missing = pd.DataFrame({'字段': df.columns, '字段类型': df.dtypes.astype(str).values, '缺失数量': df.isna().sum().values, '缺失比例': df.isna().mean().values})
display(missing.sort_values('缺失数量', ascending=False).head(20))
"""
    )
    md("## Step 2：生成完整结果\n\n下面调用生成脚本完成特征工程、模型训练、策略回测，并刷新所有输出文件。")
    code(
        """import runpy
runpy.run_path(str(BASE_DIR / 'run_task6_model_strategy.py'), run_name='__main__')
"""
    )
    md("## Step 3：查看模型评价")
    code(
        """model_metrics = pd.read_csv(OUTPUT_DIR / 'model_metrics.csv')
display(model_metrics)
"""
    )
    md("## Step 4：查看策略核心指标")
    code(
        """strategy_metrics = pd.read_csv(OUTPUT_DIR / 'strategy_metrics.csv')
display(strategy_metrics)
"""
    )
    md("## Step 5：查看每季度收益率")
    code(
        """quarterly = pd.read_csv(OUTPUT_DIR / 'strategy_quarterly_returns.csv')
display(quarterly.head(30))
"""
    )
    md("## Step 6：查看每季度选股明细")
    code(
        """holdings = pd.read_csv(OUTPUT_DIR / 'strategy_holdings.csv')
display(holdings.head(60))
"""
    )
    md("## Step 7：打开 HTML 看板\n\n看板文件路径：`task6/task6_model_strategy_dashboard.html`。")
    code("print(BASE_DIR / 'task6_model_strategy_dashboard.html')")

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


def save_outputs(context: dict):
    OUTPUT_DIR.mkdir(exist_ok=True)
    context["quality"]["overview"].to_csv(OUTPUT_DIR / "data_overview.csv", index=False, encoding="utf-8-sig")
    context["quality"]["missing"].to_csv(OUTPUT_DIR / "data_quality_summary.csv", index=False, encoding="utf-8-sig")
    context["quality"]["label_dist"].to_csv(OUTPUT_DIR / "label_distribution.csv", index=False, encoding="utf-8-sig")
    context["feature_summary"].to_csv(OUTPUT_DIR / "feature_summary.csv", index=False, encoding="utf-8-sig")
    context["train_test_summary"].to_csv(OUTPUT_DIR / "train_test_summary.csv", index=False, encoding="utf-8-sig")
    context["params"].to_csv(OUTPUT_DIR / "model_parameters.csv", index=False, encoding="utf-8-sig")
    context["model_metrics"].to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")
    context["importances"].to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    context["quarterly"].to_csv(OUTPUT_DIR / "strategy_quarterly_returns.csv", index=False, encoding="utf-8-sig")
    context["holdings"].to_csv(OUTPUT_DIR / "strategy_holdings.csv", index=False, encoding="utf-8-sig")
    context["strategy_metrics"].to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False, encoding="utf-8-sig")


def main():
    df = load_data()
    quality = build_quality_tables(df)
    X, y, feature_names, feature_summary, meta = add_engineered_features(df)
    X_train, X_test, y_train, y_test, meta_train, meta_test, split_summary = split_by_quarter(X, y, meta)
    medians = X_train.median(numeric_only=True)
    feature_summary["训练集中位数填充值"] = feature_summary["入模特征"].map(medians)

    models = build_models(feature_names)
    model_metrics, roc_data, predictions, importances = evaluate_models(
        models, X_train, X_test, y_train, y_test, meta_test, feature_names
    )

    quarterly_parts = []
    holding_parts = []
    for model_name in ["逻辑回归", "决策树", "随机森林"]:
        for weight_method in ["等权", "概率加权"]:
            q, h = run_model_strategy(predictions, model_name, weight_method)
            quarterly_parts.append(q)
            holding_parts.append(h)
    for benchmark in ["全市场等权基准", "随机选3只基准"]:
        q, h = run_benchmark(predictions, benchmark)
        quarterly_parts.append(q)
        holding_parts.append(h)

    quarterly = pd.concat(quarterly_parts, ignore_index=True)
    holdings = pd.concat(holding_parts, ignore_index=True)
    strategy_metrics = compute_strategy_metrics(quarterly)
    params = build_parameter_table()

    context = {
        "quality": quality,
        "feature_summary": feature_summary,
        "train_test_summary": split_summary,
        "params": params,
        "model_metrics": model_metrics,
        "roc_data": roc_data,
        "predictions": predictions,
        "importances": importances,
        "quarterly": quarterly,
        "holdings": holdings,
        "strategy_metrics": strategy_metrics,
        "sample_count": len(df),
        "stock_count": df["Code"].nunique(),
        "date_range": f"{date_str(df['Date'].min())} 至 {date_str(df['Date'].max())}",
    }
    save_outputs(context)
    make_dashboard(context)
    make_notebook()
    print(f"Notebook: {NOTEBOOK_PATH}")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
