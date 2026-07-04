from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TASK1 = ROOT / "task1"
OUT = ROOT / "task2" / "question1"

DATASETS = {
    "A股": {
        "key": "a",
        "path": TASK1 / "smic_a_daily.csv",
        "code": "688981.SH",
        "market_note": "A股以人民币交易，涨跌停制度和成交量单位与港股不同。",
    },
    "港股": {
        "key": "hk",
        "path": TASK1 / "smic_hk_daily.csv",
        "code": "00981.HK",
        "market_note": "港股以港币交易，交易机制、投资者结构和成交量口径与A股不同。",
    },
}

RAW_NUMERIC_COLS = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]

DERIVED_COLS = [
    "log_return",
    "amplitude_pct",
    "intraday_return_pct",
    "gap_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "vwap_proxy",
    "close_ma5_gap_pct",
    "close_ma20_gap_pct",
    "pct_chg_5d",
    "pct_chg_20d",
    "volatility_20d",
    "amount_ma20_ratio",
]

QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

FIELD_ZH = {
    "ts_code": "证券代码",
    "trade_date": "交易日期",
    "open": "开盘价",
    "high": "最高价",
    "low": "最低价",
    "close": "收盘价",
    "pre_close": "前收盘价",
    "change": "涨跌额",
    "pct_chg": "日涨跌幅(%)",
    "vol": "成交量",
    "amount": "成交额",
    "log_return": "对数收益率",
    "amplitude_pct": "日内振幅(%)",
    "intraday_return_pct": "日内收益率(%)",
    "gap_pct": "开盘跳空幅度(%)",
    "upper_shadow_pct": "上影线比例(%)",
    "lower_shadow_pct": "下影线比例(%)",
    "vwap_proxy": "近似成交均价",
    "close_ma5": "5日收盘均价",
    "close_ma20": "20日收盘均价",
    "close_ma5_gap_pct": "收盘价相对5日均线偏离(%)",
    "close_ma20_gap_pct": "收盘价相对20日均线偏离(%)",
    "pct_chg_5d": "5日收益率(%)",
    "pct_chg_20d": "20日收益率(%)",
    "volatility_20d": "20日滚动波动率",
    "amount_ma20": "20日平均成交额",
    "amount_ma20_ratio": "成交额相对20日均值倍数",
}

FIELD_EXPLANATION = {
    "ts_code": "证券在数据源中的代码，用于区分A股和港股标的。",
    "trade_date": "交易发生日期，排除了非交易日。",
    "open": "当日第一笔或开盘集合竞价形成的价格。",
    "high": "当日最高成交价格。",
    "low": "当日最低成交价格。",
    "close": "当日收盘价格，常用于收益率、趋势和回测计算。",
    "pre_close": "上一交易日收盘价，用于计算涨跌额和涨跌幅。",
    "change": "当日收盘价相对前收盘价的价格变化。",
    "pct_chg": "当日涨跌额相对前收盘价的百分比变化，是短期波动和收益分析的核心字段。",
    "vol": "当日成交量，反映交易活跃度；A股和港股单位口径不同，不宜直接比较绝对值。",
    "amount": "当日成交额，反映资金参与强度和流动性。",
    "log_return": "用对数方式衡量单日收益，便于时间序列累加和波动分析。",
    "amplitude_pct": "最高价和最低价相对前收盘价的区间波动，衡量日内交易冲击。",
    "intraday_return_pct": "收盘价相对开盘价的变化，区分盘中方向和隔夜跳空影响。",
    "gap_pct": "开盘价相对前收盘价的变化，衡量隔夜信息冲击。",
    "upper_shadow_pct": "最高价高于开盘价和收盘价中较高者的部分，反映上冲后回落压力。",
    "lower_shadow_pct": "开盘价和收盘价中较低者高于最低价的部分，反映下探后承接力量。",
    "vwap_proxy": "用成交额除以成交量近似估计成交均价，仅用于粗略观察。",
    "close_ma5": "近5个交易日收盘价均值，反映短期价格中枢。",
    "close_ma20": "近20个交易日收盘价均值，反映月度价格中枢。",
    "close_ma5_gap_pct": "收盘价相对短期均线的位置，用于判断短期超买或超卖。",
    "close_ma20_gap_pct": "收盘价相对月度均线的位置，用于判断中期趋势强弱。",
    "pct_chg_5d": "5个交易日累计收益，近似一周动量。",
    "pct_chg_20d": "20个交易日累计收益，近似一个月动量。",
    "volatility_20d": "近20个交易日日涨跌幅标准差，衡量近期风险水平。",
    "amount_ma20": "近20个交易日成交额均值，作为成交活跃度基准。",
    "amount_ma20_ratio": "当日成交额相对20日均值的倍数，用于识别放量或缩量状态。",
}

STAT_ZH = {
    "count": "有效样本数",
    "missing_count": "缺失值数量",
    "missing_ratio": "缺失值比例",
    "mean": "均值",
    "std": "标准差",
    "min": "最小值",
    "max": "最大值",
    "range": "极差",
    "median": "中位数",
    "skew": "偏度",
    "kurtosis": "峰度",
    "q01": "1%分位数",
    "q05": "5%分位数",
    "q10": "10%分位数",
    "q25": "25%分位数",
    "q50": "50%分位数",
    "q75": "75%分位数",
    "q90": "90%分位数",
    "q95": "95%分位数",
    "q99": "99%分位数",
}


def field_zh(name: str) -> str:
    return FIELD_ZH.get(name, name)


def add_field_zh(df: pd.DataFrame, field_col: str, zh_col: str) -> pd.DataFrame:
    out = df.copy()
    insert_at = out.columns.get_loc(field_col) + 1
    out.insert(insert_at, zh_col, out[field_col].map(field_zh))
    return out


def field_dictionary() -> pd.DataFrame:
    rows = []
    ordered_fields = ["ts_code", "trade_date"] + RAW_NUMERIC_COLS + [
        "close_ma5",
        "close_ma20",
        "amount_ma20",
    ] + DERIVED_COLS
    seen = set()
    for field in ordered_fields:
        if field in seen:
            continue
        seen.add(field)
        rows.append(
            {
                "field": field,
                "field_zh": field_zh(field),
                "explanation": FIELD_EXPLANATION.get(field, ""),
            }
        )
    return pd.DataFrame(rows)


def stat_dictionary() -> pd.DataFrame:
    return pd.DataFrame(
        [{"stat": key, "stat_zh": value} for key, value in STAT_ZH.items()]
    )


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in RAW_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["pre_close"])
    out["amplitude_pct"] = (out["high"] - out["low"]) / out["pre_close"] * 100
    out["intraday_return_pct"] = (out["close"] - out["open"]) / out["open"] * 100
    out["gap_pct"] = (out["open"] - out["pre_close"]) / out["pre_close"] * 100
    out["upper_shadow_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["pre_close"] * 100
    out["lower_shadow_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["pre_close"] * 100
    out["vwap_proxy"] = np.where(out["vol"] != 0, out["amount"] / out["vol"], np.nan)
    out["close_ma5"] = out["close"].rolling(5, min_periods=5).mean()
    out["close_ma20"] = out["close"].rolling(20, min_periods=20).mean()
    out["close_ma5_gap_pct"] = (out["close"] / out["close_ma5"] - 1) * 100
    out["close_ma20_gap_pct"] = (out["close"] / out["close_ma20"] - 1) * 100
    out["pct_chg_5d"] = out["close"].pct_change(5) * 100
    out["pct_chg_20d"] = out["close"].pct_change(20) * 100
    out["volatility_20d"] = out["pct_chg"].rolling(20, min_periods=20).std()
    out["amount_ma20"] = out["amount"].rolling(20, min_periods=20).mean()
    out["amount_ma20_ratio"] = out["amount"] / out["amount_ma20"]
    return out


def missing_summary(df: pd.DataFrame, market: str) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "market": market,
            "column": df.columns,
            "column_zh": [field_zh(c) for c in df.columns],
            "missing_count": df.isna().sum().values,
            "missing_ratio": df.isna().mean().values,
            "non_missing_count": df.notna().sum().values,
            "dtype": [str(df[c].dtype) for c in df.columns],
        }
    )
    return result


def descriptive_stats(df: pd.DataFrame, cols: list[str], market: str, group: str) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col]
        row = {
            "market": market,
            "metric_group": group,
            "field": col,
            "field_zh": field_zh(col),
            "count": int(s.count()),
            "missing_count": int(s.isna().sum()),
            "missing_ratio": float(s.isna().mean()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "range": s.max() - s.min(),
            "median": s.median(),
            "skew": s.skew(),
            "kurtosis": s.kurt(),
        }
        for q in QUANTILES:
            row[f"q{int(q * 100):02d}"] = s.quantile(q)
        rows.append(row)
    return pd.DataFrame(rows)


def quality_checks(df: pd.DataFrame, market: str) -> pd.DataFrame:
    checks = {
        "duplicate_trade_date": int(df["trade_date"].duplicated().sum()),
        "unique_ts_code_count": int(df["ts_code"].nunique(dropna=True)),
        "high_less_than_low": int((df["high"] < df["low"]).sum()),
        "non_positive_price": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
        "negative_volume": int((df["vol"] < 0).sum()),
        "negative_amount": int((df["amount"] < 0).sum()),
        "change_mismatch_gt_0_01": int(((df["change"] - (df["close"] - df["pre_close"])).abs() > 0.01).sum()),
        "pct_chg_mismatch_gt_0_01": int(((df["pct_chg"] - (df["change"] / df["pre_close"] * 100)).abs() > 0.01).sum()),
    }
    return pd.DataFrame([{"market": market, "check": k, "value": v} for k, v in checks.items()])


def dataset_overview(df: pd.DataFrame, market: str) -> dict[str, object]:
    return {
        "market": market,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start_date": df["trade_date"].min().strftime("%Y-%m-%d"),
        "end_date": df["trade_date"].max().strftime("%Y-%m-%d"),
        "calendar_days": int((df["trade_date"].max() - df["trade_date"].min()).days + 1),
        "trading_days": int(df["trade_date"].nunique()),
        "ts_codes": ", ".join(sorted(df["ts_code"].dropna().unique())),
        "total_missing": int(df.isna().sum().sum()),
    }


def comparison_table(all_stats: pd.DataFrame) -> pd.DataFrame:
    focus_fields = [
        "close",
        "pct_chg",
        "vol",
        "amount",
        "amplitude_pct",
        "gap_pct",
        "intraday_return_pct",
        "volatility_20d",
        "amount_ma20_ratio",
    ]
    focus = all_stats[all_stats["field"].isin(focus_fields)].copy()
    value_cols = ["mean", "std", "min", "q05", "q25", "median", "q75", "q95", "max", "range", "skew", "kurtosis"]
    wide = focus.pivot(index="field", columns="market", values=value_cols)
    wide.columns = [f"{metric}_{market}" for metric, market in wide.columns]
    wide = wide.reset_index()
    wide.insert(1, "field_zh", wide["field"].map(field_zh))
    for metric in ["mean", "std", "median", "q95"]:
        a = f"{metric}_A股"
        hk = f"{metric}_港股"
        if a in wide.columns and hk in wide.columns:
            wide[f"{metric}_A_minus_HK"] = wide[a] - wide[hk]
    return wide


def fmt_num(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(v) >= 1_000_000:
        return f"{v:,.2f}"
    if abs(v) >= 1_000:
        return f"{v:,.2f}"
    return f"{v:.{digits}f}"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(fmt_num)
        else:
            show[col] = show[col].astype(str)
    headers = [str(c) for c in show.columns]
    rows = show.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        cleaned = [cell.replace("\n", "<br>").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(cleaned) + " |")
    return "\n".join(lines)


def html_table(df: pd.DataFrame, title: str, max_rows: int | None = None) -> str:
    show = df.copy()
    if max_rows is not None:
        show = show.head(max_rows)
    return f"<h3>{html.escape(title)}</h3>\n" + show.to_html(index=False, float_format=lambda x: fmt_num(x), classes="data-table")


def field_value(stats: pd.DataFrame, market: str, field: str, metric: str) -> float:
    row = stats[(stats["market"] == market) & (stats["field"] == field)]
    return float(row.iloc[0][metric])


def build_interpretation(overviews: pd.DataFrame, missing: pd.DataFrame, quality: pd.DataFrame, stats: pd.DataFrame) -> str:
    a_close_mean = field_value(stats, "A股", "close", "mean")
    hk_close_mean = field_value(stats, "港股", "close", "mean")
    a_ret_std = field_value(stats, "A股", "pct_chg", "std")
    hk_ret_std = field_value(stats, "港股", "pct_chg", "std")
    a_amp_mean = field_value(stats, "A股", "amplitude_pct", "mean")
    hk_amp_mean = field_value(stats, "港股", "amplitude_pct", "mean")
    a_amt_skew = field_value(stats, "A股", "amount", "skew")
    hk_amt_skew = field_value(stats, "港股", "amount", "skew")
    a_vol20 = field_value(stats, "A股", "volatility_20d", "mean")
    hk_vol20 = field_value(stats, "港股", "volatility_20d", "mean")
    a_gap_std = field_value(stats, "A股", "gap_pct", "std")
    hk_gap_std = field_value(stats, "港股", "gap_pct", "std")
    raw_required_cols = ["ts_code", "trade_date"] + RAW_NUMERIC_COLS
    raw_missing_total = int(missing[missing["column"].isin(raw_required_cols)]["missing_count"].sum())
    rolling_missing_total = int(missing["missing_count"].sum() - raw_missing_total)
    issue_checks = {
        "duplicate_trade_date",
        "high_less_than_low",
        "non_positive_price",
        "negative_volume",
        "negative_amount",
        "change_mismatch_gt_0_01",
        "pct_chg_mismatch_gt_0_01",
    }
    quality_issues = quality[(quality["check"].isin(issue_checks)) & (quality["value"] != 0)]

    quality_text = "轻量逻辑检查未发现明显结构性问题。"
    if not quality_issues.empty:
        quality_text = "轻量逻辑检查发现需要关注的项目：" + "；".join(
            f"{r.market}-{r.check}={r.value}" for r in quality_issues.itertuples()
        ) + "。"

    return f"""
## 可放入 Word 的描述总结

本次分析使用 TASK1 获取的中芯国际 A 股与港股近三年每日交易数据。A 股样本覆盖 {overviews.loc[overviews['market'] == 'A股', 'start_date'].iloc[0]} 至 {overviews.loc[overviews['market'] == 'A股', 'end_date'].iloc[0]}，共 {int(overviews.loc[overviews['market'] == 'A股', 'rows'].iloc[0])} 条交易日记录；港股样本覆盖 {overviews.loc[overviews['market'] == '港股', 'start_date'].iloc[0]} 至 {overviews.loc[overviews['market'] == '港股', 'end_date'].iloc[0]}，共 {int(overviews.loc[overviews['market'] == '港股', 'rows'].iloc[0])} 条交易日记录。两份原始数据关键字段缺失值数量为 {raw_missing_total}。滚动均线、20 日波动率、5/20 日收益等衍生指标因窗口期需要，在样本开头自然产生 {rolling_missing_total} 个空值，这属于计算口径导致的预期空值，不代表原始数据质量缺陷。{quality_text}

从价格分布看，A 股收盘价均值为 {a_close_mean:.2f}，港股收盘价均值为 {hk_close_mean:.2f}。这两个数值不能直接解释为 A 股比港股“更贵”或“更便宜”，因为二者使用不同币种、不同交易单位，并且 A/H 股价格还受到市场准入、投资者结构、流动性偏好和风险定价差异影响。对策略研究而言，价格水平本身不是最重要的信号，更重要的是价格相对自身历史区间的位置、均值与中位数的偏离程度，以及价格是否持续处在高分位或低分位区域。

从日度收益波动看，A 股日涨跌幅标准差为 {a_ret_std:.2f}%，港股为 {hk_ret_std:.2f}%。振幅均值方面，A 股为 {a_amp_mean:.2f}%，港股为 {hk_amp_mean:.2f}%。如果某一市场的涨跌幅标准差、振幅和尾部分位数更高，说明该市场短期噪声和交易冲击更强，趋势策略需要更宽的止损和更强的信号过滤，均值回复策略则需要重点评估极端波动后的回归速度，而不能只根据单日大涨大跌机械入场。

从跳空和盘中行为看，A 股 gap_pct 标准差为 {a_gap_std:.2f}%，港股为 {hk_gap_std:.2f}%。跳空波动越大，隔夜信息和开盘定价冲击越明显，策略上应区分“隔夜收益”和“盘中收益”：如果收益主要来自跳空，盘中追涨策略可能捕捉不到核心收益来源；如果盘中回撤和上下影线较大，则说明单纯按收盘价回测可能低估真实执行风险。

从成交活跃度看，A 股成交额偏度为 {a_amt_skew:.2f}，港股成交额偏度为 {hk_amt_skew:.2f}。成交额偏度为正通常意味着少数交易日显著放量，市场关注度并非均匀分布，而是在事件、政策、业绩或板块行情驱动下集中爆发。未来构造策略时，成交额和成交量不应只作为过滤条件，也可以作为状态变量：放量上涨可能代表趋势确认，放量下跌可能代表风险释放或恐慌抛压，缩量横盘则可能代表趋势信号可靠性下降。

20 日波动率均值方面，A 股为 {a_vol20:.2f}%，港股为 {hk_vol20:.2f}%。这对仓位管理有直接意义：波动率较高阶段应降低单笔仓位或提高信号确认门槛，波动率较低阶段可以允许更紧的止损和更高的资金利用率。对于中芯国际这类半导体龙头，价格会同时受到行业周期、政策预期、先进制程进展、全球科技股风险偏好和 A/H 市场联动影响，因此基础诊断的价值不只是描述历史数据，而是为后续策略设定交易成本、风控阈值、调仓频率和信号稳定性假设提供依据。

综合来看，本数据适合继续开展 A/H 股联动、趋势跟随、波动率分层和放量突破等策略研究。但在正式建模前，应避免直接比较 A 股与港股的绝对价格和成交量；更稳健的做法是使用收益率、分位数位置、滚动波动率、成交额相对均值倍数等标准化指标，把两个市场转化到可比较的统计尺度上。
""".strip()


def build_markdown_report(
    overviews: pd.DataFrame,
    field_dict: pd.DataFrame,
    stat_dict: pd.DataFrame,
    missing: pd.DataFrame,
    quality: pd.DataFrame,
    raw_stats: pd.DataFrame,
    derived_stats: pd.DataFrame,
    compare: pd.DataFrame,
    interpretation: str,
) -> str:
    report = [
        "# TASK2 问题1：中芯国际 A股与港股基础诊断分析",
        "",
        "## 数据概览",
        md_table(overviews),
        "",
        "## 字段中文说明",
        "下表解释报告和 notebook 中出现的英文字段名。统计表同时保留英文名和中文名，方便后续写代码时追溯原始字段。",
        "",
        md_table(field_dict),
        "",
        "## 统计指标中文说明",
        md_table(stat_dict),
        "",
        "## 缺失值检查",
        "完整缺失值结果已分别输出为 CSV 文件。下表展示所有字段缺失情况。",
        "",
        md_table(missing),
        "",
        "## 轻量逻辑一致性检查",
        md_table(quality),
        "",
        "## 原始字段描述性统计",
        "统计指标包含 count、mean、std、min、max、range、median、skew、kurtosis，以及 q01、q05、q10、q25、q50、q75、q90、q95、q99 分位数。",
        "",
        md_table(raw_stats),
        "",
        "## 衍生指标描述性统计",
        "衍生指标包括对数收益率、日内振幅、日内收益、跳空幅度、上下影线、近似 VWAP、均线偏离、5/20 日收益、20 日波动率和成交额相对 20 日均值。",
        "",
        md_table(derived_stats),
        "",
        "## A股与港股对比表",
        md_table(compare),
        "",
        interpretation,
        "",
        "## 输出文件说明",
        "- 分散输出：A股、港股各自的缺失值表、原始字段统计表、衍生指标统计表和日度衍生指标表。",
        "- 集合输出：合并缺失值表、合并描述性统计表、A/H 对比表、Markdown 报告、HTML 报告和 notebook。",
    ]
    return "\n".join(report)


def build_html_report(
    overviews: pd.DataFrame,
    field_dict: pd.DataFrame,
    stat_dict: pd.DataFrame,
    missing: pd.DataFrame,
    quality: pd.DataFrame,
    raw_stats: pd.DataFrame,
    derived_stats: pd.DataFrame,
    compare: pd.DataFrame,
    interpretation: str,
) -> str:
    body = "\n".join(
        [
            "<h1>TASK2 问题1：中芯国际 A股与港股基础诊断分析</h1>",
            html_table(overviews, "数据概览"),
            html_table(field_dict, "字段中文说明"),
            html_table(stat_dict, "统计指标中文说明"),
            html_table(missing, "缺失值检查"),
            html_table(quality, "轻量逻辑一致性检查"),
            html_table(raw_stats, "原始字段描述性统计"),
            html_table(derived_stats, "衍生指标描述性统计"),
            html_table(compare, "A股与港股对比表"),
            "<section class='interpretation'>"
            + "\n".join(
                f"<p>{html.escape(p)}</p>" if not p.startswith("## ") else f"<h2>{html.escape(p[3:])}</h2>"
                for p in interpretation.split("\n\n")
            )
            + "</section>",
        ]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>TASK2 问题1：中芯国际基础诊断分析</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans CJK SC", sans-serif; margin: 28px; color: #1f2933; line-height: 1.58; }}
    h1 {{ font-size: 28px; margin-bottom: 20px; }}
    h2 {{ font-size: 22px; margin-top: 28px; }}
    h3 {{ font-size: 17px; margin-top: 24px; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0 18px; }}
    .data-table th, .data-table td {{ border: 1px solid #d6d9de; padding: 6px 8px; text-align: right; vertical-align: top; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table th {{ background: #eef2f7; color: #111827; position: sticky; top: 0; }}
    .interpretation {{ max-width: 980px; font-size: 15px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def build_notebook() -> dict[str, object]:
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# TASK2 问题1：中芯国际基础诊断分析\n",
                "\n",
                "本 notebook 复现缺失值检查、描述性统计、衍生指标计算和 A/H 对比分析。\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "\n",
                "ROOT = Path('../..').resolve()\n",
                "OUT = ROOT / 'task2' / 'question1'\n",
                "a = pd.read_csv(ROOT / 'task1' / 'smic_a_daily.csv', encoding='utf-8-sig')\n",
                "hk = pd.read_csv(ROOT / 'task1' / 'smic_hk_daily.csv', encoding='utf-8-sig')\n",
                "a.head(), hk.head()\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 读取已生成结果\n", "完整结果由 `run_question1_analysis.py` 统一生成，以下单元读取核心输出表。\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "field_dict = pd.read_csv(OUT / 'task2_question1_field_dictionary.csv')\n",
                "stat_dict = pd.read_csv(OUT / 'task2_question1_stat_dictionary.csv')\n",
                "missing = pd.read_csv(OUT / 'task2_question1_missing_summary.csv')\n",
                "stats = pd.read_csv(OUT / 'task2_question1_descriptive_stats.csv')\n",
                "compare = pd.read_csv(OUT / 'task2_question1_ah_comparison.csv')\n",
                "field_dict, stat_dict, missing.head(), stats.head(), compare\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 字段中文名提示\n",
                "\n",
                "- `field` / `column` 是原始英文字段名，便于和 CSV 及代码对应。\n",
                "- `field_zh` / `column_zh` 是中文字段名，便于阅读和写作业报告。\n",
                "- 衍生指标如 `gap_pct`、`amplitude_pct`、`amount_ma20_ratio` 已在字段字典中解释具体含义。\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 后续扩展建议\n",
                "\n",
                "- 可在本 notebook 中继续增加价格走势图、收益率分布图、滚动波动率图和 A/H 联动图。\n",
                "- 建模时优先使用收益率、波动率、成交额相对均值和分位数位置等标准化指标。\n",
            ],
        },
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    enriched: dict[str, pd.DataFrame] = {}
    overviews = []
    missings = []
    qualities = []
    raw_stats_list = []
    derived_stats_list = []

    for market, meta in DATASETS.items():
        df = load_dataset(meta["path"])
        df2 = add_derived_metrics(df)
        frames[market] = df
        enriched[market] = df2

        overview = dataset_overview(df, market)
        overview["market_note"] = meta["market_note"]
        overviews.append(overview)

        missing = missing_summary(df2, market)
        quality = quality_checks(df, market)
        raw_stats = descriptive_stats(df2, RAW_NUMERIC_COLS, market, "原始字段")
        derived_stats = descriptive_stats(df2, DERIVED_COLS, market, "衍生指标")

        key = meta["key"]
        missing.to_csv(OUT / f"task2_question1_{key}_missing_summary.csv", index=False, encoding="utf-8-sig")
        raw_stats.to_csv(OUT / f"task2_question1_{key}_raw_descriptive_stats.csv", index=False, encoding="utf-8-sig")
        derived_stats.to_csv(OUT / f"task2_question1_{key}_derived_descriptive_stats.csv", index=False, encoding="utf-8-sig")
        df2.to_csv(OUT / f"task2_question1_{key}_daily_with_derived_metrics.csv", index=False, encoding="utf-8-sig")

        missings.append(missing)
        qualities.append(quality)
        raw_stats_list.append(raw_stats)
        derived_stats_list.append(derived_stats)

    overview_df = pd.DataFrame(overviews)
    field_dict_df = field_dictionary()
    stat_dict_df = stat_dictionary()
    missing_df = pd.concat(missings, ignore_index=True)
    quality_df = pd.concat(qualities, ignore_index=True)
    raw_stats_df = pd.concat(raw_stats_list, ignore_index=True)
    derived_stats_df = pd.concat(derived_stats_list, ignore_index=True)
    all_stats_df = pd.concat([raw_stats_df, derived_stats_df], ignore_index=True)
    compare_df = comparison_table(all_stats_df)
    interpretation = build_interpretation(overview_df, missing_df, quality_df, all_stats_df)

    overview_df.to_csv(OUT / "task2_question1_dataset_overview.csv", index=False, encoding="utf-8-sig")
    field_dict_df.to_csv(OUT / "task2_question1_field_dictionary.csv", index=False, encoding="utf-8-sig")
    stat_dict_df.to_csv(OUT / "task2_question1_stat_dictionary.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(OUT / "task2_question1_missing_summary.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(OUT / "task2_question1_quality_checks.csv", index=False, encoding="utf-8-sig")
    raw_stats_df.to_csv(OUT / "task2_question1_raw_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    derived_stats_df.to_csv(OUT / "task2_question1_derived_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    all_stats_df.to_csv(OUT / "task2_question1_descriptive_stats.csv", index=False, encoding="utf-8-sig")
    compare_df.to_csv(OUT / "task2_question1_ah_comparison.csv", index=False, encoding="utf-8-sig")

    md = build_markdown_report(
        overview_df,
        field_dict_df,
        stat_dict_df,
        missing_df,
        quality_df,
        raw_stats_df,
        derived_stats_df,
        compare_df,
        interpretation,
    )
    (OUT / "task2_question1_diagnostic_report.md").write_text(md, encoding="utf-8")

    html_report = build_html_report(
        overview_df,
        field_dict_df,
        stat_dict_df,
        missing_df,
        quality_df,
        raw_stats_df,
        derived_stats_df,
        compare_df,
        interpretation,
    )
    (OUT / "task2_question1_diagnostic_report.html").write_text(html_report, encoding="utf-8")

    notebook = build_notebook()
    (OUT / "task2_question1_diagnostic_analysis.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(OUT), "files": sorted(p.name for p in OUT.iterdir())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
