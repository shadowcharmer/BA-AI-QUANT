from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TASK1 = ROOT / "task1"
OUT = ROOT / "task2" / "question3"

DATASETS = {
    "a": {"market": "A股", "code": "688981.SH", "path": TASK1 / "smic_a_daily.csv"},
    "hk": {"market": "港股", "code": "00981.HK", "path": TASK1 / "smic_hk_daily.csv"},
}

NUMERIC_COLS = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def wilder(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def calc_rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilder(gain, n)
    avg_loss = wilder(loss, n)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50).where(avg_gain.notna(), np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in [6, 14, 24]:
        out[f"rsi_{n}"] = calc_rsi(out["close"], n)

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd_dif"] = ema12 - ema26
    out["macd_dea"] = out["macd_dif"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (out["macd_dif"] - out["macd_dea"]) * 2

    out["bb_mid"] = out["close"].rolling(20, min_periods=20).mean()
    bb_std = out["close"].rolling(20, min_periods=20).std()
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"] * 100
    out["bb_percent_b"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr_14"] = wilder(out["tr"], 14)
    out["atr_14_pct"] = out["atr_14"] / out["close"] * 100

    direction = np.sign(out["close"].diff()).fillna(0)
    out["obv"] = (direction * out["vol"]).cumsum()
    out["obv_ma20"] = out["obv"].rolling(20, min_periods=20).mean()

    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=out.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=out.index)
    plus_di = 100 * wilder(plus_dm, 14) / out["atr_14"]
    minus_di = 100 * wilder(minus_dm, 14) / out["atr_14"]
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx_14"] = wilder(dx, 14)

    for n in [6, 12, 24]:
        ma = out["close"].rolling(n, min_periods=n).mean()
        out[f"bias_{n}"] = (out["close"] - ma) / ma * 100

    z_mean = out["close"].rolling(20, min_periods=20).mean()
    z_std = out["close"].rolling(20, min_periods=20).std()
    out["zscore_20"] = (out["close"] - z_mean) / z_std
    return out


def cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def add_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["event_macd_golden"] = cross_up(out["macd_dif"], out["macd_dea"])
    out["event_macd_death"] = cross_down(out["macd_dif"], out["macd_dea"])
    out["event_rsi_overbought"] = out["rsi_14"] > 70
    out["event_rsi_oversold"] = out["rsi_14"] < 30
    out["event_bb_break_upper"] = out["close"] > out["bb_upper"]
    out["event_bb_break_lower"] = out["close"] < out["bb_lower"]
    out["event_atr_high"] = out["atr_14_pct"] >= out["atr_14_pct"].quantile(0.90)
    out["event_adx_trend"] = cross_up(out["adx_14"], pd.Series(25, index=out.index))
    out["event_bias_high"] = out["bias_12"] >= out["bias_12"].quantile(0.90)
    out["event_bias_low"] = out["bias_12"] <= out["bias_12"].quantile(0.10)
    out["event_z_high"] = out["zscore_20"] > 2
    out["event_z_low"] = out["zscore_20"] < -2
    close_20_high = out["close"] >= out["close"].rolling(20, min_periods=20).max()
    close_20_low = out["close"] <= out["close"].rolling(20, min_periods=20).min()
    obv_not_high = out["obv"] < out["obv"].rolling(20, min_periods=20).max()
    obv_not_low = out["obv"] > out["obv"].rolling(20, min_periods=20).min()
    out["event_obv_bear_div"] = close_20_high & obv_not_high
    out["event_obv_bull_div"] = close_20_low & obv_not_low
    return out


def summarize(market: str, code: str, df: pd.DataFrame) -> dict[str, object]:
    last = df.iloc[-1]
    return {
        "market": market,
        "code": code,
        "start_date": df["trade_date"].min().strftime("%Y-%m-%d"),
        "end_date": df["trade_date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        "last_close": float(last["close"]),
        "last_rsi_14": float(last["rsi_14"]),
        "last_macd_dif": float(last["macd_dif"]),
        "last_macd_dea": float(last["macd_dea"]),
        "last_atr_14_pct": float(last["atr_14_pct"]),
        "last_adx_14": float(last["adx_14"]),
        "last_bias_12": float(last["bias_12"]),
        "last_zscore_20": float(last["zscore_20"]),
        "macd_golden_cross_count": int(df["event_macd_golden"].sum()),
        "macd_death_cross_count": int(df["event_macd_death"].sum()),
        "rsi_overbought_days": int(df["event_rsi_overbought"].sum()),
        "rsi_oversold_days": int(df["event_rsi_oversold"].sum()),
        "bb_upper_break_days": int(df["event_bb_break_upper"].sum()),
        "bb_lower_break_days": int(df["event_bb_break_lower"].sum()),
        "zscore_extreme_days": int((df["event_z_high"] | df["event_z_low"]).sum()),
    }


def fmt(v: object, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def marker_points(df: pd.DataFrame, event_col: str, y_col: str, label: str, color: str, symbol: str) -> dict[str, object]:
    rows = df[df[event_col]].copy()
    return {
        "label": label,
        "color": color,
        "symbol": symbol,
        "x": rows["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
        "y": rows[y_col].round(4).where(rows[y_col].notna(), None).tolist(),
    }


def series_obj(df: pd.DataFrame, field: str, label: str, color: str, kind: str = "line", visible: bool = True) -> dict[str, object]:
    values = df[field].replace([np.inf, -np.inf], np.nan)
    return {
        "field": field,
        "label": label,
        "color": color,
        "kind": kind,
        "visible": visible,
        "y": [None if pd.isna(v) else round(float(v), 5) for v in values],
    }


def chart_configs(df: pd.DataFrame, market: str, code: str) -> list[dict[str, object]]:
    x = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    return [
        {
            "id": "bb",
            "title": f"图1 {market} {code} 收盘价与布林带",
            "subtitle": "高亮：突破上轨、跌破下轨。",
            "x": x,
            "series": [
                series_obj(df, "close", "收盘价", "#1f77b4"),
                series_obj(df, "bb_upper", "布林带上轨", "#d62728"),
                series_obj(df, "bb_mid", "布林带中轨", "#7f7f7f"),
                series_obj(df, "bb_lower", "布林带下轨", "#2ca02c"),
            ],
            "refs": [],
            "markers": [
                marker_points(df, "event_bb_break_upper", "close", "突破上轨", "#d62728", "▲"),
                marker_points(df, "event_bb_break_lower", "close", "跌破下轨", "#2ca02c", "▼"),
            ],
        },
        {
            "id": "rsi",
            "title": f"图2 {market} {code} RSI 相对强弱指标",
            "subtitle": "高亮：RSI14 超买/超卖。",
            "x": x,
            "series": [
                series_obj(df, "rsi_6", "RSI 6", "#ff7f0e"),
                series_obj(df, "rsi_14", "RSI 14", "#1f77b4"),
                series_obj(df, "rsi_24", "RSI 24", "#9467bd"),
            ],
            "refs": [{"value": 70, "label": "超买70"}, {"value": 30, "label": "超卖30"}],
            "markers": [
                marker_points(df, "event_rsi_overbought", "rsi_14", "RSI14>70", "#d62728", "●"),
                marker_points(df, "event_rsi_oversold", "rsi_14", "RSI14<30", "#2ca02c", "●"),
            ],
        },
        {
            "id": "macd",
            "title": f"图3 {market} {code} MACD 趋势动能",
            "subtitle": "高亮：DIF 上穿 DEA 为金叉，下穿为死叉。",
            "x": x,
            "series": [
                series_obj(df, "macd_hist", "MACD柱", "#9aa5b1", "bar"),
                series_obj(df, "macd_dif", "DIF", "#1f77b4"),
                series_obj(df, "macd_dea", "DEA", "#ff7f0e"),
            ],
            "refs": [{"value": 0, "label": "0轴"}],
            "markers": [
                marker_points(df, "event_macd_golden", "macd_dif", "金叉", "#d62728", "▲"),
                marker_points(df, "event_macd_death", "macd_dif", "死叉", "#2ca02c", "▼"),
            ],
        },
        {
            "id": "atr",
            "title": f"图4 {market} {code} ATR 平均真实波幅",
            "subtitle": "高亮：ATR相对波幅处于历史前10%的高波动日。",
            "x": x,
            "series": [
                series_obj(df, "atr_14_pct", "ATR14/收盘价(%)", "#d62728"),
                series_obj(df, "atr_14", "ATR14", "#1f77b4", visible=False),
            ],
            "refs": [],
            "markers": [marker_points(df, "event_atr_high", "atr_14_pct", "高波动", "#d62728", "●")],
        },
        {
            "id": "obv",
            "title": f"图5 {market} {code} OBV 能量潮",
            "subtitle": "高亮：20日新高/新低附近的量价背离线索。",
            "x": x,
            "series": [
                series_obj(df, "obv", "OBV", "#1f77b4"),
                series_obj(df, "obv_ma20", "OBV 20日均线", "#ff7f0e"),
            ],
            "refs": [],
            "markers": [
                marker_points(df, "event_obv_bear_div", "obv", "疑似顶背离", "#d62728", "▼"),
                marker_points(df, "event_obv_bull_div", "obv", "疑似底背离", "#2ca02c", "▲"),
            ],
        },
        {
            "id": "adx",
            "title": f"图6 {market} {code} ADX 与 DI 趋势强度",
            "subtitle": "高亮：ADX 上穿 25，代表趋势强度进入较强区间。",
            "x": x,
            "series": [
                series_obj(df, "adx_14", "ADX14", "#111827"),
                series_obj(df, "plus_di", "+DI", "#d62728"),
                series_obj(df, "minus_di", "-DI", "#2ca02c"),
            ],
            "refs": [{"value": 25, "label": "趋势较强25"}, {"value": 20, "label": "震荡20"}],
            "markers": [marker_points(df, "event_adx_trend", "adx_14", "ADX上穿25", "#111827", "●")],
        },
        {
            "id": "bias",
            "title": f"图7 {market} {code} BIAS 乖离率",
            "subtitle": "高亮：BIAS12 位于历史前10%或后10%。",
            "x": x,
            "series": [
                series_obj(df, "bias_6", "BIAS 6", "#ff7f0e"),
                series_obj(df, "bias_12", "BIAS 12", "#1f77b4"),
                series_obj(df, "bias_24", "BIAS 24", "#9467bd"),
            ],
            "refs": [{"value": 0, "label": "0轴"}],
            "markers": [
                marker_points(df, "event_bias_high", "bias_12", "BIAS12高位", "#d62728", "●"),
                marker_points(df, "event_bias_low", "bias_12", "BIAS12低位", "#2ca02c", "●"),
            ],
        },
        {
            "id": "zscore",
            "title": f"图8 {market} {code} Z-score 标准分数",
            "subtitle": "高亮：Z-score 超过 +2 或低于 -2 的极端偏离。",
            "x": x,
            "series": [series_obj(df, "zscore_20", "Z-score 20", "#1f77b4")],
            "refs": [{"value": 2, "label": "+2"}, {"value": 0, "label": "0"}, {"value": -2, "label": "-2"}],
            "markers": [
                marker_points(df, "event_z_high", "zscore_20", "Z>2", "#d62728", "▲"),
                marker_points(df, "event_z_low", "zscore_20", "Z<-2", "#2ca02c", "▼"),
            ],
        },
    ]


def interpretation_for_chart(chart_id: str, market: str, df: pd.DataFrame) -> str:
    last = df.iloc[-1]
    if chart_id == "bb":
        pos = last["bb_percent_b"]
        breaks = int((df["event_bb_break_upper"] | df["event_bb_break_lower"]).sum())
        return f"样本末期收盘价在布林带中的 percent_b 为 {fmt(pos)}，样本内共出现 {breaks} 次上下轨突破。布林带突破说明价格短期偏离近期波动区间，后续可结合成交量和趋势强度判断是突破延续还是均值回归。"
    if chart_id == "rsi":
        return f"样本末期 RSI14 为 {fmt(last['rsi_14'])}。样本中 RSI14 超买 {int(df['event_rsi_overbought'].sum())} 天、超卖 {int(df['event_rsi_oversold'].sum())} 天；RSI 适合提示情绪过热或过冷，但在强趋势中可能长期停留在高位或低位。"
    if chart_id == "macd":
        trend = "DIF 高于 DEA，短线动能偏多" if last["macd_dif"] > last["macd_dea"] else "DIF 低于 DEA，短线动能偏弱"
        return f"样本末期 {trend}。样本中 MACD 金叉 {int(df['event_macd_golden'].sum())} 次、死叉 {int(df['event_macd_death'].sum())} 次；金叉和死叉是趋势切换线索，但需要用 ADX 或成交量过滤震荡期误信号。"
    if chart_id == "atr":
        return f"样本末期 ATR14/收盘价为 {fmt(last['atr_14_pct'])}%。图中高亮点是历史前10%的高波动日，这类阶段更适合降低仓位、放宽止损或暂停过于敏感的短线信号。"
    if chart_id == "obv":
        return f"样本内识别到疑似顶背离 {int(df['event_obv_bear_div'].sum())} 次、疑似底背离 {int(df['event_obv_bull_div'].sum())} 次。OBV 的重点不是绝对数值，而是价格创新高或新低时成交量趋势是否同步确认。"
    if chart_id == "adx":
        direction = "+DI 高于 -DI，多头方向占优" if last["plus_di"] > last["minus_di"] else "-DI 高于 +DI，空头方向占优"
        return f"样本末期 ADX14 为 {fmt(last['adx_14'])}，{direction}。ADX 高于 25 通常说明趋势强度较高，适合趋势类策略；低于 20 时更容易进入震荡，均值回复或区间策略更有参考价值。"
    if chart_id == "bias":
        return f"样本末期 BIAS12 为 {fmt(last['bias_12'])}%。高亮点表示价格相对均线偏离处于历史尾部区域，正乖离过大要警惕回撤，负乖离过大则可能提示修复机会。"
    if chart_id == "zscore":
        return f"样本末期 Z-score20 为 {fmt(last['zscore_20'])}。Z-score 超过 +/-2 的点代表价格显著偏离 20 日均值；它适合均值回复观察，但若 ADX 同时较高，极端值也可能代表趋势延续。"
    return ""


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>TASK2 Question3 中芯国际技术指标分析报告</title>
  <style>
    html, body { max-width: 100%; overflow-x: hidden; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Noto Sans CJK SC", sans-serif; margin: 0; color: #17202a; background: #f6f8fb; }
    header { padding: 24px 32px; background: #ffffff; border-bottom: 1px solid #d9dee7; }
    h1 { margin: 0 0 8px; font-size: 26px; }
    h2 { margin: 28px 0 12px; font-size: 21px; }
    h3 { margin: 0 0 6px; font-size: 17px; }
    p { line-height: 1.65; }
    main { padding: 22px 32px 40px; box-sizing: border-box; max-width: 100vw; }
    .tabs { display: flex; gap: 8px; margin: 14px 0 20px; }
    .tab-btn { border: 1px solid #b8c2d1; background: #fff; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: 650; }
    .tab-btn.active { background: #1f5eff; color: white; border-color: #1f5eff; }
    .market-panel { display: none; }
    .market-panel.active { display: block; }
    .summary, .indicator-note, .chart-card { background: #fff; border: 1px solid #d9dee7; border-radius: 8px; padding: 16px; margin-bottom: 16px; box-sizing: border-box; max-width: 100%; }
    .table-scroll { max-width: 100%; overflow-x: auto; overflow-y: hidden; -webkit-overflow-scrolling: touch; border: 1px solid #e5e9f0; border-radius: 6px; }
    .summary-table { min-width: 1280px; }
    .chart-card { overflow: hidden; }
    .subtitle { color: #52606d; font-size: 13px; margin-bottom: 8px; }
    .chart { width: 100%; height: 430px; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px 14px; margin: 8px 0 4px; font-size: 12px; }
    .legend-item { cursor: pointer; user-select: none; display: inline-flex; align-items: center; gap: 5px; }
    .legend-swatch { width: 18px; height: 3px; display: inline-block; }
    .legend-item.off { opacity: .35; text-decoration: line-through; }
    .tooltip { position: fixed; pointer-events: none; background: rgba(17,24,39,.92); color: #fff; padding: 7px 9px; border-radius: 5px; font-size: 12px; display: none; z-index: 20; max-width: 260px; }
    table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }
    th, td { border: 1px solid #d9dee7; padding: 7px 9px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th { background: #eef2f7; }
    .marker-label { font-size: 11px; paint-order: stroke; stroke: #fff; stroke-width: 3px; }
    .axis text { fill: #52606d; font-size: 11px; }
    .axis line { stroke: #d9dee7; }
    .ref-line { stroke: #9aa5b1; stroke-dasharray: 4 4; }
    .grid-line { stroke: #eef2f7; }
  </style>
</head>
<body>
<header>
  <h1>TASK2 Question3 中芯国际技术指标分析报告</h1>
  <p>本报告基于 TASK1 已保存的 A 股与港股日线数据，计算 RSI、MACD、布林带、ATR、OBV、ADX、BIAS 和 Z-score。图中用彩色标记高亮关键事件，例如 MACD 金叉/死叉、RSI 超买/超卖、布林带突破和 Z-score 极端偏离。点击图例可隐藏或显示对应曲线。</p>
</header>
<main>
  <section class="summary">
    <h2>数据与指标说明</h2>
    <p>ATR 按 Average True Range 平均真实波幅处理；RSI 参数为 6、14、24；MACD 参数为 12/26/9；布林带为 20 日均线和 2 倍标准差；ADX 与 ATR 为 14 日；BIAS 为 6、12、24；Z-score 为 20 日滚动窗口。</p>
    <div class="table-scroll">__SUMMARY_TABLE__</div>
  </section>
  <div class="tabs">
    <button class="tab-btn active" data-target="panel-a">A股 688981.SH</button>
    <button class="tab-btn" data-target="panel-hk">港股 00981.HK</button>
  </div>
  __PANELS__
</main>
<div id="tooltip" class="tooltip"></div>
<script>
const chartData = __CHART_DATA__;
const tooltip = document.getElementById('tooltip');
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.market-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.target).classList.add('active');
    setTimeout(renderAll, 10);
  });
});
function clean(values) { return values.filter(v => v !== null && !Number.isNaN(v)); }
function niceTicks(min, max, n=5) {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [min || 0];
  const step = (max - min) / (n - 1);
  return Array.from({length:n}, (_,i)=>min+i*step);
}
function yScale(v, min, max, top, bottom) { return bottom - (v - min) / (max - min || 1) * (bottom - top); }
function xScale(i, count, left, right) { return left + i / Math.max(count - 1, 1) * (right - left); }
function linePath(xs, ys, minY, maxY, left, right, top, bottom) {
  let d = "", open = false;
  ys.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) { open = false; return; }
    const x = xScale(i, ys.length, left, right);
    const y = yScale(v, minY, maxY, top, bottom);
    d += (open ? "L" : "M") + x.toFixed(2) + "," + y.toFixed(2);
    open = true;
  });
  return d;
}
function renderChart(container, cfg) {
  container.innerHTML = "";
  const width = container.clientWidth || 1000, height = 405;
  const m = {left: 58, right: 72, top: 18, bottom: 52};
  const left = m.left, right = width - m.right, top = m.top, bottom = height - m.bottom;
  const visibleSeries = cfg.series.filter(s => s.visible !== false);
  let vals = [];
  visibleSeries.forEach(s => vals = vals.concat(clean(s.y)));
  cfg.markers.forEach(mk => vals = vals.concat(clean(mk.y)));
  (cfg.refs || []).forEach(r => vals.push(r.value));
  let minY = Math.min(...vals), maxY = Math.max(...vals);
  const pad = (maxY - minY || 1) * 0.08;
  minY -= pad; maxY += pad;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", width); svg.setAttribute("height", height); svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const ticks = niceTicks(minY, maxY, 5);
  ticks.forEach(t => {
    const y = yScale(t, minY, maxY, top, bottom);
    svg.insertAdjacentHTML("beforeend", `<line class="grid-line" x1="${left}" x2="${right}" y1="${y}" y2="${y}"></line><text x="8" y="${y+4}" class="axis">${t.toFixed(2)}</text>`);
  });
  const xTicks = [0, Math.floor(cfg.x.length/4), Math.floor(cfg.x.length/2), Math.floor(cfg.x.length*3/4), cfg.x.length-1];
  xTicks.forEach(i => {
    const x = xScale(i, cfg.x.length, left, right);
    const anchor = i === 0 ? "start" : (i === cfg.x.length - 1 ? "end" : "middle");
    const labelX = i === 0 ? x + 2 : (i === cfg.x.length - 1 ? x - 2 : x);
    svg.insertAdjacentHTML("beforeend", `<line class="grid-line" y1="${top}" y2="${bottom}" x1="${x}" x2="${x}"></line><text x="${labelX}" y="${height-20}" text-anchor="${anchor}" class="axis">${cfg.x[i]}</text>`);
  });
  (cfg.refs || []).forEach(r => {
    const y = yScale(r.value, minY, maxY, top, bottom);
    svg.insertAdjacentHTML("beforeend", `<line class="ref-line" x1="${left}" x2="${right}" y1="${y}" y2="${y}"></line><text x="${right-70}" y="${y-4}" class="axis">${r.label}</text>`);
  });
  visibleSeries.forEach(s => {
    if (s.kind === "bar") {
      const zero = yScale(0, minY, maxY, top, bottom);
      const bw = Math.max(2, (right-left) / cfg.x.length * 0.62);
      s.y.forEach((v, i) => {
        if (v === null) return;
        const x = xScale(i, cfg.x.length, left, right) - bw/2;
        const y = yScale(v, minY, maxY, top, bottom);
        const h = Math.abs(zero - y);
        const yy = Math.min(y, zero);
        const color = v >= 0 ? "#d66b6b" : "#4aa879";
        svg.insertAdjacentHTML("beforeend", `<rect x="${x}" y="${yy}" width="${bw}" height="${Math.max(h,1)}" fill="${color}" opacity="0.55"></rect>`);
      });
    } else {
      svg.insertAdjacentHTML("beforeend", `<path d="${linePath(cfg.x, s.y, minY, maxY, left, right, top, bottom)}" fill="none" stroke="${s.color}" stroke-width="2"></path>`);
    }
  });
  cfg.markers.forEach(mk => {
    mk.x.forEach((date, j) => {
      const i = cfg.x.indexOf(date), v = mk.y[j];
      if (i < 0 || v === null) return;
      const x = xScale(i, cfg.x.length, left, right);
      const y = yScale(v, minY, maxY, top, bottom);
      const text = mk.symbol || "●";
      const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
      node.setAttribute("x", x); node.setAttribute("y", y - 7); node.setAttribute("text-anchor", "middle");
      node.setAttribute("fill", mk.color); node.setAttribute("class", "marker-label"); node.textContent = text;
      node.addEventListener("mousemove", ev => showTip(ev, `${mk.label}<br>${date}<br>${Number(v).toFixed(3)}`));
      node.addEventListener("mouseleave", hideTip);
      svg.appendChild(node);
    });
  });
  const overlay = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  overlay.setAttribute("x", left); overlay.setAttribute("y", top); overlay.setAttribute("width", right-left); overlay.setAttribute("height", bottom-top); overlay.setAttribute("fill", "transparent");
  overlay.addEventListener("mousemove", ev => {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    const idx = Math.round((loc.x - left) / (right-left) * (cfg.x.length-1));
    if (idx < 0 || idx >= cfg.x.length) return;
    let lines = [`<b>${cfg.x[idx]}</b>`];
    visibleSeries.forEach(s => { const v = s.y[idx]; if (v !== null) lines.push(`${s.label}: ${Number(v).toFixed(3)}`); });
    showTip(ev, lines.join("<br>"));
  });
  overlay.addEventListener("mouseleave", hideTip);
  svg.appendChild(overlay);
  container.appendChild(svg);
}
function showTip(ev, html) { tooltip.innerHTML = html; tooltip.style.left = (ev.clientX + 12) + "px"; tooltip.style.top = (ev.clientY + 12) + "px"; tooltip.style.display = "block"; }
function hideTip() { tooltip.style.display = "none"; }
function renderAll() {
  document.querySelectorAll(".chart").forEach(el => {
    if (!el.closest(".market-panel").classList.contains("active")) return;
    const cfg = chartData[el.dataset.market].find(c => c.id === el.dataset.chart);
    renderChart(el, cfg);
  });
}
function initLegends() {
  document.querySelectorAll(".legend-item").forEach(item => {
    item.addEventListener("click", () => {
      const cfg = chartData[item.dataset.market].find(c => c.id === item.dataset.chart);
      const s = cfg.series.find(x => x.field === item.dataset.field);
      s.visible = s.visible === false ? true : false;
      item.classList.toggle("off", s.visible === false);
      renderAll();
    });
  });
}
window.addEventListener("resize", renderAll);
initLegends(); renderAll();
</script>
</body>
</html>
"""


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="summary-table", float_format=lambda x: f"{x:.3f}")


def build_html(chart_data: dict[str, list[dict[str, object]]], summaries: pd.DataFrame, dataframes: dict[str, pd.DataFrame]) -> str:
    panels = []
    for key, meta in DATASETS.items():
        active = " active" if key == "a" else ""
        chart_cards = []
        df = dataframes[key]
        for cfg in chart_data[key]:
            legend = "".join(
                f'<span class="legend-item{" off" if s.get("visible") is False else ""}" data-market="{key}" data-chart="{cfg["id"]}" data-field="{s["field"]}">'
                f'<span class="legend-swatch" style="background:{s["color"]}"></span>{s["label"]}</span>'
                for s in cfg["series"]
            )
            interp = interpretation_for_chart(cfg["id"], meta["market"], df)
            chart_cards.append(
                f'<section class="chart-card"><h3>{cfg["title"]}</h3><div class="subtitle">{cfg["subtitle"]}</div>'
                f'<div class="legend">{legend}</div><div class="chart" data-market="{key}" data-chart="{cfg["id"]}"></div>'
                f'<p>{interp}</p></section>'
            )
        panels.append(
            f'<section id="panel-{key}" class="market-panel{active}">'
            f'<div class="indicator-note"><h2>{meta["market"]} {meta["code"]} 指标图表与解读</h2>'
            f'<p>图中所有彩色符号均为重要事件高亮。它们不是机械买卖点，而是需要结合趋势、波动、成交量和交易成本进一步筛选的候选信号。</p></div>'
            + "\n".join(chart_cards)
            + "</section>"
        )
    html = HTML_TEMPLATE.replace("__SUMMARY_TABLE__", html_table(summaries))
    html = html.replace("__PANELS__", "\n".join(panels))
    html = html.replace("__CHART_DATA__", json.dumps(chart_data, ensure_ascii=False))
    return html


def build_notebook() -> dict[str, object]:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# TASK2 Question3 技术指标分析\n", "\n", "本 notebook 复现指标计算和 HTML 报告生成流程。\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "ROOT = Path('../..').resolve()\n",
            "OUT = ROOT / 'task2' / 'question3'\n",
            "a = pd.read_csv(ROOT / 'task1' / 'smic_a_daily.csv', encoding='utf-8-sig')\n",
            "hk = pd.read_csv(ROOT / 'task1' / 'smic_hk_daily.csv', encoding='utf-8-sig')\n",
            "a.head(), hk.head()\n",
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 指标口径\n", "- RSI: 6/14/24\n- MACD: 12/26/9\n- Bollinger Bands: 20日、2倍标准差\n- ATR/ADX: 14日\n- BIAS: 6/12/24\n- Z-score: 20日\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "# 导入脚本中的可复用函数，逐步复现：加载 -> 指标计算 -> 事件识别。\n",
            "import sys\n",
            "sys.path.insert(0, str(OUT))\n",
            "import run_question3_analysis as q3\n",
            "\n",
            "a_ind = q3.add_events(q3.add_indicators(q3.load_data(ROOT / 'task1' / 'smic_a_daily.csv')))\n",
            "hk_ind = q3.add_events(q3.add_indicators(q3.load_data(ROOT / 'task1' / 'smic_hk_daily.csv')))\n",
            "a_ind[['trade_date', 'close', 'rsi_14', 'macd_dif', 'macd_dea', 'atr_14_pct', 'adx_14', 'bias_12', 'zscore_20']].tail()\n",
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## 关键事件统计\n", "下面统计图中被高亮展示的重要事件数量，例如 MACD 金叉/死叉、RSI 超买/超卖和布林带突破。\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "summary = pd.read_csv(OUT / 'task2_question3_indicator_summary.csv')\n",
            "summary\n",
        ]},
        {"cell_type": "markdown", "metadata": {}, "source": ["## HTML 报告\n", "交互式图表、图例点击和关键事件高亮均在 `task2_question3_technical_report.html` 中查看。\n"]},
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def update_spec_status(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace("本文件仅作为执行前计划，暂不计算指标、暂不生成图表、暂不生成正式 HTML 报告。", "本文件最初作为执行前计划；当前已按确认口径执行，ART 按 ATR 处理，并生成 notebook、指标 CSV 和交互式 HTML 报告。")
    text = text.replace("当前仅完成 spec 文件：\n\n- 未读取数据。\n- 未计算 RSI、MACD、Bollinger Bands、ATR、OBV、ADX、BIAS、Z-score。\n- 未生成 notebook。\n- 未生成指标 CSV。\n- 未生成 HTML 报告。", "当前已执行完成：\n\n- 已读取 A 股和港股数据。\n- 已计算 RSI、MACD、Bollinger Bands、ATR、OBV、ADX、BIAS、Z-score。\n- 已生成 notebook。\n- 已生成指标 CSV。\n- 已生成可切换 A 股/港股的交互式 HTML 报告。\n- 已在图中高亮关键事件，包括 MACD 金叉/死叉、RSI 超买/超卖、布林带突破、ATR 高波动、OBV 背离线索、ADX 趋势增强、BIAS 极端乖离和 Z-score 极端偏离。")
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chart_data: dict[str, list[dict[str, object]]] = {}
    dataframes: dict[str, pd.DataFrame] = {}
    summaries = []

    for key, meta in DATASETS.items():
        df = add_events(add_indicators(load_data(meta["path"])))
        dataframes[key] = df
        chart_data[key] = chart_configs(df, meta["market"], meta["code"])
        summaries.append(summarize(meta["market"], meta["code"], df))
        df.to_csv(OUT / f"task2_question3_indicators_{key}.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "task2_question3_indicator_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "task2_question3_technical_report.html").write_text(build_html(chart_data, summary_df, dataframes), encoding="utf-8")
    (OUT / "task2_question3_technical_indicators.ipynb").write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    update_spec_status(OUT / "task2_question3_technical_indicators_spec.md")
    print(json.dumps({"output_dir": str(OUT), "files": sorted(p.name for p in OUT.iterdir())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
