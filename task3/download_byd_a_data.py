#!/usr/bin/env python3
"""Download BYD A-share daily data and generate quality checked qfq files.

Data source: Eastmoney historical kline API.
Stock: 002594.SZ / BYD
"""

from __future__ import annotations

import csv
import json
import math
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "sample_data"
SECID = "0.002594"
TS_CODE = "002594.SZ"
NAME = "比亚迪"
TODAY = date.today()
START = TODAY - timedelta(days=365 * 3)
END = TODAY


FIELDS = [
    "trade_date",
    "ts_code",
    "name",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
    "turnover_rate",
]


def fetch_kline(fqt: int) -> list[dict[str, float | str]]:
    params = {
        "secid": SECID,
        "klt": "101",
        "fqt": str(fqt),
        "beg": START.strftime("%Y%m%d"),
        "end": END.strftime("%Y%m%d"),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    klines = payload.get("data", {}).get("klines")
    if not klines:
        raise RuntimeError(f"Eastmoney returned no kline data for fqt={fqt}: {payload}")

    rows: list[dict[str, float | str]] = []
    previous_close: float | None = None
    for line in klines:
        parts = line.split(",")
        if len(parts) < 11:
            raise RuntimeError(f"Unexpected kline row: {line}")

        trade_date = parts[0]
        open_price = float(parts[1])
        close_price = float(parts[2])
        high_price = float(parts[3])
        low_price = float(parts[4])
        vol = float(parts[5])
        amount = float(parts[6])
        pct_chg = float(parts[8])
        change = float(parts[9])
        turnover_rate = float(parts[10])

        if previous_close is None:
            if abs(change) > 1e-12:
                pre_close = close_price - change
            elif abs(pct_chg) > 1e-12:
                pre_close = close_price / (1 + pct_chg / 100)
            else:
                pre_close = close_price
        else:
            pre_close = previous_close
            change = close_price - pre_close
            pct_chg = 0 if pre_close == 0 else change / pre_close * 100

        rows.append(
            {
                "trade_date": trade_date,
                "ts_code": TS_CODE,
                "name": NAME,
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close_price, 4),
                "pre_close": round(pre_close, 4),
                "change": round(change, 4),
                "pct_chg": round(pct_chg, 6),
                "vol": round(vol, 2),
                "amount": round(amount, 2),
                "turnover_rate": round(turnover_rate, 6),
            }
        )
        previous_close = close_price

    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    return False


def quality_summary(raw_rows: list[dict[str, object]], qfq_rows: list[dict[str, object]]) -> dict[str, object]:
    dates = [str(row["trade_date"]) for row in qfq_rows]
    duplicate_count = sum(count - 1 for count in Counter(dates).values() if count > 1)
    core_cols = ["open", "high", "low", "close", "vol", "amount"]
    missing_count = sum(is_missing(row[col]) for row in qfq_rows for col in core_cols)

    price_anomalies = []
    volume_anomalies = []
    for row in qfq_rows:
        high = float(row["high"])
        low = float(row["low"])
        open_price = float(row["open"])
        close = float(row["close"])
        vol = float(row["vol"])
        amount = float(row["amount"])
        if not (high >= max(open_price, close, low) and low <= min(open_price, close, high)):
            price_anomalies.append(row["trade_date"])
        if vol < 0 or amount < 0:
            volume_anomalies.append(row["trade_date"])

    raw_by_date = {row["trade_date"]: row for row in raw_rows}
    qfq_changed_rows = 0
    factors = []
    factor_rows = []
    for row in qfq_rows:
        raw = raw_by_date.get(row["trade_date"])
        if not raw:
            continue
        raw_close = float(raw["close"])
        qfq_close = float(row["close"])
        factor = 1 if raw_close == 0 else qfq_close / raw_close
        factors.append(round(factor, 8))
        if abs(qfq_close - raw_close) > 1e-6:
            qfq_changed_rows += 1
        factor_rows.append(
            {
                "trade_date": row["trade_date"],
                "ts_code": TS_CODE,
                "name": NAME,
                "implied_qfq_factor": round(factor, 8),
            }
        )

    return {
        "ts_code": TS_CODE,
        "name": NAME,
        "source": "Eastmoney kline API",
        "adjustment": "前复权 fqt=1",
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
        "trading_days": len(qfq_rows),
        "duplicate_dates": duplicate_count,
        "core_ohlcv_missing_values": missing_count,
        "price_relation_anomaly_rows": len(price_anomalies),
        "volume_anomaly_rows": len(volume_anomalies),
        "raw_qfq_aligned_dates": len(factor_rows),
        "qfq_changed_rows": qfq_changed_rows,
        "implied_factor_unique_count": len(set(factors)),
        "min_implied_qfq_factor": min(factors) if factors else None,
        "max_implied_qfq_factor": max(factors) if factors else None,
        "factor_rows": factor_rows,
    }


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# 比亚迪 A 股近 3 年数据质量检测报告",
        "",
        f"- 代码：{summary['ts_code']}",
        f"- 名称：{summary['name']}",
        f"- 数据源：{summary['source']}",
        f"- 复权方式：{summary['adjustment']}",
        f"- 样本范围：{summary['start_date']} 至 {summary['end_date']}",
        f"- 交易日数量：{summary['trading_days']}",
        f"- 核心 OHLCV 缺失值数量：{summary['core_ohlcv_missing_values']}",
        f"- 重复日期数量：{summary['duplicate_dates']}",
        f"- 价格关系异常行数：{summary['price_relation_anomaly_rows']}",
        f"- 成交量/成交额异常行数：{summary['volume_anomaly_rows']}",
        f"- 原始与前复权对齐日期数量：{summary['raw_qfq_aligned_dates']}",
        f"- 前复权后价格发生变化的行数：{summary['qfq_changed_rows']}",
        f"- 隐含前复权因子唯一值数量：{summary['implied_factor_unique_count']}",
        f"- 隐含前复权因子范围：{summary['min_implied_qfq_factor']} 至 {summary['max_implied_qfq_factor']}",
        "",
        "## 结论",
        "",
    ]
    if (
        summary["core_ohlcv_missing_values"] == 0
        and summary["duplicate_dates"] == 0
        and summary["price_relation_anomaly_rows"] == 0
        and summary["volume_anomaly_rows"] == 0
    ):
        lines.append("数据质量检测通过；已生成原始行情、前复权行情和隐含前复权因子文件。")
    else:
        lines.append("数据存在需关注项目，请结合上方统计进一步检查。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_rows = fetch_kline(fqt=0)
    qfq_rows = fetch_kline(fqt=1)

    if [row["trade_date"] for row in raw_rows] != [row["trade_date"] for row in qfq_rows]:
        raise RuntimeError("Raw and qfq date ranges do not align.")

    raw_by_date = {row["trade_date"]: row for row in raw_rows}
    enriched_qfq_rows = []
    for row in qfq_rows:
        raw = raw_by_date[row["trade_date"]]
        merged = dict(row)
        for col in ["open", "high", "low", "close", "pre_close"]:
            merged[f"raw_{col}"] = raw[col]
        raw_close = float(raw["close"])
        merged["qfq_factor"] = round(1 if raw_close == 0 else float(row["close"]) / raw_close, 8)
        enriched_qfq_rows.append(merged)

    write_csv(OUT_DIR / "byd_a_daily.csv", raw_rows, FIELDS)

    qfq_fields = FIELDS + ["raw_open", "raw_high", "raw_low", "raw_close", "raw_pre_close", "qfq_factor"]
    write_csv(OUT_DIR / "byd_a_daily_qfq.csv", enriched_qfq_rows, qfq_fields)

    summary = quality_summary(raw_rows, qfq_rows)
    write_csv(
        OUT_DIR / "byd_a_adj_factor.csv",
        summary["factor_rows"],
        ["trade_date", "ts_code", "name", "implied_qfq_factor"],
    )
    summary_for_json = dict(summary)
    summary_for_json["factor_rows"] = f"{len(summary['factor_rows'])} rows written to byd_a_adj_factor.csv"
    (OUT_DIR / "byd_a_quality_summary.json").write_text(
        json.dumps(summary_for_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(OUT_DIR / "byd_a_quality_report.md", summary)

    print(json.dumps(summary_for_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
