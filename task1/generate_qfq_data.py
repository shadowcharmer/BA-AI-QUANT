# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


BASE = Path(__file__).resolve().parent
PRICE_COLS = ["open", "high", "low", "close"]
STANDARD_COLS = [
    "ts_code",
    "trade_date",
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

DATASETS = [
    {
        "market": "A股",
        "code": "688981.SH",
        "input": BASE / "smic_a_daily.csv",
        "output": BASE / "smic_a_daily_qfq.csv",
        "factor_output": BASE / "smic_a_adj_factor.csv",
        "factor_name": "adj_factor",
        "factor_source": "Tushare adj_factor, fetched 2026-07-10; all returned factors in 2025-07-02 to 2026-07-02 are 1.0.",
        "factor_status": "verified_identity",
    },
    {
        "market": "港股",
        "code": "00981.HK",
        "input": BASE / "smic_hk_daily.csv",
        "output": BASE / "smic_hk_daily_qfq.csv",
        "factor_output": BASE / "smic_hk_adj_factor.csv",
        "factor_name": "cum_adjfactor",
        "factor_source": "Tushare hk_adjfactor/hk_daily_adj returned permission error 40203 on 2026-07-10; identity factors are used and recorded for reproducibility.",
        "factor_status": "identity_fallback_no_adjustment_applied",
    },
]


def quantize(value: Decimal, places: str = "0.000001") -> str:
    text = str(value.quantize(Decimal(places), rounding=ROUND_HALF_UP))
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def decimal_value(row: dict[str, str], key: str) -> Decimal:
    return Decimal(str(row[key]))


def build_identity_factors(rows: list[dict[str, str]], code: str, factor_name: str) -> list[dict[str, str]]:
    return [
        {"ts_code": code, "trade_date": row["trade_date"], factor_name: "1.0"}
        for row in rows
    ]


def apply_qfq(dataset: dict[str, object]) -> dict[str, object]:
    rows = read_rows(dataset["input"])  # type: ignore[index]
    rows.sort(key=lambda row: row["trade_date"])
    factor_name = str(dataset["factor_name"])
    factors = build_identity_factors(rows, str(dataset["code"]), factor_name)
    factor_by_date = {
        item["trade_date"]: Decimal(item[factor_name])
        for item in factors
    }
    latest_factor = factor_by_date[rows[-1]["trade_date"]]
    all_identity = all(factor == latest_factor for factor in factor_by_date.values())

    out_rows: list[dict[str, str]] = []
    previous_qfq_close: Decimal | None = None
    for row in rows:
        raw = {col: row[col] for col in STANDARD_COLS}
        factor = factor_by_date[row["trade_date"]]
        qfq_factor = factor / latest_factor

        out = dict(row)
        for col in PRICE_COLS + ["pre_close"]:
            out[f"raw_{col}"] = row[col]
        out[factor_name] = quantize(factor)
        out["qfq_factor"] = quantize(qfq_factor)

        if all_identity:
            for col in PRICE_COLS + ["pre_close", "change", "pct_chg"]:
                out[col] = raw[col]
            qfq_close = decimal_value(raw, "close")
        else:
            for col in PRICE_COLS:
                out[col] = quantize(decimal_value(raw, col) * qfq_factor, "0.000001")

            if previous_qfq_close is None:
                qfq_pre_close = decimal_value(raw, "pre_close") * qfq_factor
            else:
                qfq_pre_close = previous_qfq_close
            qfq_close = Decimal(out["close"])
            out["pre_close"] = quantize(qfq_pre_close, "0.000001")
            out["change"] = quantize(qfq_close - qfq_pre_close, "0.000001")
            out["pct_chg"] = quantize(
                (qfq_close - qfq_pre_close) / qfq_pre_close * Decimal("100"),
                "0.0001",
            )
        previous_qfq_close = qfq_close
        out_rows.append(out)

    fieldnames = (
        STANDARD_COLS
        + [factor_name, "qfq_factor"]
        + [f"raw_{col}" for col in PRICE_COLS + ["pre_close"]]
    )
    write_csv(dataset["output"], out_rows, fieldnames)  # type: ignore[index]
    write_csv(dataset["factor_output"], factors, ["ts_code", "trade_date", factor_name])  # type: ignore[index]

    changed_rows = sum(
        1 for old, new in zip(rows, out_rows)
        if any(
            abs(Decimal(str(old[col])) - Decimal(str(new[col]))) > Decimal("0.000001")
            for col in PRICE_COLS + ["pre_close", "change", "pct_chg"]
        )
    )
    return {
        "market": dataset["market"],
        "code": dataset["code"],
        "input": Path(dataset["input"]).name,  # type: ignore[arg-type]
        "output": Path(dataset["output"]).name,  # type: ignore[arg-type]
        "factor_file": Path(dataset["factor_output"]).name,  # type: ignore[arg-type]
        "rows": len(out_rows),
        "start_date": out_rows[0]["trade_date"],
        "end_date": out_rows[-1]["trade_date"],
        "latest_factor": quantize(latest_factor),
        "unique_factor_count": len({item[factor_name] for item in factors}),
        "changed_rows": changed_rows,
        "factor_status": dataset["factor_status"],
        "factor_source": dataset["factor_source"],
    }


def main() -> None:
    metadata = [apply_qfq(dataset) for dataset in DATASETS]
    (BASE / "qfq_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
