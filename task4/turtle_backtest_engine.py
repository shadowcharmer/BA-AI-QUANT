#!/usr/bin/env python3
"""Command line turtle trading backtest engine for TASK4.

Core assumptions match the local dashboard:

- signal is generated after the signal day's close
- order is executed at the next trading day's open
- single-symbol long-only turtle system
- Donchian breakout uses prior N completed days, so today's high/low does not
  leak into the signal threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def to_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", ""))
    except ValueError:
        return default


def to_date(value: str) -> str:
    raw = str(value or "").strip()
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return raw[:10]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        if "date" in row and "trade_date" not in row:
            row["trade_date"] = row["date"]
        if "volume" in row and "vol" not in row:
            row["vol"] = row["volume"]
        row["date"] = to_date(row.get("trade_date", ""))
        for key in ["open", "high", "low", "close", "pre_close", "vol", "amount"]:
            row[key] = to_float(row.get(key))
        if row["date"]:
            out.append(row)
    out.sort(key=lambda r: r["date"])
    for i, row in enumerate(out):
        if row.get("pre_close") is None:
            row["pre_close"] = out[i - 1]["close"] if i else row["close"]
    return out


def rolling_mean(values: list[float | None], window: int) -> list[float | None]:
    result, total, valid = [], 0.0, 0
    for i, value in enumerate(values):
        if value is not None:
            total += value
            valid += 1
        if i >= window:
            old = values[i - window]
            if old is not None:
                total -= old
                valid -= 1
        result.append(total / window if i >= window - 1 and valid == window else None)
    return result


def prior_high(rows: list[dict], window: int) -> list[float | None]:
    out = []
    for i in range(len(rows)):
        if i < window:
            out.append(None)
        else:
            out.append(max(r["high"] for r in rows[i - window:i]))
    return out


def prior_low(rows: list[dict], window: int) -> list[float | None]:
    out = []
    for i in range(len(rows)):
        if i < window:
            out.append(None)
        else:
            out.append(min(r["low"] for r in rows[i - window:i]))
    return out


def add_indicators(rows: list[dict], cfg: dict) -> list[dict]:
    rows = [dict(r) for r in rows]
    tr = []
    for i, row in enumerate(rows):
        prev_close = rows[i - 1]["close"] if i else row["pre_close"]
        tr.append(max(row["high"] - row["low"], abs(row["high"] - prev_close), abs(row["low"] - prev_close)))
    atr = rolling_mean(tr, cfg["atr_window"])
    entry_high = prior_high(rows, cfg["entry_window"])
    exit_low = prior_low(rows, cfg["exit_window"])
    for i, row in enumerate(rows):
        row["atr"] = atr[i]
        row["entry_high"] = entry_high[i]
        row["exit_low"] = exit_low[i]
    return rows


def commission(amount: float, cfg: dict) -> float:
    return max(amount * cfg["commission_rate"], cfg["min_commission"]) if amount > 0 else 0.0


def unit_qty(total_asset: float, atr: float, price: float, cash: float, cfg: dict) -> tuple[int, float]:
    risk_budget = total_asset * cfg["risk_per_unit"]
    raw_qty = risk_budget / atr if atr > 0 else 0
    qty = math.floor(raw_qty / cfg["lot_size"]) * cfg["lot_size"]
    one_lot_cost = price * cfg["lot_size"] + commission(price * cfg["lot_size"], cfg)
    max_cash_qty = math.floor(cash / max(one_lot_cost, 1e-9)) * cfg["lot_size"]
    qty = min(qty, max_cash_qty)
    return max(0, qty), risk_budget


def make_trade(i, pending, row, direction, qty, price, amount, fee, tax, raw_price, cash, shares, units, pnl=None, ret=None):
    return {
        "id": i,
        "signal_date": pending["signal_date"],
        "execute_date": row["date"],
        "direction": direction,
        "price": price,
        "qty": qty,
        "units_after": units,
        "amount": amount,
        "commission": fee,
        "tax": tax,
        "slippage_cost": abs(price - raw_price) * qty,
        "cash_after": cash,
        "shares_after": shares,
        "reason": pending["reason"],
        "pnl": pnl,
        "return_rate": ret,
        **pending.get("context", {}),
    }


def context(row: dict, stop_price=None, next_add_price=None, units=0) -> dict:
    return {
        "close": row.get("close"),
        "entry_high": row.get("entry_high"),
        "exit_low": row.get("exit_low"),
        "atr": row.get("atr"),
        "stop_price": stop_price,
        "next_add_price": next_add_price,
        "units": units,
    }


def run_backtest(rows: list[dict], cfg: dict) -> dict:
    rows = [r for r in rows if cfg["start_date"] <= r["date"] <= cfg["end_date"]]
    data = add_indicators(rows, cfg)
    min_need = max(cfg["entry_window"], cfg["exit_window"], cfg["atr_window"]) + 2
    if len(data) < min_need:
        raise ValueError("Not enough rows in selected date range.")

    cash = cfg["initial_cash"]
    shares = 0
    avg_cost = 0.0
    units = 0
    last_entry_price = 0.0
    stop_price = None
    next_add_price = None
    peak_equity = cfg["initial_cash"]
    pending = None
    trades, signals, equity, closed = [], [], [], []
    first_close = data[0]["close"]

    for i, row in enumerate(data):
        if pending:
            raw_price = row["open"]
            if pending["direction"] == "buy":
                price = raw_price * (1 + cfg["slippage_rate"])
                total_asset_now = cash + shares * price
                qty, risk_budget = unit_qty(total_asset_now, pending["atr"], price, cash, cfg)
                amount = qty * price
                fee = commission(amount, cfg)
                if qty > 0 and amount + fee <= cash and units < cfg["max_units"]:
                    cash -= amount + fee
                    avg_cost = (avg_cost * shares + amount + fee) / (shares + qty) if shares else (amount + fee) / qty
                    shares += qty
                    units += 1
                    last_entry_price = price
                    stop_price = price - cfg["stop_atr_multiple"] * pending["atr"]
                    next_add_price = price + cfg["add_atr_step"] * pending["atr"] if units < cfg["max_units"] else None
                    trades.append(make_trade(len(trades) + 1, pending, row, "buy", qty, price, amount, fee, 0, raw_price, cash, shares, units))
                else:
                    signals.append({
                        "date": pending["signal_date"],
                        "action": "buy_unfilled",
                        "reason": f"cash_or_lot_limit;risk_budget={risk_budget:.2f}",
                        **pending.get("context", {}),
                    })
            elif pending["direction"] == "sell" and shares > 0:
                qty = shares
                price = raw_price * (1 - cfg["slippage_rate"])
                amount = qty * price
                fee = commission(amount, cfg)
                tax = amount * cfg["tax_rate"]
                pnl = amount - fee - tax - avg_cost * qty
                ret = (price - avg_cost) / avg_cost if avg_cost else 0
                cash += amount - fee - tax
                trades.append(make_trade(len(trades) + 1, pending, row, "sell", qty, price, amount, fee, tax, raw_price, cash, 0, 0, pnl, ret))
                closed.append({"pnl": pnl, "return_rate": ret})
                shares = 0
                avg_cost = 0.0
                units = 0
                last_entry_price = 0.0
                stop_price = None
                next_add_price = None
            pending = None

        holding_value = shares * row["close"]
        total_asset = cash + holding_value
        peak_equity = max(peak_equity, total_asset)
        drawdown = total_asset / peak_equity - 1 if peak_equity else 0
        equity.append({
            "date": row["date"],
            "cash": cash,
            "shares": shares,
            "units": units,
            "holding_value": holding_value,
            "total_asset": total_asset,
            "nav": total_asset / cfg["initial_cash"],
            "benchmark": row["close"] / first_close,
            "position_ratio": holding_value / total_asset if total_asset else 0,
            "drawdown": drawdown,
            "stop_price": stop_price,
            "next_add_price": next_add_price,
        })
        if i >= len(data) - 1:
            continue
        if row["atr"] is None or row["entry_high"] is None or row["exit_low"] is None:
            continue

        ctx = context(row, stop_price, next_add_price, units)
        sell_reasons = []
        if shares > 0 and stop_price is not None and row["low"] <= stop_price:
            sell_reasons.append("atr_stop")
        if shares > 0 and row["low"] < row["exit_low"]:
            sell_reasons.append("reverse_breakout_exit")
        if sell_reasons:
            pending = {"direction": "sell", "signal_date": row["date"], "reason": ";".join(sell_reasons), "context": ctx}
            signals.append({"date": row["date"], "action": "sell", "reason": pending["reason"], **ctx})
            continue

        initial_breakout = shares == 0 and row["high"] > row["entry_high"]
        add_breakout = shares > 0 and units < cfg["max_units"] and next_add_price is not None and row["high"] >= next_add_price
        if initial_breakout or add_breakout:
            reason = "donchian_entry_breakout" if initial_breakout else "pyramid_add_0.5_atr"
            pending = {
                "direction": "buy",
                "signal_date": row["date"],
                "reason": reason,
                "atr": row["atr"],
                "context": ctx,
            }
            signals.append({"date": row["date"], "action": "buy" if initial_breakout else "add", "reason": reason, **ctx})

    return {"summary": metrics(equity, trades, closed, cfg), "trades": trades, "signals": signals, "equity": equity}


def metrics(equity: list[dict], trades: list[dict], closed: list[dict], cfg: dict) -> dict:
    final_asset = equity[-1]["total_asset"]
    total_return = final_asset / cfg["initial_cash"] - 1
    annual_return = final_asset / cfg["initial_cash"]
    annual_return = annual_return ** (252 / len(equity)) - 1 if len(equity) > 1 else 0
    benchmark_return = equity[-1]["benchmark"] - 1
    max_drawdown = min((e["drawdown"] for e in equity), default=0)
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    win_rate = len(wins) / len(closed) if closed else 0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss < 0 else None
    daily = [equity[i]["nav"] / equity[i - 1]["nav"] - 1 for i in range(1, len(equity))]
    avg_daily = sum(daily) / len(daily) if daily else 0
    vol = math.sqrt(sum((x - avg_daily) ** 2 for x in daily) / len(daily)) if daily else 0
    sharpe = ((avg_daily - cfg["risk_free_rate"] / 252) / vol) * math.sqrt(252) if vol else None
    return {
        "final_asset": final_asset,
        "total_return": total_return,
        "annual_return": annual_return,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "max_drawdown": max_drawdown,
        "trade_count": len(trades),
        "closed_trade_count": len(closed),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_loss_ratio": profit_loss_ratio,
        "sharpe": sharpe,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def default_config(args) -> dict:
    return {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "initial_cash": args.initial_cash,
        "lot_size": args.lot_size,
        "risk_free_rate": args.risk_free_rate,
        "entry_window": args.entry_window,
        "atr_window": args.atr_window,
        "exit_window": args.exit_window,
        "risk_per_unit": args.risk_per_unit,
        "stop_atr_multiple": args.stop_atr_multiple,
        "add_atr_step": args.add_atr_step,
        "max_units": args.max_units,
        "commission_rate": args.commission_rate,
        "min_commission": args.min_commission,
        "tax_rate": args.tax_rate,
        "slippage_rate": args.slippage_rate,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TASK4 turtle strategy backtest engine")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("task4/exports"))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--entry-window", type=int, default=20)
    parser.add_argument("--atr-window", type=int, default=20)
    parser.add_argument("--exit-window", type=int, default=10)
    parser.add_argument("--risk-per-unit", type=float, default=0.01)
    parser.add_argument("--stop-atr-multiple", type=float, default=2.0)
    parser.add_argument("--add-atr-step", type=float, default=0.5)
    parser.add_argument("--max-units", type=int, default=4)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=0)
    parser.add_argument("--tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_csv(args.csv_path)
    if not rows:
        raise SystemExit("No rows loaded.")
    if not args.start_date:
        args.start_date = rows[0]["date"]
    if not args.end_date:
        args.end_date = rows[-1]["date"]
    cfg = default_config(args)
    result = run_backtest(rows, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "turtle_summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "turtle_trades.csv", result["trades"])
    write_csv(args.out_dir / "turtle_signals.csv", result["signals"])
    write_csv(args.out_dir / "turtle_equity.csv", result["equity"])
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
