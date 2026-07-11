#!/usr/bin/env python3
"""Command line double moving-average backtest engine for TASK3.

The dashboard is the primary deliverable. This script provides a reproducible
CLI implementation of the same core assumptions:

- signal is generated after the signal day's close
- order is executed at the next trading day's open
- single-symbol backtest
- Kelly sizing uses the first M completed trades once available
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
    result = []
    total = 0.0
    valid = 0
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


def add_indicators(rows: list[dict], cfg: dict) -> list[dict]:
    rows = [dict(r) for r in rows]
    close = [r["close"] for r in rows]
    vol = [r["vol"] for r in rows]
    short_ma = rolling_mean(close, cfg["short_window"])
    long_ma = rolling_mean(close, cfg["long_window"])
    volume_ma = rolling_mean(vol, cfg["volume_window"])
    trend_ma = rolling_mean(close, cfg["trend_window"])
    tr = []
    for i, row in enumerate(rows):
        prev_close = rows[i - 1]["close"] if i else row["pre_close"]
        tr.append(max(row["high"] - row["low"], abs(row["high"] - prev_close), abs(row["low"] - prev_close)))
    atr = rolling_mean(tr, cfg["atr_window"])
    for i, row in enumerate(rows):
        row["short_ma"] = short_ma[i]
        row["long_ma"] = long_ma[i]
        row["volume_ma"] = volume_ma[i]
        row["volume_ratio"] = row["vol"] / volume_ma[i] if volume_ma[i] else None
        row["trend_ma"] = trend_ma[i]
        row["atr"] = atr[i]
    return rows


def commission(amount: float, cfg: dict) -> float:
    return max(amount * cfg["commission_rate"], cfg["min_commission"]) if amount > 0 else 0.0


def buy_budget(total_asset: float, cash: float, closed: list[dict], cfg: dict, is_add: bool) -> float:
    if not cfg["use_position"]:
        return cash
    if cfg["position_mode"] == "fixed_amount":
        return cfg["add_amount"] if is_add else cfg["initial_amount"]
    if cfg["position_mode"] == "kelly":
        if len(closed) < cfg["kelly_min_trades"]:
            return total_asset * cfg["initial_ratio"]
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        win_rate = len(wins) / len(closed) if closed else 0
        avg_win = sum(t["return_rate"] for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t["return_rate"] for t in losses) / len(losses)) if losses else 0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0
        kelly = win_rate - (1 - win_rate) / payoff if payoff > 0 else 0
        return max(0.0, total_asset * kelly * cfg["kelly_scale"])
    return total_asset * (cfg["add_ratio"] if is_add else cfg["initial_ratio"])


def run_backtest(rows: list[dict], cfg: dict) -> dict:
    rows = [r for r in rows if cfg["start_date"] <= r["date"] <= cfg["end_date"]]
    data = add_indicators(rows, cfg)
    if len(data) < cfg["long_window"] + 2:
        raise ValueError("Not enough rows in selected date range.")

    cash = cfg["initial_cash"]
    shares = 0
    avg_cost = 0.0
    high_since_entry = 0.0
    peak_equity = cfg["initial_cash"]
    paused = False
    pending = None
    last_add_signal_index = -10**9
    trades = []
    signals = []
    equity = []
    closed = []
    first_close = data[0]["close"]

    for i, row in enumerate(data):
        if pending:
            raw_price = row["open"]
            if pending["direction"] == "buy":
                price = raw_price * (1 + cfg["slippage_rate"])
                total_asset_now = cash + shares * price
                max_holding = total_asset_now * cfg["max_position"]
                room = max(0.0, max_holding - shares * price)
                budget = min(pending["budget"], cash, room)
                qty = math.floor((budget / price) / cfg["lot_size"]) * cfg["lot_size"]
                amount = qty * price
                fee = commission(amount, cfg)
                if qty > 0 and amount + fee <= cash:
                    avg_cost = (avg_cost * shares + amount + fee) / (shares + qty) if shares else (amount + fee) / qty
                    cash -= amount + fee
                    shares += qty
                    high_since_entry = max(high_since_entry, row["high"])
                    trades.append(make_trade(len(trades) + 1, pending, row, "buy", qty, price, amount, fee, 0, raw_price, cash, shares))
                    if pending.get("is_add"):
                        last_add_signal_index = pending["signal_index"]
                else:
                    one_lot_cost = price * cfg["lot_size"] + commission(price * cfg["lot_size"], cfg)
                    signals.append({
                        "date": pending["signal_date"],
                        "action": "add_unfilled" if pending.get("is_add") else "buy_unfilled",
                        "reason": f"budget_or_lot_limit;budget={budget:.2f};one_lot_cost={one_lot_cost:.2f}",
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
                trades.append(make_trade(len(trades) + 1, pending, row, "sell", qty, price, amount, fee, tax, raw_price, cash, 0, pnl, ret))
                closed.append({"pnl": pnl, "return_rate": ret})
                shares = 0
                avg_cost = 0.0
                high_since_entry = 0.0
            pending = None

        if shares > 0:
            high_since_entry = max(high_since_entry, row["high"])
        holding_value = shares * row["close"]
        total_asset = cash + holding_value
        peak_equity = max(peak_equity, total_asset)
        drawdown = total_asset / peak_equity - 1 if peak_equity else 0
        if paused and total_asset >= peak_equity * cfg["recover_ratio"]:
            paused = False
        equity.append({
            "date": row["date"],
            "cash": cash,
            "shares": shares,
            "holding_value": holding_value,
            "total_asset": total_asset,
            "nav": total_asset / cfg["initial_cash"],
            "benchmark": row["close"] / first_close,
            "position_ratio": holding_value / total_asset if total_asset else 0,
            "drawdown": drawdown,
        })
        if i == 0 or i >= len(data) - 1:
            continue

        prev = data[i - 1]
        cross_up = prev["short_ma"] is not None and prev["long_ma"] is not None and row["short_ma"] is not None and row["long_ma"] is not None and prev["short_ma"] <= prev["long_ma"] and row["short_ma"] > row["long_ma"]
        cross_down = prev["short_ma"] is not None and prev["long_ma"] is not None and row["short_ma"] is not None and row["long_ma"] is not None and prev["short_ma"] >= prev["long_ma"] and row["short_ma"] < row["long_ma"]
        short_above_long = shares > 0 and row["short_ma"] is not None and row["long_ma"] is not None and row["short_ma"] > row["long_ma"]
        short_keeps_rising = shares > 0 and prev["short_ma"] is not None and row["short_ma"] is not None and row["short_ma"] > prev["short_ma"]
        long_keeps_rising = shares > 0 and prev["long_ma"] is not None and row["long_ma"] is not None and row["long_ma"] > prev["long_ma"]
        add_interval_ok = shares > 0 and (i - last_add_signal_index >= cfg["add_interval_days"])
        add_signal = short_above_long and short_keeps_rising and long_keeps_rising

        sell_reasons = []
        if shares > 0 and cfg["use_drawdown_control"] and drawdown <= -cfg["max_drawdown_limit"]:
            sell_reasons.append("max_drawdown")
            paused = True
        if shares > 0 and cfg["use_risk"]:
            if cfg["stop_mode"] == "fixed" and row["close"] <= avg_cost * (1 - cfg["stop_pct"]):
                sell_reasons.append("fixed_stop")
            if cfg["stop_mode"] == "atr" and row["atr"] is not None and row["close"] <= avg_cost - cfg["atr_multiple"] * row["atr"]:
                sell_reasons.append("atr_stop")
        if shares > 0 and cfg["use_trailing"] and high_since_entry and row["close"] / high_since_entry - 1 <= -cfg["trailing_pct"]:
            sell_reasons.append("trailing_stop")
        if shares > 0 and cross_down:
            sell_reasons.append("death_cross")
        if sell_reasons:
            pending = {"direction": "sell", "signal_date": row["date"], "reason": ";".join(sell_reasons), "context": context(row)}
            signals.append({"date": row["date"], "action": "sell", "reason": pending["reason"], **context(row)})
            continue

        initial_buy_signal = shares == 0 and cross_up
        buy_signal = initial_buy_signal or add_signal
        if buy_signal:
            blocks = []
            if paused:
                blocks.append("paused_by_drawdown")
            if cfg["use_volume_filter"] and not (row["volume_ma"] is not None and row["vol"] >= row["volume_ma"] and row["volume_ratio"] >= cfg["volume_multiplier"]):
                blocks.append("volume_filter_failed")
            if cfg["use_trend_filter"] and not (row["trend_ma"] is not None and row["close"] > row["trend_ma"]):
                blocks.append("trend_filter_failed")
            if add_signal and not cfg["allow_add"]:
                blocks.append("already_holding")
            if add_signal and not add_interval_ok:
                blocks.append("add_interval_not_met")
            signal_reason = "golden_cross" if initial_buy_signal else "add_short_above_long_both_rising"
            if blocks:
                signals.append({"date": row["date"], "action": "buy_filtered" if initial_buy_signal else "add_filtered", "reason": signal_reason + ";" + ";".join(blocks), **context(row)})
            else:
                budget = buy_budget(total_asset, cash, closed, cfg, shares > 0)
                pending = {"direction": "buy", "signal_date": row["date"], "signal_index": i, "is_add": add_signal, "reason": signal_reason, "budget": budget, "context": context(row)}
                signals.append({"date": row["date"], "action": "buy" if initial_buy_signal else "add", "reason": signal_reason, **context(row)})

    return {"summary": metrics(equity, trades, closed, cfg), "trades": trades, "signals": signals, "equity": equity}


def context(row: dict) -> dict:
    return {key: row.get(key) for key in ["close", "short_ma", "long_ma", "volume_ma", "volume_ratio", "trend_ma", "atr"]}


def make_trade(i, pending, row, direction, qty, price, amount, fee, tax, raw_price, cash, shares, pnl=None, ret=None):
    return {
        "id": i,
        "signal_date": pending["signal_date"],
        "execute_date": row["date"],
        "direction": direction,
        "price": price,
        "qty": qty,
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
        "short_window": args.short_window,
        "long_window": args.long_window,
        "use_volume_filter": args.use_volume_filter,
        "volume_window": args.volume_window,
        "volume_multiplier": args.volume_multiplier,
        "use_position": True,
        "position_mode": args.position_mode,
        "initial_amount": args.initial_amount,
        "add_amount": args.add_amount,
        "initial_ratio": args.initial_ratio,
        "add_ratio": args.add_ratio,
        "add_interval_days": args.add_interval_days,
        "kelly_min_trades": args.kelly_min_trades,
        "kelly_scale": args.kelly_scale,
        "max_position": args.max_position,
        "allow_add": not args.no_add,
        "use_risk": True,
        "stop_mode": args.stop_mode,
        "stop_pct": args.stop_pct,
        "atr_window": args.atr_window,
        "atr_multiple": args.atr_multiple,
        "use_trailing": args.use_trailing,
        "trailing_pct": args.trailing_pct,
        "use_drawdown_control": args.use_drawdown_control,
        "max_drawdown_limit": args.max_drawdown_limit,
        "recover_ratio": args.recover_ratio,
        "commission_rate": args.commission_rate,
        "min_commission": args.min_commission,
        "tax_rate": args.tax_rate,
        "slippage_rate": args.slippage_rate,
        "use_trend_filter": args.use_trend_filter,
        "trend_window": args.trend_window,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TASK3 double MA backtest engine")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("task3/exports"))
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--initial-cash", type=float, default=100_000)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--risk-free-rate", type=float, default=0)
    parser.add_argument("--short-window", type=int, default=10)
    parser.add_argument("--long-window", type=int, default=30)
    parser.add_argument("--use-volume-filter", action="store_true")
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--volume-multiplier", type=float, default=1.5)
    parser.add_argument("--position-mode", choices=["fixed_ratio", "fixed_amount", "kelly"], default="fixed_ratio")
    parser.add_argument("--initial-amount", type=float, default=30_000)
    parser.add_argument("--add-amount", type=float, default=10_000)
    parser.add_argument("--initial-ratio", type=float, default=0.3)
    parser.add_argument("--add-ratio", type=float, default=0.1)
    parser.add_argument("--add-interval-days", type=int, default=5)
    parser.add_argument("--kelly-min-trades", type=int, default=20)
    parser.add_argument("--kelly-scale", type=float, default=0.5)
    parser.add_argument("--max-position", type=float, default=1.0)
    parser.add_argument("--no-add", action="store_true")
    parser.add_argument("--stop-mode", choices=["fixed", "atr"], default="fixed")
    parser.add_argument("--stop-pct", type=float, default=0.08)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--atr-multiple", type=float, default=2)
    parser.add_argument("--use-trailing", action="store_true")
    parser.add_argument("--trailing-pct", type=float, default=0.10)
    parser.add_argument("--use-drawdown-control", action="store_true")
    parser.add_argument("--max-drawdown-limit", type=float, default=0.20)
    parser.add_argument("--recover-ratio", type=float, default=0.95)
    parser.add_argument("--commission-rate", type=float, default=0.0003)
    parser.add_argument("--min-commission", type=float, default=0)
    parser.add_argument("--tax-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0002)
    parser.add_argument("--use-trend-filter", action="store_true")
    parser.add_argument("--trend-window", type=int, default=60)
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
    (args.out_dir / "double_ma_summary.json").write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "double_ma_trades.csv", result["trades"])
    write_csv(args.out_dir / "double_ma_signals.csv", result["signals"])
    write_csv(args.out_dir / "double_ma_equity.csv", result["equity"])
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
