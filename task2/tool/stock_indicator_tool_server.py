#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = ROOT / "task2" / "tool"
TASK1 = ROOT / "task1"
CONFIG_PATH = TOOL_DIR / "stock_indicator_tool_config.json"
EXAMPLE_CONFIG_PATH = TOOL_DIR / "stock_indicator_tool_config.example.json"

BUILTIN_NAMES = {
    "688981.SH": "中芯国际",
    "00981.HK": "中芯国际",
}

REQUIRED = ["trade_date", "open", "high", "low", "close", "vol"]
NUMERIC = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]


def load_config() -> dict[str, Any]:
    path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    if not path.exists():
        return {"profiles": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def error_payload(code: str, message: str, details: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    return {"ok": False, "error_code": code, "message": message, "details": details or {}}


def normalize_code(market: str, code: str) -> str:
    code = code.strip().upper()
    if market == "A股":
        if re.fullmatch(r"\d{6}", code):
            raise ValueError("A股仅输入 6 位数字时无法可靠判断交易所，请使用 688981.SH 或 000001.SZ 格式。")
        if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code):
            raise ValueError("A股代码格式错误，应类似 688981.SH、600000.SH、000001.SZ。")
        return code
    if market == "港股":
        if re.fullmatch(r"\d{1,5}", code):
            code = code.zfill(5) + ".HK"
        if not re.fullmatch(r"\d{5}\.HK", code):
            raise ValueError("港股代码格式错误，应类似 00981.HK；输入 981 时会补全为 00981.HK。")
        return code
    raise ValueError("市场类型错误，只支持 A股 或 港股。")


def parse_date(value: Any) -> str:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return dt.datetime.strptime(text, "%Y%m%d").strftime("%Y-%m-%d")
    return dt.datetime.fromisoformat(text[:10]).strftime("%Y-%m-%d")


def to_float(value: Any, field: str, row_idx: int) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"第 {row_idx} 行字段 {field} 为空。")
    return float(value)


def normalize_rows(rows: list[dict[str, Any]], market: str, code: str, name: Optional[str] = None) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not rows:
        raise ValueError("数据为空。")

    aliases = {
        "date": "trade_date",
        "datetime": "trade_date",
        "tradedate": "trade_date",
        "volume": "vol",
        "turnover": "amount",
        "value": "amount",
    }
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(rows, start=1):
        row = {str(k).strip().replace("\ufeff", ""): v for k, v in raw.items()}
        lowered = {k.lower().replace("_", ""): k for k in row}
        for alias, target in aliases.items():
            if target not in row and alias in lowered:
                row[target] = row[lowered[alias]]
        missing = [field for field in REQUIRED if field not in row]
        if missing:
            raise ValueError(f"缺少核心字段：{', '.join(missing)}。")
        out: dict[str, Any] = {
            "ts_code": str(row.get("ts_code") or code).strip().upper(),
            "name": str(row.get("name") or row.get("stock_name") or row.get("security_name") or name or BUILTIN_NAMES.get(code, "未知名称")),
            "trade_date": parse_date(row["trade_date"]),
        }
        for field in NUMERIC:
            if field in row and str(row[field]).strip() != "":
                out[field] = to_float(row[field], field, idx)
        for field in ["open", "high", "low", "close", "vol"]:
            if field not in out:
                raise ValueError(f"第 {idx} 行缺少核心字段 {field}。")
        normalized.append(out)

    normalized.sort(key=lambda item: item["trade_date"])
    seen: set[str] = set()
    for idx, row in enumerate(normalized):
        if row["trade_date"] in seen:
            raise ValueError(f"交易日期重复：{row['trade_date']}。")
        seen.add(row["trade_date"])
        if row["high"] < row["low"]:
            raise ValueError(f"{row['trade_date']} high 小于 low。")
        if min(row["open"], row["high"], row["low"], row["close"]) <= 0:
            raise ValueError(f"{row['trade_date']} 存在非正价格。")
        if row["vol"] < 0:
            raise ValueError(f"{row['trade_date']} 成交量为负。")
        if row.get("amount", 0) < 0:
            raise ValueError(f"{row['trade_date']} 成交额为负。")
        if idx == 0 and "pre_close" not in row:
            row["pre_close"] = row["close"]
            warnings.append("首行 pre_close 缺失，已用当日 close 补齐。")
        if idx > 0 and "pre_close" not in row:
            row["pre_close"] = normalized[idx - 1]["close"]
            if "pre_close" not in warnings:
                warnings.append("pre_close 缺失，已用上一交易日 close 补齐。")
        if "change" not in row:
            row["change"] = row["close"] - row["pre_close"]
        if "pct_chg" not in row:
            row["pct_chg"] = row["change"] / row["pre_close"] * 100 if row["pre_close"] else 0.0
        if "amount" not in row:
            row["amount"] = 0.0

    if len(normalized) < 60:
        raise ValueError(f"样本数量不足：仅 {len(normalized)} 行，至少需要 60 个交易日。")
    closes = {row["close"] for row in normalized}
    if len(closes) <= 1:
        raise ValueError("close 全部相同，无法进行有效技术分析。")
    return normalized, sorted(set(warnings))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(text.splitlines()))


def example_rows(market: str) -> tuple[str, Path]:
    if market == "A股":
        return "688981.SH", TASK1 / "smic_a_daily_qfq.csv"
    if market == "港股":
        return "00981.HK", TASK1 / "smic_hk_daily_qfq.csv"
    raise ValueError("市场类型错误。")


def fetch_via_http(profile: dict[str, Any], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = profile.get("url")
    if not url:
        raise ValueError("HTTP MCP profile 缺少 url。")
    headers = {"Content-Type": "application/json"}
    headers.update(profile.get("headers", {}))
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method=profile.get("method", "POST"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_via_command(profile: dict[str, Any], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    command = profile.get("command")
    if not isinstance(command, list) or not command:
        raise ValueError("command MCP profile 需要 command 数组。")
    proc = subprocess.run(
        command,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"命令退出码 {proc.returncode}")
    return json.loads(proc.stdout)


def fetch_dynamic(payload: dict[str, Any]) -> dict[str, Any]:
    market = payload.get("market", "")
    code = normalize_code(market, payload.get("ts_code", ""))
    timeout = float(payload.get("timeout", 30))
    profile_name = payload.get("profile", "local_example")
    config = load_config()
    profile = config.get("profiles", {}).get(profile_name)
    if not profile:
        raise ValueError(f"未找到 profile：{profile_name}。请检查 stock_indicator_tool_config.json。")

    today = dt.date.today()
    start_date = payload.get("start_date") or (today - dt.timedelta(days=366 * 3)).strftime("%Y%m%d")
    end_date = payload.get("end_date") or today.strftime("%Y%m%d")
    req_payload = {
        "market": market,
        "ts_code": code,
        "start_date": start_date,
        "end_date": end_date,
        "tool": profile.get("tool", ""),
        "params": profile.get("params", {}),
    }
    kind = profile.get("kind")
    if kind == "local_example":
        example_code, path = example_rows(market)
        if code != example_code:
            raise ValueError(f"local_example 仅提供 {example_code} 示例，当前请求 {code}。")
        rows = read_csv_rows(path)
        name = BUILTIN_NAMES.get(code, "中芯国际")
        normalized, warnings = normalize_rows(rows, market, code, name)
        return {"ok": True, "market": market, "ts_code": code, "name": name, "source": "local_example", "rows": normalized, "warnings": warnings}
    if kind == "http":
        raw = fetch_via_http(profile, req_payload, timeout)
    elif kind == "command":
        raw = fetch_via_command(profile, req_payload, timeout)
    else:
        raise ValueError(f"profile kind 不支持：{kind}。支持 local_example/http/command。")

    if raw.get("ok") is False:
        return raw
    rows = raw.get("rows") or raw.get("data") or []
    name = raw.get("name") or BUILTIN_NAMES.get(code, "未知名称")
    normalized, warnings = normalize_rows(rows, market, code, name)
    return {"ok": True, "market": market, "ts_code": code, "name": name, "source": profile_name, "rows": normalized, "warnings": warnings + raw.get("warnings", [])}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[tool-server] " + fmt % args + "\n")

    def do_OPTIONS(self) -> None:
        json_response(self, 200, {"ok": True})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            json_response(self, 200, {"ok": True, "service": "stock_indicator_tool_server"})
            return
        if path == "/config":
            config = load_config()
            profiles = {name: {"kind": p.get("kind"), "label": p.get("label", name)} for name, p in config.get("profiles", {}).items()}
            json_response(self, 200, {"ok": True, "profiles": profiles})
            return
        if path == "/" or path == "/stock_indicator_tool.html":
            file_path = TOOL_DIR / "stock_indicator_tool.html"
        else:
            file_path = TOOL_DIR / path.lstrip("/")
        if not file_path.exists() or not file_path.is_file() or TOOL_DIR not in file_path.resolve().parents:
            self.send_error(404)
            return
        content = file_path.read_bytes()
        content_type = "text/html; charset=utf-8" if file_path.suffix == ".html" else "text/plain; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except json.JSONDecodeError:
            json_response(self, 400, error_payload("BAD_JSON", "请求 JSON 无法解析。"))
            return
        try:
            if self.path == "/lookup_name":
                code = normalize_code(payload.get("market", ""), payload.get("ts_code", ""))
                json_response(self, 200, {"ok": True, "ts_code": code, "name": BUILTIN_NAMES.get(code, "未知名称")})
                return
            if self.path == "/fetch_daily":
                json_response(self, 200, fetch_dynamic(payload))
                return
            json_response(self, 404, error_payload("NOT_FOUND", "接口不存在。"))
        except subprocess.TimeoutExpired:
            json_response(self, 504, error_payload("TIMEOUT", "动态获取数据超时，请检查 MCP server 或网络连接。"))
        except urllib.error.URLError as exc:
            json_response(self, 502, error_payload("MCP_UNAVAILABLE", "HTTP MCP 请求失败。", {"reason": str(exc)}))
        except ValueError as exc:
            json_response(self, 400, error_payload("DATA_VALIDATION_ERROR", str(exc)))
        except Exception as exc:
            json_response(self, 500, error_payload("SERVER_ERROR", str(exc)))


def main() -> None:
    host = "127.0.0.1"
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving stock indicator tool at http://{host}:{port}/stock_indicator_tool.html")
    server.serve_forever()


if __name__ == "__main__":
    main()
