# 股票技术指标交互工具使用说明

## 1. 文件说明

目录：`task2/tool/`

- `stock_indicator_tool.html`：工具主网页。
- `stock_indicator_tool_server.py`：本地后端，用于加载默认示例、动态获取数据、调用配置化 MCP/接口。
- `stock_indicator_tool_config.example.json`：配置示例。
- `stock_indicator_tool_spec.md`：设计 spec。
- `stock_indicator_tool_readme.md`：本说明文件。
- `smic_a_daily.csv`：内置 A 股示例数据。
- `smic_hk_daily.csv`：内置港股示例数据。

## 2. 功能概览

工具支持日线数据分析：

- 本地 CSV 导入。
- 通过本地后端动态获取近三年日线数据。
- A股/港股区分。
- 股票中文名显示和补全。
- 数据完整性和合理性校验。
- RSI、MACD、Bollinger Bands、ATR、OBV、ADX、BIAS、Z-score 指标计算。
- 与 question3 一致的原生 SVG 交互图。
- legend 点击隐藏/显示曲线。
- MACD 金叉/死叉、RSI 超买/超卖、布林带突破等关键事件高亮。

目前只支持日线数据，不支持分钟线。

## 3. 本地 CSV 模式

如果只使用本地 CSV，可以直接打开：

`stock_indicator_tool.html`

操作：

1. 左侧“数据来源”选择“本地 CSV 导入”。
2. 点击“选择 CSV 文件”。
3. 选择包含日线数据的 CSV。
4. 点击“校验并生成图表”。

CSV 推荐字段：

- `ts_code`
- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `change`
- `pct_chg`
- `vol`
- `amount`

核心必需字段：

- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `vol`

如果 `pre_close`、`change`、`pct_chg` 缺失，工具会尝试自动补齐，并在 warning 中说明。

## 4. 内置 CSV 示例数据

tool 目录中保留了两份示例 CSV：

- `task2/tool/smic_a_daily.csv`
- `task2/tool/smic_hk_daily.csv`

需要使用示例数据时，请在左侧“本地 CSV 导入”中点击“选择 CSV 文件”，手动选择上述 CSV。

## 5. 启动本地后端

动态获取股票代码需要本地后端。本地 CSV 导入不需要后端。

在当前作业目录运行：

```bash
python3 task2/tool/stock_indicator_tool_server.py
```

默认地址：

```text
http://127.0.0.1:8765/stock_indicator_tool.html
```

如果要指定端口：

```bash
python3 task2/tool/stock_indicator_tool_server.py 8899
```

然后在浏览器打开：

```text
http://127.0.0.1:8899/stock_indicator_tool.html
```

## 6. 动态获取配置

后端默认读取：

1. `stock_indicator_tool_config.json`
2. 如果不存在，则读取 `stock_indicator_tool_config.example.json`

正式联调时建议复制一份：

```bash
cp task2/tool/stock_indicator_tool_config.example.json task2/tool/stock_indicator_tool_config.json
```

然后修改 `stock_indicator_tool_config.json`。

## 7. MCP/接口 Profile

配置文件中每个 profile 表示一个数据来源。

### 7.1 local_example

```json
{
  "kind": "local_example"
}
```

只读取本地中芯国际示例，不调用网络。

### 7.2 HTTP 模式

```json
{
  "kind": "http",
  "url": "http://127.0.0.1:9000/fetch_daily",
  "method": "POST",
  "tool": "daily",
  "headers": {},
  "params": {}
}
```

后端会 POST：

```json
{
  "market": "A股",
  "ts_code": "688981.SH",
  "start_date": "20230705",
  "end_date": "20260704",
  "tool": "daily",
  "params": {}
}
```

接口应返回：

```json
{
  "ok": true,
  "market": "A股",
  "ts_code": "688981.SH",
  "name": "中芯国际",
  "rows": []
}
```

`rows` 中字段应尽量使用标准字段。

### 7.3 Command 模式

```json
{
  "kind": "command",
  "command": [
    "python3",
    "/absolute/path/to/your_mcp_fetch_daily_adapter.py"
  ],
  "tool": "daily",
  "params": {}
}
```

后端会把请求 JSON 写入命令的 stdin，并读取 stdout JSON。

这种模式适合把你现有的 MCP 调用封装成一个本地 adapter。

## 8. A股和港股代码

A股：

- 使用完整格式，如 `688981.SH`、`600000.SH`、`000001.SZ`。
- 不建议只输入 6 位数字，因为无法可靠判断交易所。

港股：

- 使用完整格式，如 `00981.HK`。
- 输入 `981` 时，工具会补全为 `00981.HK`。

中文名：

- 后端如果返回 `name`，网页会显示该名称。
- 本地中芯国际示例已内置 `中芯国际`。
- 无法补全时显示“未知名称”，不阻断分析。

## 9. 数据校验

无论本地还是线上，都会先校验。

严重错误会停止执行：

- 文件无法读取。
- 接口超时。
- 股票代码格式错误。
- 市场和代码不匹配。
- 数据为空。
- 缺少核心字段。
- 核心字段无法转换为数字。
- 价格字段小于等于 0。
- 成交量为负。
- 日期重复。
- 样本少于 60 个交易日。
- 收盘价全部相同。

轻微问题会继续执行并提示 warning：

- `pre_close` 缺失但可补齐。
- `change` 缺失但可计算。
- `pct_chg` 缺失但可计算。
- 中文名无法补全。

## 10. 常见问题

### 动态获取接口无法连接

确认后端是否启动。启动后访问：

```text
http://127.0.0.1:8765/health
```

应返回：

```json
{"ok": true}
```

然后打开：

```text
http://127.0.0.1:8765/stock_indicator_tool.html
```

### 如何使用示例 CSV

在网页左侧选择“本地 CSV 导入”，点击“选择 CSV 文件”，手动选择：

- `task2/tool/smic_a_daily.csv`
- `task2/tool/smic_hk_daily.csv`

### 动态获取失败

检查：

- `stock_indicator_tool_config.json` 是否存在。
- profile 名称是否和网页选择一致。
- HTTP URL 或 command 路径是否正确。
- MCP 服务是否可用。
- 超时时间是否过短。
- 返回 JSON 是否包含 `rows` 或 `data`。

### 页面能打开但图表不生成

查看左侧“状态”和“校验结果”。工具设计为校验失败即停止，不会生成不可信图表。

## 11. 当前限制

- 只支持日线数据。
- 静态 HTML 无法直接调用 MCP，动态获取必须通过本地后端。
- 默认配置只提供本地中芯国际示例；真实 MCP 联调需要填写 `stock_indicator_tool_config.json`。
- 图表使用原生 SVG，不依赖外部 CDN。
