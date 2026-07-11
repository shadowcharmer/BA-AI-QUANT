# TASK3 双均线策略回测看板

这是一个单标的双均线策略回测看板，支持在浏览器中交互设置参数、查看交易信号、净值曲线、持仓曲线、绩效指标和交易明细，也提供 Python 命令行脚本用于复现核心回测逻辑。

## 主要功能

- 数据导入与质量检测：检查日期范围、核心 OHLCV 缺失、重复日期、价格关系异常、成交量异常，并提示是否疑似包含复权信息。
- 双均线信号：短均线上穿长均线买入，短均线下穿长均线卖出。
- 成交量过滤：可设置成交量均线窗口和放大倍数，买入时要求成交量条件满足。
- 仓位管理：支持固定金额、固定比例和静态凯利公式；支持初始仓位、加仓仓位、最大仓位上限和加仓间隔。
- 加仓规则：持仓后，只要短均线高于长均线，且短均线和长均线同时上行，并满足间隔 N 个交易日，即触发加仓。
- 风险控制：支持固定比例止损、ATR 止损、移动止损和最大回撤暂停交易。
- 交易成本：支持手续费、最低手续费、卖出税费和滑点模拟。
- 趋势过滤：可要求股价在长期均线上方才允许买入。
- 图表交互：价格、均线、买卖点、净值、持仓均可视化；图例可交互，信号点支持 tooltip。
- 报告导出：支持导出 HTML 报告，也可通过浏览器打印为 PDF。

## 文件结构

```text
task3/
  README.md
  task3_double_ma_backtest_dashboard_spec.md
  double_ma_backtest_dashboard.html
  double_ma_backtest_engine.py
  double_ma_backtest_readme.md
  download_byd_a_data.py
  sample_data/
    smic_a_daily_qfq.csv
    byd_a_daily.csv
    byd_a_daily_qfq.csv
    byd_a_adj_factor.csv
    byd_a_quality_report.md
    byd_a_quality_summary.json
```

## 快速开始

直接用浏览器打开：

```text
task3/double_ma_backtest_dashboard.html
```

页面默认会加载 `sample_data/smic_a_daily_qfq.csv`。也可以手动导入 `sample_data/byd_a_daily_qfq.csv`，用于比亚迪 A 股近 3 年前复权数据回测。

默认参数：

- 初始资金：`100000`
- 短均线：`10`
- 长均线：`30`
- 趋势过滤长期均线：`60`
- 固定比例模式初始仓位：`30%`
- 固定比例模式加仓仓位：`10%`

## 命令行复现

运行中芯国际 A 股示例：

```bash
python3 -B task3/double_ma_backtest_engine.py task3/sample_data/smic_a_daily_qfq.csv --out-dir task3/exports_smic_default
```

运行比亚迪 A 股示例：

```bash
python3 -B task3/double_ma_backtest_engine.py task3/sample_data/byd_a_daily_qfq.csv --out-dir task3/exports_byd_default
```

启用成交量过滤、移动止损、最大回撤控制和趋势过滤：

```bash
python3 -B task3/double_ma_backtest_engine.py task3/sample_data/byd_a_daily_qfq.csv \
  --out-dir task3/exports_byd_filtered \
  --use-volume-filter \
  --use-trailing \
  --use-drawdown-control \
  --use-trend-filter
```

命令行会输出：

- `double_ma_summary.json`
- `double_ma_trades.csv`
- `double_ma_signals.csv`
- `double_ma_equity.csv`

这些输出目录已被 `.gitignore` 排除，可以按需本地重新生成。

## 下载比亚迪数据

如需刷新比亚迪 A 股近 3 年数据：

```bash
python3 -B task3/download_byd_a_data.py
```

脚本会生成：

- `sample_data/byd_a_daily.csv`：原始行情
- `sample_data/byd_a_daily_qfq.csv`：前复权行情，包含原始价字段和 `qfq_factor`
- `sample_data/byd_a_adj_factor.csv`：隐含前复权因子
- `sample_data/byd_a_quality_report.md`：质量检测报告
- `sample_data/byd_a_quality_summary.json`：质量检测摘要

## 数据字段要求

必需字段：

- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `vol`

可选字段：

- `ts_code`
- `name`
- `pre_close`
- `amount`
- `adj_factor`
- `qfq_factor`
- `cum_adjfactor`

如果缺少 `pre_close`，看板和命令行脚本会使用上一交易日收盘价补齐。

## 回测口径

- 信号在第 `t` 日收盘后产生。
- 订单在第 `t+1` 个交易日开盘价成交；遇到周末或休市日会顺延到下一个有数据的交易日。
- 买入价格包含正向滑点，卖出价格包含反向滑点。
- 买入可能受到成交量过滤、趋势过滤、最大仓位和最大回撤暂停影响。
- 卖出不受成交量过滤和趋势过滤影响。
- 死叉、止损、移动止损、最大回撤控制任一触发即可卖出。
- 基准净值采用首日按真实撮合规则满仓买入并持有的口径。

## 注意事项

- 自动复权判断只作为提示，不能替代对数据源复权口径的确认。
- 数据区间太短时，长均线、ATR 或趋势过滤可能无法计算，看板会阻止执行或提示原因。
- HTML 报告可直接保存；PDF 报告通过浏览器打印功能生成。
