# TASK3 双均线策略回测看板使用说明

## 文件说明

- `task3_double_ma_backtest_dashboard_spec.md`：确认后的需求 spec。
- `double_ma_backtest_dashboard.html`：本地交互式回测看板，直接用浏览器打开。
- `double_ma_backtest_engine.py`：命令行回测脚本，用于复现核心逻辑并导出 CSV。
- `sample_data/smic_a_daily_qfq.csv`：从 TASK1 迁移过来的中芯国际 A 股前复权示例数据，HTML 默认自动导入。
- `exports/`：命令行脚本默认输出目录，运行后自动生成。

## 快速开始

1. 用浏览器打开 `task3/double_ma_backtest_dashboard.html`。
2. 页面会默认导入 `task3/sample_data/smic_a_daily_qfq.csv`。也可以在左侧选择其他 CSV 文件，例如：
   - `task1/smic_a_daily_qfq.csv`
   - `task1/smic_hk_daily_qfq.csv`
3. 检查页面顶部的数据质量与复权提示。
4. 设置回测时间范围、初始资金、均线周期、仓位、风控和成本参数。
5. 点击“执行回测”。
6. 在右侧查看图表、绩效指标、交易明细、信号日志和参数快照。
7. 点击“导出 HTML 报告”保存报告，或点击“打印 / 导出 PDF”通过浏览器打印为 PDF。

## 数据格式

必需字段：

- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `vol`

可选字段：

- `ts_code`
- `pre_close`
- `amount`
- `adj_factor`
- `qfq_factor`
- `cum_adjfactor`

如果缺少 `pre_close`，看板和脚本会用上一交易日收盘价补齐。

## 回测口径

- 首版只支持单股票单标的回测。
- 默认初始资金为 `100,000`。
- 固定金额模式下，默认初始买入金额为 `30,000`，默认加仓金额为 `10,000`。
- 默认短均线为 `10`，长均线为 `30`。
- 默认趋势过滤周期为 `60`。
- 加仓规则：初始建仓仍由金叉触发；已持仓后，只有短均线高于长均线，且短均线和长均线当天都比前一交易日上行时，才触发加仓。
- 加仓间隔默认 `5` 个交易日；两次加仓信号之间必须至少间隔 N 个交易日。
- 信号在第 `t` 日收盘后产生。
- 订单在第 `t+1` 个交易日开盘价成交；如果中间是周末或休市日，会顺延到下一个有数据的交易日。
- 买入价包含正向滑点，卖出价包含反向滑点。
- 买入受成交量过滤、趋势过滤和最大回撤暂停影响。
- 卖出不受成交量过滤和趋势过滤影响。
- 死叉、固定比例止损、ATR 止损、移动止损、最大回撤控制任一触发即可卖出。
- 最大回撤控制触发后清仓并暂停新买入，直到净值恢复至历史高点的设定比例。
- 凯利公式不要求手动输入胜率和盈亏比，使用前 `M` 笔已完成交易估计；不足 `M` 笔时使用固定初始仓位比例。

## 命令行复现

示例：

```bash
python3 -B task3/double_ma_backtest_engine.py task1/smic_a_daily_qfq.csv --out-dir task3/exports
```

启用成交量过滤、移动止损、最大回撤控制和趋势过滤：

```bash
python3 -B task3/double_ma_backtest_engine.py task1/smic_a_daily_qfq.csv \
  --out-dir task3/exports \
  --use-volume-filter \
  --use-trailing \
  --use-drawdown-control \
  --use-trend-filter
```

命令行输出：

- `task3/exports/double_ma_summary.json`
- `task3/exports/double_ma_trades.csv`
- `task3/exports/double_ma_signals.csv`
- `task3/exports/double_ma_equity.csv`

## 注意事项

- 自动复权判断只能作为提示，不能保证完全准确。
- 若数据区间太短，无法计算长均线或下一日成交，看板会阻止执行。
- PDF 首版采用浏览器打印方案；HTML 报告是完整可保存文件。
