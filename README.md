# 港股基本面量化选股与回测框架

这是一个从零开始的港股基本面量化研究框架，目标是把你给出的 6 个条件落成可执行的选股、回测和研究输出流程。
当前版本支持两条数据路径：

- 演示路径：本地 CSV / 合成样例数据
- 真实路径：AkShare 同步港股主板数据到本地 CSV，再复用同一套回测引擎

## 覆盖内容

- 可替换数据接口 `DataProvider`
- 默认本地 CSV 数据适配器 `LocalCSVDataProvider`
- `AkshareHKDataProvider` 真实港股数据同步器
- 6 个基本面因子计算与加权评分
- 港股主板普通股股票池过滤
- 月度调仓、下一交易日开盘成交的回测引擎
- 交易成本、滑点、停牌处理
- 因子 IC / Rank IC、分层收益、命中率统计
- 报表导出与净值图绘制
- 可重复生成的演示数据

## 目录结构

```text
hk_stock_quant/
  data/
  backtest.py
  cli.py
  config.py
  demo_data.py
  factors.py
  reporting.py
  strategy.py
  universe.py
scripts/
tests/
```

## 因子定义

- `roic = nopat / invested_capital`，若数据源直接给出 `roic` 则优先使用
- `net_margin = net_income / revenue`，若数据源直接给出 `net_margin` 则优先使用
- `debt_to_cashflow = total_liabilities / operating_cashflow`
- `revenue_growth_yoy = (revenue - prev_revenue) / prev_revenue`，若数据源直接给出同比增速则优先使用
- `net_income_growth_yoy = (net_income - prev_net_income) / prev_net_income`，若数据源直接给出同比增速则优先使用
- `fcf_conversion = free_cashflow / net_income`

默认处理规则：

- 分母小于等于 0 或关键字段缺失时，因子记为缺失值
- 横截面先做 winsorize，再做 z-score 标准化
- 负向因子 `debt_to_cashflow` 会自动反向
- 初版 6 因子等权
- 至少 4 个有效因子才参与综合打分

## CSV 数据格式

### `security_master.csv`

必需列：

- `symbol`
- `name`
- `board`
- `security_type`
- `industry`
- `list_date`
- `delist_date`

### `price_history.csv`

必需列：

- `date`
- `symbol`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `turnover`
- `is_suspended`

### `financials.csv`

必需列：

- `symbol`
- `period_end`
- `report_type`
- `revenue`
- `net_income`
- `ebit`
- `effective_tax_rate`
- `invested_capital`
- `total_liabilities`
- `operating_cashflow`
- `capital_expenditure`
- `free_cashflow`
- `nopat`

兼容扩展列：

- `roic`
- `net_margin`
- `revenue_growth_yoy`
- `net_income_growth_yoy`
- `fcf_conversion`
- `debt_to_cashflow`

### `release_calendar.csv`

必需列：

- `symbol`
- `period_end`
- `release_date`

## 快速开始

生成样例数据：

```bash
python -m hk_stock_quant.cli generate-demo-data --output-dir sample_data
```

运行回测：

```bash
python -m hk_stock_quant.cli run-backtest --data-dir sample_data --output-dir outputs
```

## 接入真实港股数据

同步指定股票到本地数据集：

```bash
python -m hk_stock_quant.cli sync-akshare-hk --output-dir real_data_hk_sample --symbols 00700,00941,00005,00388,01024 --start 2022-01-01 --end 2025-12-31
```

同步主板前 50 只股票：

```bash
python -m hk_stock_quant.cli sync-akshare-hk --output-dir real_data_hk --max-symbols 50 --start 2022-01-01 --end 2025-12-31
```

然后复用同一条回测命令：

```bash
python -m hk_stock_quant.cli run-backtest --data-dir real_data_hk_sample --output-dir outputs_real --top-n 3 --start 2023-01-03 --end 2025-12-31
```

## AkShare 口径说明

- 港股日线来自 `stock_hk_hist`
- 主板股票池来自 `stock_hk_main_board_spot_em`
- 基准默认使用恒指 `HSI`，历史来自 `stock_hk_index_daily_sina`
- 财务指标来自 `stock_financial_hk_analysis_indicator_em`
- 资产负债表与现金流量表明细来自 `stock_financial_hk_report_em`
- 若免费源没有精确财报发布日期，当前实现按报告类型回退为保守滞后：年报 90 天，中报 60 天，其他报告 45 天

## 输出文件

回测输出目录默认包含：

- `nav_history.csv`
- `trades.csv`
- `holdings_history.csv`
- `selection_history.csv`
- `metrics.json`
- `config.json`
- `ic_timeseries.csv`
- `ic_summary.csv`
- `quantile_returns.csv`
- `hit_rate.csv`
- `nav_vs_benchmark.png`

## 验证

```bash
python -m unittest discover -s tests -v
```

## 当前边界

- 初版只做多头，不做行业中性和风险模型约束
- 默认按日线近似港股交易日历，不含真实节假日撮合细节
- 未处理港股最小买卖单位和公司行为复权
- AkShare 全市场主板同步较慢，建议先按 `--symbols` 或 `--max-symbols` 小范围同步后再扩容
