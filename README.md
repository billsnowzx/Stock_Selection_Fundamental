# Stock Selection Fundamental

面向港股 / A 股的基本面量化研究框架，目标是把因子研究、选股、组合构建、回测、研究诊断和报告输出拆成清晰模块，并用配置文件驱动实验。

本仓库当前包含两套实现：

- `stock_selection_fundamental/`：新版分层架构（本次重构主线）
- `hk_stock_quant/`：旧版兼容实现（保留可运行能力，便于平滑迁移）

## 模块结构

```text
stock_selection_fundamental/
  providers/
  universe/
  factors/
  signals/
  portfolio/
  risk/
  backtest/
  research/
  reporting/
configs/
  markets/
  strategies/
  backtests/
  risk/
```

## 安装

```bash
python -m pip install -e .
```

## 数据准备

### 1. 生成演示数据

```bash
python -m stock_selection_fundamental.cli generate-demo-data
```

默认输出到 `sample_data/`，包含：

- `security_master.csv`
- `price_history.csv`
- `financials.csv`
- `release_calendar.csv`

### 2. 同步 AkShare 港股

```bash
python -m stock_selection_fundamental.cli sync-akshare-hk --symbols 0700.HK 0939.HK
```

也可使用更大样本：

```bash
python -m stock_selection_fundamental.cli sync-akshare-hk --max-symbols 300
```

### 3. 同步 AkShare A 股

```bash
python -m stock_selection_fundamental.cli sync-akshare-cn --symbols 600519.SH 000001.SZ
```

## 运行回测（配置驱动）

```bash
python -m stock_selection_fundamental.cli run-backtest --config configs/backtests/hk_top20.yaml
```

同理可运行 A 股配置：

```bash
python -m stock_selection_fundamental.cli run-backtest --config configs/backtests/cn_top30.yaml
```

## 配置说明

- `configs/markets/*.yaml`：市场和股票池规则（板块、上市天数、停牌/流动性/ST 等）
- `configs/strategies/*.yaml`：因子权重、标准化方式、选股规则、持仓构建方式
- `configs/backtests/*.yaml`：回测区间、初始资金、成本滑点、输入输出路径
- `configs/risk/*.yaml`：风险约束预留配置

## 当前实现重点

- 统一 `DataProvider` 接口与字段标准化映射
- 财务数据按 `release_date <= 调仓日` 做时点控制
- Universe 过滤：上市天数、停牌、缺失价格、流动性、ST(A股)、可选行业筛选
- 因子层拆分：盈利/成长/杠杆/现金流 + winsorize/zscore/rank
- Signals：加权综合打分 + topN / 百分位选股
- Portfolio：等权/分数加权 + 单票上限 + 持仓上下限
- Backtest：月度信号、次日开盘成交、成本/滑点、停牌跳过
- Research：IC / Rank IC、分层收益、覆盖率/稳定性
- Reporting：CSV + HTML 报告

## 输出文件

回测输出目录（`output_dir`）包含：

- `nav_history.csv`
- `trades.csv`
- `holdings_history.csv`
- `selection_history.csv`
- `metrics.json`
- `config_snapshot.json`
- `ic_summary.csv`
- `ic_timeseries.csv`
- `rolling_ic.csv`
- `quantile_returns.csv`
- `factor_coverage.csv`
- `factor_moments.csv`
- `factor_correlation.csv`
- `nav_vs_benchmark.png`
- `drawdown.png`
- `report.html`

## 测试

```bash
python -m unittest discover -s tests -v
```

## 迁移说明

- 原命令 `python -m hk_stock_quant.cli ...` 仍可使用。
- 建议新实验迁移到 `python -m stock_selection_fundamental.cli ...`。
- 第二阶段接口已预留：行业中性、风格约束、风险归因、参数批量实验、walk-forward。
