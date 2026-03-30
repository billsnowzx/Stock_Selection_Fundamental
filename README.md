# Stock Selection Fundamental

港股 / A 股基本面量化研究框架（配置驱动、模块化、可扩展）。

当前仓库包含两套代码：

- `stock_selection_fundamental/`：重构后的主框架（推荐使用）
- `hk_stock_quant/`：历史兼容实现（保留可运行能力）

## 1. 核心能力

- 统一 `DataProvider` 接口（本地 CSV + AkShare HK/CN）
- 可见性控制（`release_date <= signal_date`）
- 数据标准化映射（字段、市场、行业）
- 6 因子评分、月度调仓、交易成本与滑点
- 组合约束：单票上限、持仓上下限、流动性约束、lot size、最小成交额
- 风险能力：行业中性（可选）、风格软约束（占位可跑）
- 归因：Brinson-lite（日频 market/industry/selection）
- 研究：IC / Rank IC / 分层收益 / 稳定性
- 报告：CSV + HTML
- 基线回归：`baseline_metrics.json` 冻结与对比
- 实验平台：参数网格、walk-forward、regime 测试

## 2. 目录结构

```text
stock_selection_fundamental/
  providers/   universe/   factors/   signals/
  portfolio/   risk/       backtest/
  research/    reporting/
configs/
  markets/ strategies/ backtests/ risk/ experiments/
scripts/
tests/
```

## 3. 安装

```bash
python -m pip install -e .
```

## 4. 快速开始

### 4.1 生成 demo 数据

```bash
python -m stock_selection_fundamental.cli generate-demo-data --output-dir sample_data
```

### 4.2 准备 curated 快照（推荐先做）

```bash
python -m stock_selection_fundamental.cli prepare-curated \
  --config configs/backtests/hk_top20.yaml \
  --data-dir sample_data \
  --output-dir data/curated/hk_phase1
```

### 4.3 运行回测（配置驱动）

```bash
python -m stock_selection_fundamental.cli run-backtest \
  --config configs/backtests/hk_top20.yaml \
  --data-dir data/curated/hk_phase1 \
  --output-dir outputs/hk_top20_phase1 \
  --run-id baseline_hk
```

### 4.4 冻结并校验基线

```bash
python -m stock_selection_fundamental.cli freeze-baseline \
  --metrics-file outputs/hk_top20_phase1/baseline_hk/metrics.json \
  --output tests/baseline/baseline_metrics.json

python -m stock_selection_fundamental.cli check-baseline \
  --baseline tests/baseline/baseline_metrics.json \
  --metrics-file outputs/hk_top20_phase1/baseline_hk/metrics.json \
  --tolerance-bps 50
```

### 4.5 运行实验

```bash
python -m stock_selection_fundamental.cli run-experiment \
  --config configs/experiments/smoke.yaml
```

## 5. CLI 命令

- `generate-demo-data`
- `sync-akshare-hk`
- `sync-akshare-cn`
- `prepare-curated`
- `run-backtest`
- `run-experiment`
- `freeze-baseline`
- `check-baseline`

## 6. 配置文件

- `configs/markets/*.yaml`：市场与 universe 规则
- `configs/strategies/*.yaml`：因子、标准化、选股、权重方法
- `configs/backtests/*.yaml`：回测区间、成本、输出、存储
- `configs/risk/*.yaml`：行业中性/流动性/风格约束
- `configs/experiments/*.yaml`：参数网格、walk-forward、regime

## 7. 输出说明

`run-backtest` 输出目录（`output_dir/run_id/`）包含：

- `nav_history.csv`, `trades.csv`, `holdings_history.csv`, `selection_history.csv`
- `metrics.json`, `config_snapshot.json`, `run_metadata.json`
- `ic_summary.csv`, `ic_timeseries.csv`, `rolling_ic.csv`
- `quantile_returns.csv`, `factor_coverage.csv`, `factor_moments.csv`, `factor_correlation.csv`
- `constraint_stats.csv`, `attribution_daily.csv`
- `nav_vs_benchmark.png`, `drawdown.png`, `report.html`

可选 parquet：在 backtest 配置中设置 `storage.write_parquet: true`。

## 8. 测试

```bash
python -m unittest discover -s tests -v
```

## 9. 兼容性

历史命令 `python -m hk_stock_quant.cli ...` 仍可使用；新开发建议统一迁移到 `stock_selection_fundamental`。
