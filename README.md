# Stock Selection Fundamental

港股 / A 股基本面量化研究框架（配置驱动、模块化、可扩展）。

当前仓库包含两套实现：

- `stock_selection_fundamental/`: 重构后的主框架（推荐）
- `hk_stock_quant/`: 历史兼容实现（保留可运行能力）

## Core Features

- 统一 `DataProvider` 接口
- 财报可见性控制（`release_date <= signal_date`）
- 数据标准化映射（字段、市场、行业）
- 6 因子评分、月度调仓、交易成本与滑点
- 组合约束：单票上限、持仓上下限、流动性、lot size、最小成交额
- 风险能力：可选行业中性、风格软约束（占位可跑）
- 日频 Brinson-lite 归因（market / industry / selection）
- 研究输出：IC / Rank IC / 分层收益 / 稳定性
- 回测报告：CSV + HTML
- 基线冻结与回归对比（`baseline_metrics.json`）
- 实验平台：参数网格、walk-forward、regime 测试
- 运行审计：`run_id`、`config_hash`、`data_hash`
- AkShare 增量同步与断点检查点（`sync_checkpoint.json`）

## Project Structure

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

## Install

```bash
python -m pip install -e .
```

## Quick Start

### 1) Generate demo data

```bash
python -m stock_selection_fundamental.cli generate-demo-data --output-dir sample_data
```

### 2) Prepare curated snapshot (recommended)

```bash
python -m stock_selection_fundamental.cli prepare-curated \
  --config configs/backtests/hk_top20.yaml \
  --data-dir sample_data \
  --output-dir data/curated/hk_phase1
```

### 3) Run backtest

```bash
python -m stock_selection_fundamental.cli run-backtest \
  --config configs/backtests/hk_top20.yaml \
  --data-dir data/curated/hk_phase1 \
  --output-dir outputs/hk_top20_phase1 \
  --run-id baseline_hk
```

### 4) Freeze and check baseline

```bash
python -m stock_selection_fundamental.cli freeze-baseline \
  --metrics-file outputs/hk_top20_phase1/baseline_hk/metrics.json \
  --output tests/baseline/baseline_metrics.json

python -m stock_selection_fundamental.cli check-baseline \
  --baseline tests/baseline/baseline_metrics.json \
  --metrics-file outputs/hk_top20_phase1/baseline_hk/metrics.json \
  --tolerance-bps 50
```

### 5) Run experiment suite

```bash
python -m stock_selection_fundamental.cli run-experiment \
  --config configs/experiments/smoke.yaml
```

## CLI Commands

- `generate-demo-data`
- `sync-akshare-hk` (supports incremental sync; use `--full-sync` to disable)
- `sync-akshare-cn` (supports incremental sync; use `--full-sync` to disable)
- `prepare-curated`
- `run-backtest`
- `run-experiment`
- `freeze-baseline`
- `check-baseline`

## Config Files

- `configs/markets/*.yaml`: market and universe rules
- `configs/strategies/*.yaml`: factors, transforms, selection, weighting
- `configs/backtests/*.yaml`: dates, costs, outputs, storage
- `configs/risk/*.yaml`: industry neutrality, liquidity, style limits
- `configs/experiments/*.yaml`: grid, walk-forward, regimes

## Output Files

`run-backtest` writes to `output_dir/run_id/`:

- `nav_history.csv`, `trades.csv`, `holdings_history.csv`, `selection_history.csv`
- `metrics.json`, `config_snapshot.json`, `run_metadata.json`
- `ic_summary.csv`, `ic_timeseries.csv`, `rolling_ic.csv`
- `quantile_returns.csv`, `factor_coverage.csv`, `factor_moments.csv`, `factor_correlation.csv`
- `constraint_stats.csv`, `attribution_daily.csv`, `corporate_action_ledger.csv`
- `nav_vs_benchmark.png`, `drawdown.png`, `report.html`

Optional parquet output: set `storage.write_parquet: true` in backtest config.

## Testing

```bash
python -m unittest discover -s tests -v
```

## Compatibility

Legacy command `python -m hk_stock_quant.cli ...` remains available. New development should use `stock_selection_fundamental`.
