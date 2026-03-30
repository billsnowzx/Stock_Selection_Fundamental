from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from hk_stock_quant.demo_data import write_demo_dataset
from stock_selection_fundamental.research.experiments import run_experiment_suite


class ExperimentIntegrationTests(unittest.TestCase):
    def test_experiment_runner_outputs_summary_and_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            out_dir = root / "exp_outputs"
            write_demo_dataset(data_dir, start="2021-01-01", end="2025-12-31", n_symbols=15, seed=17)

            exp_cfg = {
                "experiment_id": "exp_test",
                "base_backtest_config": "configs/backtests/hk_top20.yaml",
                "output_root": str(out_dir),
                "scenarios": [
                    {
                        "name": "demo_scenario",
                        "backtest.data_dir": str(data_dir),
                        "selection.top_n": 10,
                    }
                ],
                "walk_forward": {
                    "folds": [
                        {
                            "name": "wf_1",
                            "train_start": "2023-01-03",
                            "train_end": "2023-12-29",
                            "test_start": "2024-01-02",
                            "test_end": "2024-12-31",
                        }
                    ]
                },
                "regimes": [
                    {"name": "regime_a", "start": "2025-01-02", "end": "2025-12-31"},
                ],
            }
            cfg_path = root / "experiment.yaml"
            cfg_path.write_text(yaml.safe_dump(exp_cfg, sort_keys=False), encoding="utf-8")

            result = run_experiment_suite(cfg_path)
            self.assertFalse(result.summary.empty)
            self.assertIn("rank_final", result.summary.columns)
            self.assertIn("walk_forward_test_sharpe_mean", result.summary.columns)
            self.assertIn("regime_sharpe_mean", result.summary.columns)

            exp_dir = result.output_dir
            self.assertTrue((exp_dir / "experiment_summary.csv").exists())
            self.assertTrue((exp_dir / "experiment_ranking.csv").exists())
            self.assertTrue((exp_dir / "walk_forward_summary.csv").exists())
            self.assertTrue((exp_dir / "regime_summary.csv").exists())

            scenario_dir = exp_dir / "scenarios" / "demo_scenario"
            self.assertTrue((scenario_dir / "full_period" / "metrics.json").exists())
            self.assertTrue((scenario_dir / "full_period" / "report.html").exists())
            self.assertTrue((scenario_dir / "walk_forward_summary.csv").exists())
            self.assertTrue((scenario_dir / "regime_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
