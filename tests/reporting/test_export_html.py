from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stock_selection_fundamental.reporting.export_html import export_html_report
from stock_selection_fundamental.types import BacktestArtifacts


class ExportHtmlTests(unittest.TestCase):
    def test_report_contains_monthly_selection_section(self) -> None:
        nav_history = pd.DataFrame(
            [
                {"date": pd.Timestamp("2024-01-31"), "nav": 1.0, "benchmark_nav": 1.0},
                {"date": pd.Timestamp("2024-02-29"), "nav": 1.02, "benchmark_nav": 1.01},
            ]
        )
        selection_history = pd.DataFrame(
            [
                {"signal_date": pd.Timestamp("2024-01-31"), "symbol": "600519.SH", "rank": 1},
                {"signal_date": pd.Timestamp("2024-01-31"), "symbol": "000001.SZ", "rank": 2},
                {"signal_date": pd.Timestamp("2024-02-29"), "symbol": "601318.SH", "rank": 1},
            ]
        )
        artifacts = BacktestArtifacts(
            nav_history=nav_history,
            trades=pd.DataFrame(),
            holdings_history=pd.DataFrame(),
            selection_history=selection_history,
            metrics={"total_return": 0.02, "sharpe": 1.0},
            research_outputs={},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = export_html_report(
                artifacts=artifacts,
                output_dir=temp_dir,
                config_snapshot={"name": "test"},
            )
            text = Path(report_path).read_text(encoding="utf-8")
            self.assertIn("按月选股清单", text)
            self.assertIn("2024-01", text)
            self.assertIn("600519.SH, 000001.SZ", text)
            self.assertIn("2024-02", text)
            self.assertIn("601318.SH", text)


if __name__ == "__main__":
    unittest.main()
