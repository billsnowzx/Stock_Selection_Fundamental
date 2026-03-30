from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stock_selection_fundamental.runtime import append_run_audit, generate_run_id


class RuntimeUtilsTests(unittest.TestCase):
    def test_generate_run_id_prefix(self) -> None:
        run_id = generate_run_id(prefix="bt")
        self.assertTrue(run_id.startswith("bt_"))

    def test_append_run_audit_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = append_run_audit(root, {"kind": "backtest", "run_id": "r1", "status": "success"})
            self.assertTrue(path.exists())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["run_id"], "r1")
            self.assertEqual(payload["status"], "success")
            self.assertIn("logged_at", payload)


if __name__ == "__main__":
    unittest.main()
