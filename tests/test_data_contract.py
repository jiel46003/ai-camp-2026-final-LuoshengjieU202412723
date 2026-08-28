"""数据契约单测：验证真实数据格式符合预期（只验格式，不证明内容正确）。

运行：python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main

DATA = Path(__file__).resolve().parents[1] / "data/raw/all_seasons.csv"
REQUIRED = [
    "player_name", "age", "player_height", "player_weight",
    "draft_round", "draft_number", "pts",
]


def _is_float(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class TestDataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = main.read_csv(DATA)
        cls.header = list(cls.rows[0].keys()) if cls.rows else []

    def test_file_exists(self):
        self.assertTrue(DATA.exists(), "data/raw/all_seasons.csv 不存在")

    def test_row_count_range(self):
        self.assertGreaterEqual(len(self.rows), 10000, "行数应 >= 10000")

    def test_required_columns(self):
        missing = [c for c in REQUIRED if c not in self.header]
        self.assertEqual(missing, [], f"缺少必需字段: {missing}")

    def test_target_numeric(self):
        bad = [r["player_name"] for r in self.rows if not _is_float(r.get("pts"))]
        self.assertEqual(bad[:5], [], f"pts 含非数值，示例: {bad[:5]}")

    def test_clean_data_handles_undrafted(self):
        cleaned = main.clean_data([dict(r) for r in self.rows[:200]])
        for r in cleaned:
            for c in ("draft_round", "draft_number"):
                self.assertTrue(_is_float(r[c]), f"{c} 清理后仍非数值: {r[c]}")


if __name__ == "__main__":
    unittest.main()
