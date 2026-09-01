from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts.select_task import load_records, task_name, to_jsonable
from tts_search.eval_utils import build_submit_grade_and_medal
from tts_search.reward_func_utils import extract_code, get_clear_log, score2reward


class RewardUtilsTests(unittest.TestCase):
    def test_extract_code_from_fence(self) -> None:
        code = extract_code("```python\nprint('ok')\n```")
        self.assertIn('print("ok")', code)

    def test_power_sigmoid_stays_in_unit_interval(self) -> None:
        metadata = {
            "higher_is_better": True,
            "theoretical_min": 0.0,
            "theoretical_max": 1.0,
            "leaderboard_min": 0.1,
            "leaderboard_max": 0.9,
        }
        reward = score2reward(0.5, metadata, mode="power_sigmoid")
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)

    def test_clear_log_extracts_sandbox_output(self) -> None:
        run_log = "before\n--- OUTPUT START ---\nhello\n--- OUTPUT END ---\nafter"
        self.assertEqual(get_clear_log(run_log).strip(), "hello")


class EvalUtilsTests(unittest.TestCase):
    def test_grade_and_medal(self) -> None:
        leaderboard = pd.DataFrame({"score": [1.0, 0.9, 0.8, 0.7, 0.6]})
        grade, medal = build_submit_grade_and_medal(1.0, leaderboard)
        self.assertEqual(grade, 0.2)
        self.assertEqual(medal, "gold")


class SelectTaskTests(unittest.TestCase):
    def test_load_json_records(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            data_file = Path(temporary_directory) / "tasks.json"
            data_file.write_text(
                '[{"metadata":{"task_name":"demo"},"prompt":[]}]',
                encoding="utf-8",
            )
            records = load_records(data_file)
        self.assertEqual(task_name(records[0]), "demo")

    @patch("scripts.select_task.pd.read_parquet")
    def test_load_parquet_records(self, read_parquet) -> None:
        read_parquet.return_value = pd.DataFrame(
            [{"metadata": {"task_name": "demo"}, "prompt": []}]
        )
        records = load_records(Path("tasks.parquet"))
        self.assertEqual(task_name(records[0]), "demo")
        read_parquet.assert_called_once_with(Path("tasks.parquet"))

    def test_numpy_scalar_is_jsonable(self) -> None:
        scalar = pd.Series([3]).iloc[0]
        self.assertEqual(to_jsonable(scalar), 3)


if __name__ == "__main__":
    unittest.main()
