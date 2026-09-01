from __future__ import annotations

import unittest

from evaluate import (
    DEFAULT_DATA_FILE,
    HARNESS_CLAUDE_CODE,
    HARNESS_CODEX,
    build_harness_command,
    extract_python_code,
    resolve_harness_and_model,
)


class HarnessConfigurationTests(unittest.TestCase):
    def test_requested_parquet_is_the_default(self) -> None:
        self.assertEqual(
            str(DEFAULT_DATA_FILE),
            "/hpc_data/ktian/superml/dataset/automl_parquet_valid_low_current_fixed/eval.parquet",
        )

    def test_harness_defaults_bind_expected_model_families(self) -> None:
        self.assertEqual(
            resolve_harness_and_model(HARNESS_CODEX, None),
            (HARNESS_CODEX, "gpt-5.4"),
        )
        harness, model = resolve_harness_and_model(HARNESS_CLAUDE_CODE, None)
        self.assertEqual(harness, HARNESS_CLAUDE_CODE)
        self.assertFalse(model.startswith("gpt-"))

    def test_cross_family_bindings_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_harness_and_model(HARNESS_CODEX, "claude-sonnet-4-6")
        with self.assertRaises(ValueError):
            resolve_harness_and_model(HARNESS_CLAUDE_CODE, "gpt-5.4")

    def test_commands_use_the_selected_cli_without_credentials(self) -> None:
        codex = build_harness_command(HARNESS_CODEX, "gpt-5.4", "medium", "sys")
        claude = build_harness_command(
            HARNESS_CLAUDE_CODE, "deepseek-v4", "medium", "sys"
        )
        self.assertEqual(codex[:2], ["codex", "exec"])
        self.assertEqual(claude[:2], ["claude", "--print"])
        self.assertNotIn("API_KEY", " ".join(codex + claude))

    def test_print_only_python_output_is_recovered(self) -> None:
        raw = "Done.\n```python\nimport os\nprint(os.getcwd())\n```"
        code = extract_python_code(raw)
        self.assertIn("import os", code)
        self.assertIn("print(os.getcwd())", code)


if __name__ == "__main__":
    unittest.main()
