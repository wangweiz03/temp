from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluate_codex import branch_uses_planning, route_skills_for_branch
from prompts import SYSTEM_PROMPT

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PaperAlignmentTests(unittest.TestCase):
    def test_baseline_system_prompt_matches_appendix(self) -> None:
        expected = """You are a top-ranked Kaggle grandmaster with extensive competition experience.
Your objective is to solve a Kaggle competition,
with the goal of maximizing the Position Score (Your rank in the leaderboard) in limited steps.
You must use Machine Learning/Deep Learning/Computer Vision/NLP/etc. methods to solve the problem,
the score of random guess or without any ML/DL/CV/NLP methods will be cancelled finally.
You are likely to train models according to specific competition requirements.

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. DO NOT execute any code - do not run Python scripts or shell commands
2. DO NOT run the solution yourself
3. Your ONLY task is to create a file named `solution.py` containing the complete solution code
4. Create `solution.py` by writing it to the current working directory
5. The code will be executed later in a separate sandbox environment
6. After creating the file, simply confirm its creation - DO NOT RUN IT

Code Requirements:
- The code should be a single-file python program that is self-contained and can be executed as-is
- No parts of the code should be skipped, don't terminate the script before finishing the code
- All input data is located in the base path specified by the DATA_DIR environment variable. Always access the data path using os.environ.get("DATA_DIR") and never hardcode absolute paths.
- **Save test predictions to `submission.csv` in ./ directory as specified in the task description**
- The code can use GPU and PyTorch/CUDA for faster training if needed

Remember: CREATE the solution.py file, DO NOT EXECUTE it."""
        self.assertEqual(SYSTEM_PROMPT, expected)

    def test_single_source_routing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task_skills = root / "tasks"
            task_skills.mkdir()
            task_file = task_skills / "SKILL_demo.md"
            task_file.write_text("# Task Skill\n", encoding="utf-8")
            failure_file = root / "SKILL_error.md"
            failure_file.write_text("# Failure Skill\n", encoding="utf-8")

            draft = route_skills_for_branch("demo", "draft", task_skills, failure_file)
            debug = route_skills_for_branch("demo", "debug", task_skills, failure_file)
            improve = route_skills_for_branch(
                "demo", "improve", task_skills, failure_file
            )

        self.assertEqual(draft.sources, [str(task_file)])
        self.assertEqual(debug.sources, [str(failure_file)])
        self.assertEqual(improve.sources, [str(task_file)])
        self.assertIn("# Task Skill", draft.content)
        self.assertIn("# Failure Skill", debug.content)

    def test_only_draft_uses_planning(self) -> None:
        self.assertTrue(branch_uses_planning("draft"))
        self.assertFalse(branch_uses_planning("debug"))
        self.assertFalse(branch_uses_planning("improve"))

    def test_failure_skill_uses_appendix_workflow(self) -> None:
        text = (REPOSITORY_ROOT / "skills" / "SKILL_error.md").read_text(
            encoding="utf-8"
        )
        self.assertTrue(
            text.startswith("# ML Failure Prevention Skill: Plan-Code Version")
        )
        self.assertIn("3. Run the script in the sandbox.", text)
        self.assertIn("## 7. Output Schema Contract", text)
        self.assertNotIn("Plan-Code-Review", text)
        self.assertNotIn("Post-Code Review Checklist", text)

    def test_appendix_artifacts_are_present(self) -> None:
        synthesis = (
            REPOSITORY_ROOT / "paper_prompts" / "task_level_skill_synthesis_prompt.txt"
        )
        baseline = REPOSITORY_ROOT / "paper_prompts" / "baseline_experiment_prompt.txt"
        task_example = REPOSITORY_ROOT / "skills" / "SKILL_mlsp-2013-birds.md"
        self.assertTrue(synthesis.is_file())
        self.assertTrue(baseline.is_file())
        self.assertTrue(task_example.is_file())
        self.assertIn(
            " ".join(SYSTEM_PROMPT.split()),
            " ".join(baseline.read_text(encoding="utf-8").split()),
        )
        self.assertIn(
            "You are an expert Kaggle Grandmaster-style skill writer",
            synthesis.read_text(encoding="utf-8"),
        )
        self.assertTrue(
            task_example.read_text(encoding="utf-8").startswith(
                "# Skill: mlsp-2013-birds"
            )
        )


if __name__ == "__main__":
    unittest.main()
