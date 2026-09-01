"""
Multi-harness evaluation driver for Experience Routed Search (ERS).

ERS performs adaptive draft/debug/improve search with failure taxonomy and
compact per-task memory:
1. Choose a strategy branch for each round.
2. Assign a search intent that controls improve-best vs alternative exploration.
3. Route exactly one full Skill source: Task Skill for draft/improve, Failure Skill for debug.
4. Generate planning.md before coding on draft rounds.
5. Let debug and improve rounds code directly from the routed full Skill.
6. Validate, classify failures, archive a commit, update branch state, method-category state, and task memory.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import os
import re
import textwrap
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from dotenv import load_dotenv

from prompts import PLANNING_SYSTEM_PROMPT, RUNTIME_HARDENING_CONTEXT, SYSTEM_PROMPT
from tts_search import eval_utils
from tts_search.reward_func_utils import (
    extract_code,
    format_sandbox_feedback,
    get_clear_log,
    get_sandbox_result,
    score2reward,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parent
EXTERNAL_ROOT = REPOSITORY_ROOT / "external"
SKILLS_ROOT = REPOSITORY_ROOT / "skills"
LEADERBOARD_DIR = os.environ.get("MLE_LEADERBOARD_DIR") or None
SUBMIT_DATA_ROOT = Path(
    os.environ.get("MLE_SUBMIT_DATA_ROOT") or EXTERNAL_ROOT / "submission_data"
)
DEFAULT_TASK_SKILLS_DIR = Path(os.environ.get("MLE_TASK_SKILLS_DIR") or SKILLS_ROOT)
DEFAULT_ERROR_SKILL_FILE = Path(
    os.environ.get("MLE_ERROR_SKILL_FILE") or SKILLS_ROOT / "SKILL_error.md"
)
SANDBOX_BASE_URL = os.environ.get("MLE_SANDBOX_BASE_URL", "http://127.0.0.1:6580")
DEFAULT_DATA_FILE = Path(
    "/hpc_data/ktian/superml/dataset/automl_parquet_valid_low_current_fixed/eval.parquet"
)
HARNESS_CODEX = "codex"
HARNESS_CLAUDE_CODE = "claude-code"
HARNESS_CHOICES = (HARNESS_CODEX, HARNESS_CLAUDE_CODE)
DEFAULT_MODELS = {
    HARNESS_CODEX: "gpt-5.4",
    HARNESS_CLAUDE_CODE: "claude-sonnet-4-6-cc",
}
SKILL_FILE_SUFFIXES = {".md", ".py", ".json", ".txt", ".yaml", ".yml"}
SKILL_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}

BACKEND_ID = "multi_harness_ers_eema_task_failure_skill_routing"

CLAUDE_OUTPUT_GUARDS = {
    "planning": """You are running through a print-only Claude Code harness. Do not edit files.
Return the complete planning.md content in exactly one fenced markdown block. Do not omit sections.""",
    "coding": """You are running through a print-only Claude Code harness. Do not edit files.
Return the complete solution.py content in exactly one fenced python block. Do not abbreviate or omit code.""",
    "memory": """You are running through a print-only Claude Code harness. Do not edit files.
Return the complete task memory in exactly one fenced markdown block.""",
}

DRAFT_TASK_SKILL_GUARD = """Draft branch contract:
- Use the full task-specific Kaggle skill as the primary modeling recipe.
- Verify files, columns, target, metric, resource risks, and submission format in the generated solution.
- Do not spend the first round on generic AutoML search.
- Implement the skill's highest-ROI stable baseline directly, with conservative resource bounds and a fallback submission path."""

IMPROVE_BEST_GUARD = """Improve-best contract:
- Anchor on best_local_cv when available and make bounded, high-confidence micro-tuning changes.
- Prefer folds/seeds, feature caps, regularization, calibration, blending, postprocessing, TTA, and small fallback repairs.
- Do not replace the whole method category unless the search intent explicitly asks for explore_alternative.
- If score is already stable and recent attempts do not improve, spend effort on robust fallback/format/runtime fixes before broad novelty."""

ALTERNATIVE_GUARD = """Explore-alternative contract:
- The alternative must still be a high-priority candidate from the task-specific Kaggle skill.
- best_local_cv is a comparison anchor, not the implementation template.
- Avoid only the explicitly listed method category; do not ban the best category globally for future improve-best rounds.
- Keep the implementation bounded and include a fallback to a stable submission path."""

DEBUG_ERROR_GUARD = """Debug branch contract:
- Repair only the latest concrete failure unless the feedback proves the approach cannot run.
- Classify and fix schema, dependency, timeout, OOM, submission, metric, data parsing, or output-format errors first.
- Do not use debug rounds to introduce a new model family or broad ensemble."""


@dataclass(frozen=True)
class BranchSpec:
    name: str
    title: str
    goal: str
    instructions: str


BRANCH_SPECS: tuple[BranchSpec, ...] = (
    BranchSpec(
        name="draft",
        title="Kaggle-Skill Draft",
        goal="Implement the task-specific Kaggle skill's highest-ROI stable recipe with correct schema, resources, and submission format.",
        instructions=(
            "Use the full task skill as the modeling anchor from the first round. Prioritize robust data loading, "
            "schema inference, metric alignment, bounded training, and a guaranteed fallback submission."
        ),
    ),
    BranchSpec(
        name="debug",
        title="Debug Repair",
        goal="Fix the latest concrete failure with the smallest necessary code change.",
        instructions=(
            "Read the latest failed commit feedback first. Do not redesign the whole solution unless the failure proves "
            "the approach is impossible. Focus on schema, dependencies, timeout/OOM, submission, metric, data parsing, and output-format fixes."
        ),
    ),
    BranchSpec(
        name="improve",
        title="Improve Score",
        goal="Improve the current solution with bounded best-local-CV micro-tuning or an explicitly requested high-confidence alternative.",
        instructions=(
            "Use the full task skill as the main knowledge source. Default to best_local_cv anchoring: folds, seeds, "
            "feature caps, regularization, calibration, blending, TTA, and postprocessing. Explore a different method category only "
            "when the search intent explicitly requires it."
        ),
    ),
)

DEFAULT_WARMUP_BRANCHES = ("draft", "improve")
BRANCH_SPEC_BY_NAME = {spec.name: spec for spec in BRANCH_SPECS}
BRANCH_ALIASES = {
    "baseline": "draft",
    "repair": "debug",
    "feature": "improve",
    "model": "improve",
    "exploit": "improve",
}

INTENT_IMPROVE_BEST = "improve_best"
INTENT_EXPLORE_ALTERNATIVE = "explore_alternative"
INTENT_ABLATE_BEST = "ablate_best"
INTENT_REPAIR_FAILURE = "repair_failure"
INTENT_RESET_BASELINE = "reset_baseline"


def normalize_branch_name(branch: str) -> str:
    """Map legacy branch aliases to the ERS draft/debug/improve action set."""
    clean = branch.strip().lower()
    return BRANCH_ALIASES.get(clean, clean)


def normalize_branch_sequence(branches: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize and de-duplicate branch names while preserving order."""
    seen: set[str] = set()
    normalized: list[str] = []
    for branch in branches:
        clean = normalize_branch_name(branch)
        if clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    return tuple(normalized)


def branch_uses_planning(branch: str) -> bool:
    """Return whether a branch runs the separate planning phase."""
    return normalize_branch_name(branch) == "draft"


@dataclass(frozen=True)
class EvalContext:
    phase: str
    data_dir: str


@dataclass(frozen=True)
class SkillRoute:
    branch: str
    reason: str
    sources: list[str]
    content: str


def resolve_submit_data_dir(val_data_dir: str) -> str:
    """Resolve submit data directory from validation data directory."""
    data_path = Path(val_data_dir)
    task_name = data_path.name
    return str(SUBMIT_DATA_ROOT / task_name)


def build_metadata_prompt(metadata: dict[str, Any]) -> str:
    """Format parquet metadata into the prompt."""
    lines = [
        "[METADATA]",
        f"Task: {metadata.get('task_name', 'unknown')}",
        f"Resource Type: {metadata.get('cpu_gpu', 'unknown')}",
        f"Data Directory: {metadata.get('data_dir', 'unknown')}",
        (
            "Evaluation Metric Range: "
            f"[{metadata.get('theoretical_min', 'unknown')}, {metadata.get('theoretical_max', 'unknown')}]"
        ),
        f"Higher is Better: {metadata.get('higher_is_better', 'unknown')}",
    ]

    data_description = metadata.get("data_description")
    if data_description:
        lines.extend(["", "[DATA DESCRIPTION]", str(data_description)])

    task_description = metadata.get("task_description")
    if task_description:
        lines.extend(["", "[TASK DESCRIPTION]", str(task_description)])

    return "\n".join(lines)


def _fence_lang(path: Path) -> str:
    return {
        ".md": "markdown",
        ".py": "python",
        ".json": "json",
        ".txt": "text",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(path.suffix.lower(), "text")


def _safe_read_text(path: Path, limit: int | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    return _truncate_text(text, limit) if limit else text


def load_task_skill(task_name: str, skills_dir: Path) -> tuple[str | None, str | None]:
    skill_file = skills_dir / f"SKILL_{task_name}.md"
    content = _safe_read_text(skill_file)
    if content is None:
        logger.warning("Task skill not found: %s", skill_file)
        return None, None
    return str(skill_file), content


def _iter_skill_package_files(skill_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in skill_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKILL_SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SKILL_FILE_SUFFIXES:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(skill_dir).as_posix())


def load_skill_package(
    skill_path: Path, limit: int = 60000
) -> tuple[str | None, str | None]:
    if not skill_path.exists():
        logger.warning("Skill package not found: %s", skill_path)
        return None, None
    if skill_path.is_file():
        return str(skill_path), _safe_read_text(skill_path, limit=limit)
    if not skill_path.is_dir():
        return None, None

    sections = [f"Package path: {skill_path}"]
    loaded = 0
    for file_path in _iter_skill_package_files(skill_path):
        rel = file_path.relative_to(skill_path).as_posix()
        content = _safe_read_text(file_path, limit=30000)
        if content is None:
            continue
        sections.extend(
            [
                "",
                f"## File: {rel}",
                f"```{_fence_lang(file_path)}",
                content.rstrip(),
                "```",
            ]
        )
        loaded += 1
    if loaded == 0:
        return None, None
    return str(skill_path), _truncate_text("\n".join(sections), limit)


def extract_markdown_sections(text: str, wanted_titles: list[str]) -> str:
    """Extract selected level-2 markdown sections by fuzzy title match."""
    if not text:
        return ""
    lines = text.splitlines()
    wanted = [title.lower() for title in wanted_titles]
    selected: list[str] = []
    in_section = False
    for line in lines:
        if line.startswith("## "):
            title = line[3:].strip().lower()
            in_section = any(key in title for key in wanted)
        elif line.startswith("# ") and selected:
            in_section = False
        if in_section:
            selected.append(line)
    return "\n".join(selected).strip()


def route_skills_for_branch(
    task_name: str,
    branch: str,
    task_skills_dir: Path,
    error_skill_file: Path,
) -> SkillRoute:
    """Route exactly one full Skill source for the current ERS branch.

    draft -> Task Skill
    debug -> Failure Skill
    improve -> Task Skill
    """
    branch = normalize_branch_name(branch)
    sources: list[str] = []
    sections: list[str] = []

    def add_source(title: str, path: str | None, content: str | None) -> None:
        if not content:
            return
        if path:
            sources.append(path)
        sections.extend([f"## {title}", content.strip()])

    if branch == "draft":
        task_skill_path, task_skill = load_task_skill(task_name, task_skills_dir)
        add_source("Task Skill", task_skill_path, task_skill)
        reason = "draft routes the full task-specific Skill"
    elif branch == "debug":
        error_skill_path, error_skill = load_skill_package(
            error_skill_file, limit=200000
        )
        add_source("Failure Skill", error_skill_path, error_skill)
        reason = "debug routes the full failure-prevention Skill"
    elif branch == "improve":
        task_skill_path, task_skill = load_task_skill(task_name, task_skills_dir)
        add_source("Task Skill", task_skill_path, task_skill)
        reason = "improve routes the full task-specific Skill"
    else:
        task_skill_path, task_skill = load_task_skill(task_name, task_skills_dir)
        add_source("Task-Specific Knowledge", task_skill_path, task_skill)
        reason = "fallback route"

    content = "\n\n".join(sections).strip()
    if not content:
        content = "No routed skill content was available; rely on task description, branch state, and memory."
    return SkillRoute(branch=branch, reason=reason, sources=sources, content=content)


def memory_file_for_task(task_name: str, task_dir: Path) -> Path:
    return task_dir / "memories" / f"MEMORY_{task_name}.md"


def load_memory_for_task(task_name: str, task_dir: Path, limit: int = 16000) -> str:
    memory_file = memory_file_for_task(task_name, task_dir)
    content = _safe_read_text(memory_file, limit=limit)
    return content or "No task memory yet."


def init_git_structure(
    task_dir: Path, branch_specs: tuple[BranchSpec, ...] = BRANCH_SPECS
) -> None:
    """Initialize git-style directory structure."""
    (task_dir / "commits").mkdir(exist_ok=True)
    (task_dir / "index").mkdir(exist_ok=True)
    (task_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
    (task_dir / "refs" / "tags").mkdir(parents=True, exist_ok=True)
    (task_dir / "traces").mkdir(exist_ok=True)
    (task_dir / "memories").mkdir(exist_ok=True)

    for spec in branch_specs:
        (task_dir / "refs" / "heads" / spec.name).write_text("", encoding="utf-8")
    (task_dir / "refs" / "heads" / "main").write_text("", encoding="utf-8")
    (task_dir / "index" / "commit_log.jsonl").touch()
    branch_summary = {}
    for spec in branch_specs:
        branch_summary[spec.name] = {
            "title": spec.title,
            "goal": spec.goal,
            "head": "",
            "score": None,
            "best_score": None,
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "total_time": 0.0,
            "last_status": None,
            "last_round": None,
            "updated_at": None,
        }
    (task_dir / "index" / "branch_summary.json").write_text(
        json.dumps(branch_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (task_dir / "index" / "tag_registry.json").write_text("{}", encoding="utf-8")
    (task_dir / "index" / "branch_decisions.jsonl").touch()
    specs_payload = {
        spec.name: {
            "title": spec.title,
            "goal": spec.goal,
            "instructions": spec.instructions,
        }
        for spec in branch_specs
    }
    (task_dir / "index" / "branch_specs.json").write_text(
        json.dumps(specs_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_commit_hash(round_num: int, timestamp: str) -> str:
    """Generate a short commit hash."""
    return hashlib.sha1(f"{round_num}_{timestamp}".encode("utf-8")).hexdigest()[:8]


def save_commit(
    task_dir: Path,
    commit_hash: str,
    planning_text: str,
    solution_code: str,
    feedback: str,
    result: dict[str, Any],
    round_summary: dict[str, str] | None = None,
) -> None:
    """Save commit payload."""
    commit_dir = task_dir / "commits" / commit_hash
    commit_dir.mkdir(parents=True, exist_ok=True)

    (commit_dir / "planning.md").write_text(planning_text, encoding="utf-8")
    (commit_dir / "solution.py").write_text(solution_code, encoding="utf-8")
    (commit_dir / "validation_feedback.txt").write_text(feedback, encoding="utf-8")
    if round_summary:
        (commit_dir / "round_summary.json").write_text(
            json.dumps(round_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        summary_md = (
            f"Method: {round_summary.get('method_summary', '')}\n"
            f"Category: {round_summary.get('method_category', '')}\n"
            f"Relative change: {round_summary.get('relative_change', '')}\n"
            f"Reflection: {round_summary.get('result_reflection', '')}\n"
        )
        (commit_dir / "round_summary.md").write_text(summary_md, encoding="utf-8")
    (commit_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def update_commit_log(
    task_dir: Path,
    commit_hash: str,
    branch: str,
    message: str,
    score: float | None,
    wall_time: float,
    status: str,
    round_summary: dict[str, str] | None = None,
) -> None:
    """Append one record to the lightweight commit log."""
    log_entry = {
        "hash": commit_hash,
        "branch": branch,
        "msg": message[:80],
        "score": score,
        "time": round(wall_time, 1),
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
    if round_summary:
        log_entry["method_summary"] = round_summary.get("method_summary", "")
        log_entry["result_reflection"] = round_summary.get("result_reflection", "")
        log_entry["method_category"] = round_summary.get("method_category", "")
        log_entry["relative_change"] = round_summary.get("relative_change", "")

    log_file = task_dir / "index" / "commit_log.jsonl"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def update_branch_ref(
    task_dir: Path,
    branch: str,
    commit_hash: str,
    score: float | None,
    status: str,
    wall_time: float,
    round_num: int,
    higher_is_better: bool,
) -> None:
    """Update branch pointer and branch summary with simple branch statistics."""
    branch = normalize_branch_name(branch)
    (task_dir / "refs" / "heads" / branch).write_text(commit_hash, encoding="utf-8")

    summary_file = task_dir / "index" / "branch_summary.json"
    summary = (
        json.loads(summary_file.read_text(encoding="utf-8"))
        if summary_file.exists()
        else {}
    )
    spec = BRANCH_SPEC_BY_NAME.get(branch)
    current = summary.get(branch, {})
    best_score = current.get("best_score")
    if score is not None:
        if best_score is None:
            best_score = score
        elif higher_is_better and score > best_score:
            best_score = score
        elif not higher_is_better and score < best_score:
            best_score = score

    attempts = int(current.get("attempts", 0)) + 1
    successes = int(current.get("successes", 0)) + (1 if score is not None else 0)
    failures = int(current.get("failures", 0)) + (0 if score is not None else 1)
    total_time = float(current.get("total_time", 0.0)) + float(wall_time or 0.0)
    summary[branch] = {
        "title": current.get("title") or (spec.title if spec else branch),
        "goal": current.get("goal") or (spec.goal if spec else ""),
        "head": commit_hash,
        "score": score,
        "best_score": best_score,
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / attempts if attempts else 0.0,
        "total_time": round(total_time, 2),
        "last_status": status,
        "last_round": round_num,
        "updated_at": datetime.now().isoformat(),
    }
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def record_branch_attempt_without_commit(
    task_dir: Path,
    branch: str,
    status: str,
    wall_time: float,
    round_num: int,
) -> None:
    """Record a branch attempt that did not produce an archived commit."""
    branch = normalize_branch_name(branch)
    summary_file = task_dir / "index" / "branch_summary.json"
    summary = (
        json.loads(summary_file.read_text(encoding="utf-8"))
        if summary_file.exists()
        else {}
    )
    spec = BRANCH_SPEC_BY_NAME.get(branch)
    current = summary.get(branch, {})
    attempts = int(current.get("attempts", 0)) + 1
    successes = int(current.get("successes", 0))
    failures = int(current.get("failures", 0)) + 1
    total_time = float(current.get("total_time", 0.0)) + float(wall_time or 0.0)
    summary[branch] = {
        "title": current.get("title") or (spec.title if spec else branch),
        "goal": current.get("goal") or (spec.goal if spec else ""),
        "head": current.get("head", ""),
        "score": current.get("score"),
        "best_score": current.get("best_score"),
        "attempts": attempts,
        "successes": successes,
        "failures": failures,
        "success_rate": successes / attempts if attempts else 0.0,
        "total_time": round(total_time, 2),
        "last_status": status,
        "last_round": round_num,
        "updated_at": datetime.now().isoformat(),
    }
    summary_file.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def get_current_branch(task_dir: Path) -> str:
    """Get current branch name."""
    head_file = task_dir / "HEAD"
    if head_file.exists():
        branch = head_file.read_text(encoding="utf-8").strip()
        if branch:
            return normalize_branch_name(branch)
    return "main"


def set_current_branch(task_dir: Path, branch: str) -> None:
    """Set current branch name."""
    (task_dir / "HEAD").write_text(normalize_branch_name(branch), encoding="utf-8")


def create_tag(
    task_dir: Path,
    tag_name: str,
    commit_hash: str,
    reason: str,
    score: float | None,
    branch: str,
) -> None:
    """Create a tag pointing to a commit."""
    branch = normalize_branch_name(branch)
    (task_dir / "refs" / "tags" / tag_name).write_text(commit_hash, encoding="utf-8")

    registry_file = task_dir / "index" / "tag_registry.json"
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    registry[tag_name] = {
        "commit": commit_hash,
        "score": score,
        "branch": branch,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    }
    registry_file.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def get_commit_log_summary(task_dir: Path, limit: int = 10) -> str:
    """Build a compact commit log summary for the model."""
    log_file = task_dir / "index" / "commit_log.jsonl"
    if not log_file.exists():
        return "No commits yet."

    lines = [
        line
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return "No commits yet."

    recent = lines[-limit:] if len(lines) > limit else lines
    summary_parts = ["Recent commits:"]
    for line in reversed(recent):
        entry = json.loads(line)
        status_icon = "OK" if entry["status"] == "success" else "FAIL"
        score_str = f"{entry['score']:.4f}" if entry["score"] is not None else "N/A"
        summary_parts.append(
            f"{entry['hash']}  {entry['branch']:16s}  {entry['msg'][:36]:36s}  {score_str:8s}  {entry['time']:6.1f}s  {status_icon}"
        )
        method_summary = entry.get("method_summary")
        result_reflection = entry.get("result_reflection")
        if method_summary:
            summary_parts.append(f"  method: {method_summary}")
        method_category = entry.get("method_category")
        relative_change = entry.get("relative_change")
        if method_category:
            summary_parts.append(f"  category: {method_category}")
        if relative_change:
            summary_parts.append(f"  relative change: {relative_change}")
        if result_reflection:
            summary_parts.append(f"  reflection: {result_reflection}")

    return "\n".join(summary_parts)


def get_tag_summary(task_dir: Path) -> str:
    """Build a compact tag summary for the model."""
    registry_file = task_dir / "index" / "tag_registry.json"
    if not registry_file.exists():
        return "No tags yet."

    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if not registry:
        return "No tags yet."

    summary_parts = ["Tags:"]
    for tag_name, info in registry.items():
        score_str = f"{info['score']:.4f}" if info["score"] is not None else "N/A"
        summary_parts.append(
            f"- {tag_name:16s} -> {info['commit']}  branch={info.get('branch', 'main')}  ({score_str})  {info['reason']}"
        )

    return "\n".join(summary_parts)


def get_latest_failure_context(task_dir: Path, limit: int = 8000) -> str:
    """Return compact details for the latest failed validation attempt."""
    summary_file = task_dir / "rounds_summary.json"
    if not summary_file.exists():
        return "No validation failure context yet."
    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        return "No readable validation failure context yet."
    rounds = summary.get("rounds", [])
    for round_result in reversed(rounds):
        validation = round_result.get("validation") or {}
        if validation.get("score") is not None:
            continue
        taxonomy = validation.get("failure_taxonomy") or {}
        feedback = validation.get("feedback") or validation.get("clear_run_log") or ""
        payload = [
            "[LATEST FAILED VALIDATION]",
            f"round: {round_result.get('round')}",
            f"branch: {round_result.get('branch')}",
            f"status: {validation.get('status')}",
            f"taxonomy: {json.dumps(taxonomy, ensure_ascii=False)}",
            "",
            _truncate_text(str(feedback), limit),
        ]
        return "\n".join(payload)
    return "No failed validation attempts recorded."


def load_branch_summary(task_dir: Path) -> dict[str, Any]:
    """Load branch scoreboard."""
    summary_file = task_dir / "index" / "branch_summary.json"
    if not summary_file.exists():
        return {}
    return json.loads(summary_file.read_text(encoding="utf-8"))


def score_better(
    score: float | None, baseline: float | None, higher_is_better: bool
) -> bool:
    """Return whether score is better than baseline."""
    if score is None:
        return False
    if baseline is None:
        return True
    return score > baseline if higher_is_better else score < baseline


def get_branch_scoreboard(task_dir: Path, higher_is_better: bool) -> str:
    """Build a concise branch scoreboard for prompts."""
    summary = load_branch_summary(task_dir)
    if not summary:
        return "No branch scoreboard yet."

    rows = []
    for spec in BRANCH_SPECS:
        info = summary.get(spec.name, {})
        best_score = info.get("best_score")
        best_text = (
            f"{best_score:.4f}" if isinstance(best_score, (float, int)) else "N/A"
        )
        rows.append(
            (
                spec.name,
                best_score,
                f"- {spec.name:10s} attempts={int(info.get('attempts', 0)):2d} "
                f"success={int(info.get('successes', 0)):2d} fail={int(info.get('failures', 0)):2d} "
                f"best={best_text:8s} last={info.get('last_status') or 'none'} "
                f"head={info.get('head') or '-'}",
            )
        )

    def sort_key(row: tuple[str, Any, str]) -> tuple[int, float]:
        score = row[1]
        if not isinstance(score, (float, int)):
            return (0, 0.0)
        return (1, float(score) if higher_is_better else -float(score))

    sorted_rows = sorted(rows, key=sort_key, reverse=True)
    return "Branch scoreboard:\n" + "\n".join(row[2] for row in sorted_rows)


def get_best_branch(summary: dict[str, Any], higher_is_better: bool) -> str | None:
    """Find branch with best validation score."""
    best_name = None
    best_score = None
    for branch, info in summary.items():
        branch = normalize_branch_name(branch)
        if branch not in BRANCH_SPEC_BY_NAME:
            continue
        score = info.get("best_score")
        if not isinstance(score, (float, int)):
            continue
        if best_name is None or score_better(
            float(score), best_score, higher_is_better
        ):
            best_name = branch
            best_score = float(score)
    return best_name


def latest_round_failed(all_rounds: list[dict[str, Any]]) -> bool:
    """Return whether the latest round failed to produce a validation score."""
    if not all_rounds:
        return False
    latest = all_rounds[-1]
    return latest.get("validation", {}).get("score") is None


def get_round_branch(round_result: dict[str, Any]) -> str:
    """Read branch name from a round result or its branch decision."""
    return normalize_branch_name(
        str(
            round_result.get("branch")
            or round_result.get("branch_decision", {}).get("branch")
            or ""
        )
    )


def consecutive_branch_count(all_rounds: list[dict[str, Any]], branch: str) -> int:
    """Count consecutive trailing rounds on a branch."""
    count = 0
    for round_result in reversed(all_rounds):
        if get_round_branch(round_result) != branch:
            break
        count += 1
    return count


def recent_valid_scores(
    all_rounds: list[dict[str, Any]], limit: int = 3
) -> list[float]:
    """Return recent successful validation scores, newest first."""
    scores: list[float] = []
    for round_result in reversed(all_rounds):
        score = round_result.get("validation", {}).get("score")
        if score is None:
            continue
        scores.append(float(score))
        if len(scores) >= limit:
            break
    return scores


def recent_valid_no_best_improvement(
    all_rounds: list[dict[str, Any]],
    best_score: float | None,
    higher_is_better: bool,
    limit: int = 3,
) -> bool:
    """Return whether the recent successful rounds failed to match or beat current best."""
    if best_score is None:
        return False
    scores = recent_valid_scores(all_rounds, limit=limit)
    if len(scores) < limit:
        return False
    tolerance = 1e-12
    for score in scores:
        if abs(score - best_score) <= tolerance:
            return False
        if score_better(score, best_score, higher_is_better):
            return False
    return True


def recent_method_category_counts(
    all_rounds: list[dict[str, Any]], limit: int = 5
) -> Counter[str]:
    """Count method categories in recent completed rounds."""
    counts: Counter[str] = Counter()
    seen = 0
    for round_result in reversed(all_rounds):
        category = str(
            round_result.get("round_summary", {}).get("method_category", "")
        ).strip()
        if not category:
            continue
        counts[category] += 1
        seen += 1
        if seen >= limit:
            break
    return counts


def repeated_recent_method_category(
    all_rounds: list[dict[str, Any]], limit: int = 5, threshold: int = 3
) -> str | None:
    """Find a repeated method category to avoid during alternative exploration."""
    counts = recent_method_category_counts(all_rounds, limit=limit)
    if not counts:
        return None
    category, count = counts.most_common(1)[0]
    return category if count >= threshold else None


def choose_alternative_branch(summary: dict[str, Any]) -> str:
    """Choose the ERS action used for alternative exploration."""
    return "improve"


def load_tag_registry(task_dir: Path) -> dict[str, Any]:
    """Load tag registry."""
    registry_file = task_dir / "index" / "tag_registry.json"
    if not registry_file.exists():
        return {}
    return json.loads(registry_file.read_text(encoding="utf-8"))


def choose_branch_for_round(
    task_dir: Path,
    round_num: int,
    all_rounds: list[dict[str, Any]],
    higher_is_better: bool,
    branch_strategy: str,
    warmup_branches: tuple[str, ...],
) -> dict[str, Any]:
    """Choose the branch to run for the next round."""
    summary = load_branch_summary(task_dir)
    registry = load_tag_registry(task_dir)
    best_tag = registry.get("best_local_cv", {})
    best_score = best_tag.get("score")
    best_score_float = (
        float(best_score) if isinstance(best_score, (float, int)) else None
    )
    repeated_category = repeated_recent_method_category(all_rounds)
    no_recent_best_improvement = recent_valid_no_best_improvement(
        all_rounds,
        best_score_float,
        higher_is_better,
        limit=3,
    )
    improve_blockers: list[str] = []
    if consecutive_branch_count(all_rounds, "improve") >= 3:
        improve_blockers.append("consecutive_improve_limit")
    if no_recent_best_improvement:
        improve_blockers.append("recent_valid_no_best_improvement")
    if repeated_category and no_recent_best_improvement:
        improve_blockers.append(f"repeated_method_category:{repeated_category}")

    valid_warmup = tuple(
        branch
        for branch in normalize_branch_sequence(warmup_branches)
        if branch in BRANCH_SPEC_BY_NAME
    )
    if not valid_warmup:
        valid_warmup = DEFAULT_WARMUP_BRANCHES

    if round_num < len(valid_warmup):
        branch = valid_warmup[round_num]
        reason = "warmup_diversity"
        search_intent = (
            INTENT_RESET_BASELINE if branch == "draft" else INTENT_EXPLORE_ALTERNATIVE
        )
    elif latest_round_failed(all_rounds):
        branch = "debug"
        reason = "debug_latest_failure"
        search_intent = INTENT_REPAIR_FAILURE
    elif branch_strategy == "branch_cycle":
        cycle = valid_warmup + ("improve", "debug")
        branch = cycle[round_num % len(cycle)]
        reason = "fixed_branch_cycle"
        search_intent = (
            INTENT_IMPROVE_BEST
            if branch == "improve" and not improve_blockers
            else (
                INTENT_REPAIR_FAILURE
                if branch == "debug"
                else (
                    INTENT_RESET_BASELINE
                    if branch == "draft"
                    else INTENT_EXPLORE_ALTERNATIVE
                )
            )
        )
    else:
        untouched = []
        for spec in BRANCH_SPECS:
            if int(summary.get(spec.name, {}).get("attempts", 0)) != 0:
                continue
            if spec.name == "debug":
                continue
            untouched.append(spec.name)
        if untouched:
            branch = untouched[0]
            reason = "cover_untouched_branch"
            search_intent = (
                INTENT_EXPLORE_ALTERNATIVE
                if branch == "improve"
                else INTENT_RESET_BASELINE
            )
        elif "best_local_cv" in registry and round_num % 2 == 0:
            branch = "improve"
            if improve_blockers:
                reason = (
                    f"anti_repetition_improve_alternative:{','.join(improve_blockers)}"
                )
                search_intent = INTENT_EXPLORE_ALTERNATIVE
            else:
                reason = "improve_best_local_cv"
                search_intent = INTENT_IMPROVE_BEST
        elif round_num % 5 == 3:
            branch = "improve"
            reason = "scheduled_improve_alternative"
            search_intent = INTENT_EXPLORE_ALTERNATIVE
        elif round_num % 5 == 4:
            branch = "draft"
            reason = "scheduled_draft_reset"
            search_intent = INTENT_RESET_BASELINE
        else:
            best_branch = get_best_branch(summary, higher_is_better)
            if best_branch == "debug":
                branch = "improve" if "best_local_cv" in registry else "draft"
                reason = "continue_after_debug"
            else:
                branch = best_branch if best_branch in BRANCH_SPEC_BY_NAME else "draft"
                reason = "continue_best_branch"
            search_intent = (
                INTENT_IMPROVE_BEST
                if branch == "improve" and not improve_blockers
                else (
                    INTENT_REPAIR_FAILURE
                    if branch == "debug"
                    else (
                        INTENT_RESET_BASELINE
                        if branch == "draft"
                        else INTENT_EXPLORE_ALTERNATIVE
                    )
                )
            )

    if (
        branch == "improve"
        and improve_blockers
        and search_intent == INTENT_IMPROVE_BEST
    ):
        reason = f"anti_repetition_improve_alternative:{','.join(improve_blockers)}"
        search_intent = INTENT_EXPLORE_ALTERNATIVE

    spec = BRANCH_SPEC_BY_NAME.get(branch, BRANCH_SPEC_BY_NAME["draft"])
    decision = {
        "round": round_num,
        "branch": spec.name,
        "branch_title": spec.title,
        "reason": reason,
        "search_intent": search_intent,
        "anti_repetition": {
            "improve_blockers": improve_blockers,
            "avoid_method_category": (
                repeated_category
                if search_intent == INTENT_EXPLORE_ALTERNATIVE
                else None
            ),
            "observed_repeated_method_category": repeated_category,
            "recent_method_categories": dict(
                recent_method_category_counts(all_rounds, limit=5)
            ),
            "recent_valid_scores": recent_valid_scores(all_rounds, limit=3),
        },
        "goal": spec.goal,
        "instructions": spec.instructions,
        "best_local_cv_commit": best_tag.get("commit"),
        "best_local_cv_score": best_tag.get("score"),
        "timestamp": datetime.now().isoformat(),
    }

    decision_log = task_dir / "index" / "branch_decisions.jsonl"
    with decision_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    (task_dir / "index" / "current_branch_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return decision


def build_refinement_context(
    task_dir: Path, round_num: int, higher_is_better: bool
) -> str | None:
    """Build git-style navigation context for planning or coding."""
    if round_num == 0:
        current_branch = get_current_branch(task_dir)
        spec = BRANCH_SPEC_BY_NAME.get(current_branch, BRANCH_SPEC_BY_NAME["draft"])
        return f"""
=== Round {round_num + 1} - Branch Search Start ===

Current branch: {current_branch}
Branch goal: {spec.goal}
Branch instructions: {spec.instructions}

This is the first attempt on this branch. Use the branch goal to decide the current round's focused direction.
"""

    current_branch = get_current_branch(task_dir)
    spec = BRANCH_SPEC_BY_NAME.get(current_branch, BRANCH_SPEC_BY_NAME["draft"])
    commit_log = get_commit_log_summary(task_dir, limit=10)
    tag_summary = get_tag_summary(task_dir)
    latest_failure_context = get_latest_failure_context(task_dir)
    branch_scoreboard = get_branch_scoreboard(
        task_dir, higher_is_better=higher_is_better
    )
    decision_file = task_dir / "index" / "current_branch_decision.json"
    branch_decision = (
        decision_file.read_text(encoding="utf-8") if decision_file.exists() else "{}"
    )

    return f"""
=== Round {round_num + 1} - Branch-Oriented Search Navigation ===

Current branch: {current_branch}
Branch title: {spec.title}
Branch goal: {spec.goal}
Branch instructions: {spec.instructions}

Current branch decision:
{branch_decision}

{branch_scoreboard}

{commit_log}

{tag_summary}

{latest_failure_context}

To inspect prior attempts:
- Read `commits/{{hash}}/planning.md` for the implementation contract
- Read `commits/{{hash}}/solution.py` for code
- Read `commits/{{hash}}/validation_feedback.txt` for validation feedback
- Read `commits/{{hash}}/round_summary.json` for method/reflection summary
- Read `commits/{{hash}}/result.json` for full metadata
- Read `index/branch_summary.json` for per-branch stats
- Read `index/branch_decisions.jsonl` for branch scheduling history

Suggested workflow:
1. Follow the current branch goal; do not collapse every branch into the same generic solution
2. Review branch scoreboard and tags before opening full code
3. Use method/reflection notes to compare ideas, not just scores
4. Open only the most relevant commit details
5. Reuse strong ideas only when the search intent permits it; if the intent asks for an alternative, use best_local_cv as a baseline to beat rather than as the implementation template
"""


def build_search_intent_context(branch_decision: dict[str, Any]) -> str:
    """Explain the search intent and novelty requirements for planning."""
    intent = branch_decision.get("search_intent") or INTENT_IMPROVE_BEST
    anti_repetition = branch_decision.get("anti_repetition", {}) or {}
    avoid_category = anti_repetition.get("avoid_method_category")
    blockers = anti_repetition.get(
        "improve_blockers", anti_repetition.get("exploit_blockers", [])
    )

    common = [
        "[SEARCH INTENT CONTRACT]",
        f"Search intent: {intent}",
        f"Improve blockers: {json.dumps(blockers, ensure_ascii=False)}",
        f"Avoid method category: {avoid_category or 'none'}",
        "",
        "Every planning.md must include a `## Novelty Contract` section.",
        "Every planning.md must include a `## Runtime Hardening Contract` section.",
    ]

    if intent == INTENT_EXPLORE_ALTERNATIVE:
        common.extend(
            [
                "For this round, best_local_cv is a baseline to beat, not the implementation template.",
                "Pick a high-confidence alternative from the task-specific Kaggle skill, not an arbitrary model-family trial.",
                "The difference may be representation, model family, validation design, objective, or postprocessing, but it must be bounded.",
                "State which best-commit component is deliberately not reused.",
                "If an avoid method category is listed, do not use it as the primary method category in this alternative round.",
            ]
        )
    elif intent == INTENT_ABLATE_BEST:
        common.extend(
            [
                "For this round, isolate one important component of the best method.",
                "Remove, replace, or simplify that component while keeping the rest comparable.",
                "The plan must say what evidence would prove the component is useful or harmful.",
            ]
        )
    elif intent == INTENT_REPAIR_FAILURE:
        common.extend(
            [
                "For this round, repair the latest concrete failure with the smallest necessary change.",
                "Do not redesign the method unless the failure proves the current approach cannot work.",
                "The Novelty Contract may say `none: repair-only`, but it must identify what is intentionally unchanged.",
            ]
        )
    elif intent == INTENT_RESET_BASELINE:
        common.extend(
            [
                "For this round, implement the task-specific Kaggle skill's highest-ROI stable recipe from task/data contracts.",
                "Do not inherit fragile complexity from previous best commits.",
                "Runtime schema checks may constrain the implementation, but they must not replace task-skill modeling guidance.",
                "The Novelty Contract should state why this baseline is simpler or more robust than recent attempts.",
            ]
        )
    else:
        common.extend(
            [
                "For this round, improve best_local_cv with one or two bounded, high-confidence changes.",
                "Prefer folds, seeds, feature caps, regularization, calibration, blending, postprocessing, TTA, or fallback repairs.",
                "Do not switch the primary method category unless search intent is explore_alternative.",
                "The Novelty Contract must name the small change and the preserved anchor.",
            ]
        )

    return "\n".join(common)


def build_branch_execution_guard(branch_decision: dict[str, Any]) -> str:
    """Return branch-specific hard constraints for planning and coding."""
    branch = normalize_branch_name(str(branch_decision.get("branch") or "draft"))
    intent = branch_decision.get("search_intent") or INTENT_RESET_BASELINE
    parts = ["[BRANCH EXECUTION GUARD]"]
    if branch == "draft":
        parts.append(DRAFT_TASK_SKILL_GUARD)
    elif branch == "debug":
        parts.append(DEBUG_ERROR_GUARD)
    elif branch == "improve" and intent == INTENT_EXPLORE_ALTERNATIVE:
        parts.append(ALTERNATIVE_GUARD)
    elif branch == "improve":
        parts.append(IMPROVE_BEST_GUARD)
    parts.append(RUNTIME_HARDENING_CONTEXT)
    return "\n\n".join(parts)


def build_no_plan_placeholder(
    branch: str, branch_decision: dict[str, Any], skill_route: SkillRoute
) -> str:
    """Return archived context for branches that intentionally skip planning."""
    spec = BRANCH_SPEC_BY_NAME.get(branch, BRANCH_SPEC_BY_NAME["draft"])
    return (
        "# No Planning Phase\n\n"
        f"Branch `{branch}` intentionally skips planning and codes directly from its routed Skill.\n\n"
        "## Branch Objective\n"
        f"{spec.goal}\n\n"
        "## Search Intent\n"
        f"{branch_decision.get('search_intent', 'unknown')}\n\n"
        "## Selected Knowledge\n"
        f"{skill_route.reason}\n\n"
        "## Coding Contract\n"
        "- Use the full routed Skill directly during solution.py generation.\n"
        "- Follow the branch intent, task memory, latest failure context, and runtime hardening guard.\n"
        "- Preserve DATA_DIR loading and the submission.csv contract.\n"
    )


def resolve_harness_and_model(harness_model: str, model: str | None) -> tuple[str, str]:
    """Validate the harness/model binding and fill the harness-specific default."""
    harness = harness_model.strip().lower()
    if harness not in HARNESS_CHOICES:
        raise ValueError(
            f"Unknown harness model {harness_model!r}; choose one of {HARNESS_CHOICES}"
        )

    resolved_model = (model or "").strip() or DEFAULT_MODELS[harness]
    is_gpt = resolved_model.lower().startswith("gpt-")
    if harness == HARNESS_CODEX and not is_gpt:
        raise ValueError(
            "The codex harness is restricted to GPT models (model name must start with 'gpt-')."
        )
    if harness == HARNESS_CLAUDE_CODE and is_gpt:
        raise ValueError(
            "The claude-code harness is for non-GPT API models; use --harness-model codex for GPT models."
        )
    return harness, resolved_model


def build_harness_command(
    harness_model: str,
    model: str,
    reasoning_level: str,
    system_prompt: str,
) -> list[str]:
    """Build a credential-free CLI command for the selected harness."""
    if harness_model == HARNESS_CODEX:
        return [
            "codex",
            "exec",
            "--full-auto",
            "--ephemeral",
            "--skip-git-repo-check",
            "--model",
            model,
            "-c",
            f"reasoning_level={json.dumps(reasoning_level)}",
        ]
    if harness_model == HARNESS_CLAUDE_CODE:
        return [
            "claude",
            "--print",
            "--model",
            model,
            "--append-system-prompt",
            system_prompt,
        ]
    raise ValueError(f"Unsupported harness model: {harness_model}")


def extract_fenced_text(raw_text: str, language: str | None = None) -> str:
    """Extract the first fenced payload, falling back to the complete response."""
    if not raw_text:
        return ""
    if language:
        pattern = rf"```{re.escape(language)}\s*\n?(.*?)```"
        matches = re.findall(pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            return textwrap.dedent(matches[0]).strip()
    matches = re.findall(r"```(?:[A-Za-z0-9_+.-]+)?\s*\n?(.*?)```", raw_text, re.DOTALL)
    return textwrap.dedent(matches[0]).strip() if matches else raw_text.strip()


def extract_python_code(raw_text: str) -> str:
    """Recover a complete parseable Python program from either harness output."""
    candidates: list[str] = []
    try:
        extracted = extract_code(raw_text)
        if extracted:
            candidates.append(extracted)
    except Exception:
        pass
    candidates.extend(
        re.findall(
            r"```(?:python|py)\s*\n?(.*?)```", raw_text, re.DOTALL | re.IGNORECASE
        )
    )
    candidates.extend(re.findall(r"```\s*\n?(.*?)```", raw_text, re.DOTALL))
    candidates.append(raw_text)

    for candidate in candidates:
        code = textwrap.dedent(candidate).strip()
        if not code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        return code + ("\n" if not code.endswith("\n") else "")
    return ""


async def run_harness_prompt(
    work_dir: Path,
    prompt: str,
    system_prompt: str,
    harness_model: str,
    model: str,
    reasoning_level: str,
    max_tokens: int,
    temperature: float,
    trace_file: Path | None,
    trace_context: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Run a prompt through Codex or Claude Code with a common trace contract."""
    work_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_harness_command(
        harness_model=harness_model,
        model=model,
        reasoning_level=reasoning_level,
        system_prompt=system_prompt,
    )
    env = dict(os.environ)
    if harness_model == HARNESS_CLAUDE_CODE:
        env.pop("CLAUDECODE", None)
        env["ANTHROPIC_MODEL"] = model

    start_time = datetime.now()
    trace_data = {
        "timestamp": start_time.isoformat(),
        "harness_model": harness_model,
        "model": model,
        "reasoning_level": reasoning_level,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "work_dir": str(work_dir),
        "prompt": prompt,
        "cmd": cmd,
        "anthropic_base_url_set": bool(env.get("ANTHROPIC_BASE_URL")),
        "anthropic_api_key_set": bool(env.get("ANTHROPIC_API_KEY")),
        "response_text": "",
        "stderr": "",
        "return_code": None,
        "usage": {},
        "duration_seconds": 0,
        **trace_context,
    }

    logger.debug(
        "Running %s harness with model %s in %s", harness_model, model, work_dir
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(work_dir),
        env=env,
    )
    timeout_seconds = float(
        os.environ.get("MLE_HARNESS_TIMEOUT_SECONDS")
        or os.environ.get("CLAUDE_CLI_TIMEOUT_SECONDS")
        or 3600
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(
            f"{harness_model} harness timed out after {timeout_seconds:.0f}s"
        )

    response_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    usage = {
        "input_tokens": len(prompt) // 4,
        "output_tokens": len(response_text) // 4,
    }
    end_time = datetime.now()
    trace_data.update(
        {
            "response_text": response_text,
            "stderr": stderr_text,
            "return_code": proc.returncode,
            "usage": usage,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "end_timestamp": end_time.isoformat(),
        }
    )
    if trace_file:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            json.dumps(trace_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{harness_model} harness failed with code {proc.returncode}: {stderr_text[:500]}"
        )
    return response_text, usage


async def call_harness_cli(
    work_dir: Path,
    prompt_messages: list[dict[str, str]],
    metadata: dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    harness_model: str = HARNESS_CODEX,
    model: str = DEFAULT_MODELS[HARNESS_CODEX],
    reasoning_level: str = "high",
    max_tokens: int = 32768,
    temperature: float = 0.6,
    trace_file: Path | None = None,
    refinement_context: str | None = None,
    skill_context: str | None = None,
    phase_name: str = "coding",
) -> tuple[str, dict[str, Any]]:
    """Call the selected coding-agent CLI with a shared ERS prompt."""
    task_name = metadata.get("task_name", "unknown")
    prompt_parts = []

    for msg in prompt_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            prompt_parts.append(f"[{role.upper()}]\n{content}")

    if refinement_context:
        prompt_parts.append(f"\n[REFINEMENT CONTEXT]\n{refinement_context}")

    prompt_parts.append(build_metadata_prompt(metadata))

    prompt_sections: list[str] = []
    if harness_model == HARNESS_CODEX:
        prompt_sections.extend(["[SYSTEM INSTRUCTIONS]", system_prompt])
    elif phase_name in CLAUDE_OUTPUT_GUARDS:
        prompt_sections.extend(
            [
                "[OUTPUT CONTRACT]",
                CLAUDE_OUTPUT_GUARDS[phase_name],
            ]
        )
    if skill_context:
        prompt_sections.extend(["", "[SELECTED SKILL CONTEXT]", skill_context])
    prompt_sections.extend(["", "[USER TASK]", *prompt_parts])
    full_prompt = "\n\n".join(prompt_sections)

    return await run_harness_prompt(
        work_dir=work_dir,
        prompt=full_prompt,
        system_prompt=system_prompt,
        harness_model=harness_model,
        model=model,
        reasoning_level=reasoning_level,
        max_tokens=max_tokens,
        temperature=temperature,
        trace_file=trace_file,
        trace_context={
            "task_name": task_name,
            "prompt_messages": prompt_messages,
            "metadata": metadata,
            "phase_name": phase_name,
            "system_prompt": system_prompt,
            "refinement_context": refinement_context,
            "skill_context_used": bool(skill_context),
            "full_prompt": full_prompt,
        },
    )


async def generate_round_planning(
    work_dir: Path,
    output_dir: Path,
    round_num: int,
    prompt_messages: list[dict[str, str]],
    metadata: dict[str, Any],
    harness_model: str,
    model: str,
    reasoning_level: str,
    max_tokens: int,
    temperature: float,
    higher_is_better: bool,
    task_skills_dir: Path,
    error_skill_file: Path,
) -> tuple[str, str, dict[str, Any], SkillRoute]:
    """Generate planning.md using only branch-routed skill context."""
    planning_file = work_dir / "planning.md"
    solution_file = work_dir / "solution.py"
    task_name = metadata.get("task_name", "unknown")
    branch = get_current_branch(output_dir)
    route = route_skills_for_branch(
        task_name=task_name,
        branch=branch,
        task_skills_dir=task_skills_dir,
        error_skill_file=error_skill_file,
    )
    memory = load_memory_for_task(task_name, output_dir)
    navigation_context = build_refinement_context(
        output_dir, round_num, higher_is_better=higher_is_better
    )
    branch_decision_file = output_dir / "index" / "current_branch_decision.json"
    branch_decision = (
        branch_decision_file.read_text(encoding="utf-8")
        if branch_decision_file.exists()
        else "{}"
    )
    try:
        branch_decision_payload = json.loads(branch_decision)
    except json.JSONDecodeError:
        branch_decision_payload = {}
    planning_context = "\n\n".join(
        [
            part
            for part in [
                navigation_context,
                f"[BRANCH DECISION]\n{branch_decision}",
                build_search_intent_context(branch_decision_payload),
                build_branch_execution_guard(branch_decision_payload),
                f"[TASK MEMORY]\n{memory}",
                (
                    "[SKILL ROUTING]\n"
                    f"Selected branch: {route.branch}\n"
                    f"Routing reason: {route.reason}\n"
                    f"Sources: {json.dumps(route.sources, ensure_ascii=False)}\n"
                    "Use this routed full skill source only. Do not request or depend on other raw skills."
                ),
            ]
            if part
        ]
    )

    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    raw_text = ""
    max_plan_retries = 2

    for retry_idx in range(max_plan_retries + 1):
        retry_suffix = f"_retry_{retry_idx}" if retry_idx > 0 else ""
        trace_file = (
            output_dir
            / "traces"
            / f"round_{round_num}{retry_suffix}_planning_trace.json"
        )
        current_context = planning_context
        if retry_idx > 0:
            current_context += (
                f"\n\n[CRITICAL - PLANNING RETRY {retry_idx}/{max_plan_retries}]\n"
                "Your previous attempt did NOT create planning.md. Create exactly one file named planning.md. "
                "Do not create solution.py in planning."
            )
            if planning_file.exists():
                planning_file.unlink()
        if solution_file.exists():
            solution_file.unlink()

        raw_text, attempt_usage = await call_harness_cli(
            work_dir=work_dir,
            prompt_messages=prompt_messages,
            metadata=metadata,
            system_prompt=PLANNING_SYSTEM_PROMPT,
            harness_model=harness_model,
            model=model,
            reasoning_level=reasoning_level,
            max_tokens=max_tokens,
            temperature=temperature,
            trace_file=trace_file,
            refinement_context=current_context,
            skill_context=route.content,
            phase_name="planning",
        )
        usage["input_tokens"] += attempt_usage.get("input_tokens", 0)
        usage["output_tokens"] += attempt_usage.get("output_tokens", 0)

        if solution_file.exists():
            solution_file.unlink()

        if planning_file.exists():
            planning_text = planning_file.read_text(encoding="utf-8").strip()
            if len(planning_text) > 50:
                return planning_text, raw_text, usage, route

        response_plan = extract_fenced_text(raw_text, "markdown")
        if len(response_plan) > 50:
            planning_file.write_text(response_plan + "\n", encoding="utf-8")
            return response_plan, raw_text, usage, route

    fallback_plan = (
        "# Round Plan\n\n"
        f"## Branch Objective\n{branch}: {BRANCH_SPEC_BY_NAME.get(branch, BRANCH_SPEC_BY_NAME['draft']).goal}\n\n"
        "## Previous Problems\nPlanning generation failed; inspect memory and prior feedback before coding.\n\n"
        "## Selected Knowledge\nSkill routing was unavailable or failed to produce a plan.\n\n"
        "## Data and Submission Contracts\nRead data from DATA_DIR, infer schema at runtime, preserve test row order, and write submission.csv exactly as required.\n\n"
        "## Current Method\nImplement a robust ML baseline aligned with the task description and branch objective.\n\n"
        "## Novelty Contract\nUse this fallback as a reliable reset plan; do not copy fragile recent complexity unless required by the data contract.\n\n"
        "## Implementation Checklist\n"
        "- Load data via DATA_DIR.\n"
        "- Read sample submission when present.\n"
        "- Infer target, ID, feature, and output columns from files.\n"
        "- Align train/test features before prediction.\n"
        "- Use a compatible model and fallback.\n"
        "- Write submission.csv.\n\n"
        "## Expected Risks and Fallbacks\nIf optional libraries or columns are missing, fall back to stable sklearn/pandas paths.\n"
    )
    planning_file.write_text(fallback_plan, encoding="utf-8")
    return fallback_plan, raw_text, usage, route


def _truncate_text(text: str | None, limit: int) -> str:
    """Keep prompts bounded while preserving both beginning and end."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = max(limit // 2, 0)
    tail = max(limit - head, 0)
    return text[:head] + "\n...[truncated]...\n" + text[-tail:]


def _parse_round_summary(raw_text: str) -> dict[str, str]:
    """Parse the four compact memory fields from harness output."""
    text = raw_text.strip()
    candidates = [text]

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        method_summary = str(payload.get("method_summary", "")).strip()
        result_reflection = str(payload.get("result_reflection", "")).strip()
        method_category = str(payload.get("method_category", "")).strip()
        relative_change = str(payload.get("relative_change", "")).strip()
        if method_summary or result_reflection:
            return {
                "method_summary": method_summary[:240],
                "result_reflection": result_reflection[:240],
                "method_category": method_category[:80],
                "relative_change": relative_change[:240],
            }

    raise ValueError("Harness summary response did not contain valid JSON fields")


def classify_validation_failure(status: str, feedback: str) -> dict[str, Any]:
    """Classify common runtime failures so debug rounds stay concrete."""
    text = f"{status}\n{feedback}".lower()
    checks = [
        ("timeout", ("timeout", "timed out", "time limit", "wall time", "deadline")),
        (
            "oom",
            (
                "out of memory",
                "oom",
                "cuda error: out of memory",
                "memoryerror",
                "killed",
            ),
        ),
        (
            "dependency",
            (
                "modulenotfounderror",
                "importerror",
                "no module named",
                "not installed",
                "distributionnotfound",
            ),
        ),
        (
            "schema",
            (
                "keyerror",
                "column",
                "columns",
                "not in index",
                "feature names",
                "shape mismatch",
            ),
        ),
        (
            "submission",
            (
                "submission.csv",
                "sample_submission",
                "submission file",
                "wrong number of rows",
                "missing column",
            ),
        ),
        (
            "metric",
            (
                "metric",
                "scoring",
                "auc",
                "rmse",
                "log_loss",
                "quadratic weighted kappa",
            ),
        ),
        (
            "data_parsing",
            (
                "parsererror",
                "unicode",
                "decode",
                "bad lines",
                "file not found",
                "filenotfounderror",
                "is a directory",
            ),
        ),
        ("output_format", ("invalid", "nan", "inf", "dtype", "json", "csv", "header")),
    ]
    matched = [
        name for name, needles in checks if any(needle in text for needle in needles)
    ]
    if not matched and ("traceback" in text or "error" in text or "exception" in text):
        matched = ["runtime_exception"]
    return {
        "primary": matched[0] if matched else "unknown",
        "all": matched,
        "debug_instruction": (
            "Repair the classified failure with the smallest necessary code change; preserve the current method unless "
            "the taxonomy and feedback prove it cannot run."
        ),
    }


def inspect_solution_contract(code: str) -> dict[str, Any]:
    """Lightweight static contract check before sandbox validation."""
    checks = {
        "uses_data_dir_env": "DATA_DIR" in code and "os.environ" in code,
        "writes_submission_csv": "submission.csv" in code,
        "mentions_sample_submission": "sample_submission" in code.lower(),
        "has_dependency_fallback": any(
            token in code
            for token in (
                "ImportError",
                "ModuleNotFoundError",
                "try:",
                "except ImportError",
            )
        ),
        "has_output_validation": any(
            token in code.lower()
            for token in ("validate", "assert", "columns", "shape")
        ),
        "has_resource_downgrade_hint": any(
            token in code.lower()
            for token in (
                "fallback",
                "sample",
                "n_estimators",
                "epochs",
                "timeout",
                "memory",
            )
        ),
    }
    missing = [name for name, ok in checks.items() if not ok]
    return {
        "checks": checks,
        "missing": missing,
        "status": "pass" if not missing else "warn",
    }


async def call_harness_round_summary(
    work_dir: Path,
    metadata: dict[str, Any],
    round_num: int,
    solution_code: str,
    validation_feedback: str,
    validation_status: str,
    validation_score: float | None,
    harness_model: str,
    model: str,
    reasoning_level: str,
    trace_file: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Ask the selected harness for notes before archiving a commit."""
    work_dir.mkdir(parents=True, exist_ok=True)

    score_text = f"{validation_score:.6f}" if validation_score is not None else "N/A"
    prompt = f"""
You are summarizing one completed ML benchmark attempt for future search.
Do not create, edit, or delete files. Only print compact JSON.

Return exactly this JSON object:
{{
  "method_summary": "main model family, feature pipeline, validation split, and output writer",
  "result_reflection": "whether the run failed, improved, or exposed a new task fact",
  "method_category": "short category such as linear model, tree ensemble, CNN, transformer, or post-processing",
  "relative_change": "one short sentence describing what changed from the current best program and what stayed fixed"
}}

Keep values concise, factual, and useful for future branch scheduling. Avoid over-indexing on the score alone.

Task: {metadata.get("task_name", "unknown")}
Round: {round_num + 1}
Higher is better: {metadata.get("higher_is_better", "unknown")}
Validation status: {validation_status}
Validation score: {score_text}

[SOLUTION.PY]
{_truncate_text(solution_code, 12000)}

[VALIDATION FEEDBACK]
{_truncate_text(validation_feedback, 6000)}
""".strip()

    response_text, usage = await run_harness_prompt(
        work_dir=work_dir,
        prompt=prompt,
        system_prompt="Return only the requested compact JSON and do not modify files.",
        harness_model=harness_model,
        model=model,
        reasoning_level=reasoning_level,
        max_tokens=2048,
        temperature=0.0,
        trace_file=trace_file,
        trace_context={
            "task_name": metadata.get("task_name", "unknown"),
            "round": round_num,
            "phase_name": "summary",
        },
    )
    return _parse_round_summary(response_text), usage


async def update_memory_after_round(
    task_dir: Path,
    metadata: dict[str, Any],
    round_num: int,
    commit_hash: str,
    branch: str,
    planning_text: str,
    solution_code: str,
    validation_feedback: str,
    validation_status: str,
    validation_score: float | None,
    round_summary: dict[str, str],
    harness_model: str,
    model: str,
    reasoning_level: str,
) -> dict[str, Any]:
    """Create or revise the per-task memory after validation."""
    task_name = metadata.get("task_name", "unknown")
    memories_dir = task_dir / "memories"
    memories_dir.mkdir(exist_ok=True)
    memory_file = memory_file_for_task(task_name, task_dir)
    previous_memory = _safe_read_text(memory_file, limit=20000) or ""

    if memory_file.exists():
        backup_file = (
            memories_dir / f"MEMORY_{task_name}_before_round_{round_num + 1}.md"
        )
        backup_file.write_text(
            memory_file.read_text(encoding="utf-8"), encoding="utf-8"
        )

    score_text = f"{validation_score:.6f}" if validation_score is not None else "N/A"
    prompt = f"""
You are updating a compact task memory for future ML benchmark attempts.

Rules:
- Only create or rewrite `{memory_file.name}` in the memories directory.
- Do not modify original skill files, commits, code, or planning files.
- Keep the memory concise and useful for branch selection and future planning.
- Prefer evidence from this validation over generic advice.

Required memory sections:
# Task Memory
## Stable Facts
## Best Known Approach
## Failed Attempts
## Validation and Submission Gotchas
## Next-Round Advice

Task: {task_name}
Round: {round_num + 1}
Commit: {commit_hash}
Branch: {branch}
Validation status: {validation_status}
Validation score: {score_text}
Higher is better: {metadata.get("higher_is_better", "unknown")}

[PREVIOUS MEMORY]
{_truncate_text(previous_memory, 12000) if previous_memory else "No previous memory."}

[ROUND SUMMARY]
{json.dumps(round_summary, ensure_ascii=False, indent=2)}

[PLANNING.MD]
{_truncate_text(planning_text, 12000)}

[SOLUTION.PY]
{_truncate_text(solution_code, 16000)}

[VALIDATION FEEDBACK]
{_truncate_text(validation_feedback, 12000)}
""".strip()

    trace_file = task_dir / "traces" / f"round_{round_num}_memory_update_trace.json"
    response_text, usage = await run_harness_prompt(
        work_dir=memories_dir,
        prompt=(
            f"[OUTPUT CONTRACT]\n{CLAUDE_OUTPUT_GUARDS['memory']}\n\n{prompt}"
            if harness_model == HARNESS_CLAUDE_CODE
            else prompt
        ),
        system_prompt="Maintain compact evidence-based task memory and follow the requested output contract.",
        harness_model=harness_model,
        model=model,
        reasoning_level=reasoning_level,
        max_tokens=12000,
        temperature=0.0,
        trace_file=trace_file,
        trace_context={
            "task_name": task_name,
            "round": round_num,
            "commit_hash": commit_hash,
            "branch": branch,
            "memory_file": str(memory_file),
            "phase_name": "memory",
        },
    )
    if harness_model == HARNESS_CLAUDE_CODE:
        memory_text = extract_fenced_text(response_text, "markdown")
        if memory_text:
            memory_file.write_text(memory_text.rstrip() + "\n", encoding="utf-8")
    if not memory_file.exists() or not memory_file.read_text(encoding="utf-8").strip():
        fallback = (
            "# Task Memory\n\n"
            "## Stable Facts\n"
            f"- Task: {task_name}\n\n"
            "## Best Known Approach\n"
            f"- Round {round_num + 1} branch `{branch}` status `{validation_status}` score `{score_text}`.\n\n"
            "## Failed Attempts\n"
            "- Inspect validation feedback for concrete failures.\n\n"
            "## Validation and Submission Gotchas\n"
            "- Preserve DATA_DIR loading and submission.csv contract.\n\n"
            "## Next-Round Advice\n"
            "- Use branch state, validation feedback, and this memory to choose the next focused change.\n"
        )
        memory_file.write_text(fallback, encoding="utf-8")
    return {"status": "updated", "usage": usage, "memory_file": str(memory_file)}


async def evaluate_single_task_single_round(
    sandbox_client: httpx.AsyncClient,
    task: dict[str, Any],
    round_num: int,
    output_dir: Path,
    harness_model: str,
    model: str,
    reasoning_level: str,
    max_tokens: int,
    temperature: float,
    best_score_so_far: float | None = None,
    higher_is_better: bool = True,
    task_skills_dir: Path = DEFAULT_TASK_SKILLS_DIR,
    error_skill_file: Path = DEFAULT_ERROR_SKILL_FILE,
) -> dict[str, Any]:
    """
    Evaluate a single task for one round using search-oriented storage.
    """
    round_start_time = time.time()

    if "metadata" not in task or "prompt" not in task:
        raise ValueError(
            f"Task must have 'metadata' and 'prompt' fields, got: {list(task.keys())}"
        )

    metadata = task["metadata"]
    task_name = metadata["task_name"]

    prompt_messages = task["prompt"]
    if hasattr(prompt_messages, "tolist"):
        prompt_messages = prompt_messages.tolist()

    resource_type = metadata.get("cpu_gpu", "cpu")
    resource_type = "cpu" if str(resource_type).lower() == "cpu" else "gpu"
    val_data_dir = metadata.get("data_dir", "")

    logger.info("[%s] Round %s - Starting", task_name, round_num + 1)

    work_dir = output_dir
    planning_file = work_dir / "planning.md"
    solution_file = work_dir / "solution.py"
    current_branch = get_current_branch(output_dir)
    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

    branch_decision_file = output_dir / "index" / "current_branch_decision.json"
    branch_decision = (
        json.loads(branch_decision_file.read_text(encoding="utf-8"))
        if branch_decision_file.exists()
        else {"branch": get_current_branch(output_dir)}
    )
    coding_skill_context: str | None = None
    if branch_uses_planning(current_branch):
        try:
            planning_text, planning_raw_text, planning_usage, skill_route = (
                await generate_round_planning(
                    work_dir=work_dir,
                    output_dir=output_dir,
                    round_num=round_num,
                    prompt_messages=prompt_messages,
                    metadata=metadata,
                    harness_model=harness_model,
                    model=model,
                    reasoning_level=reasoning_level,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    higher_is_better=higher_is_better,
                    task_skills_dir=task_skills_dir,
                    error_skill_file=error_skill_file,
                )
            )
            usage["input_tokens"] += planning_usage.get("input_tokens", 0)
            usage["output_tokens"] += planning_usage.get("output_tokens", 0)
        except Exception as e:
            logger.error(
                "[%s] Round %s - Planning failed: %s",
                task_name,
                round_num + 1,
                e,
            )
            return {
                "round": round_num,
                "task_name": task_name,
                "error": str(e),
                "status": "planning_error",
                "round_wall_time": time.time() - round_start_time,
                **usage,
            }

        refinement_context = "\n\n".join(
            [
                part
                for part in [
                    build_refinement_context(
                        output_dir, round_num, higher_is_better=higher_is_better
                    ),
                    (
                        "[ROUND PLANNING]\n"
                        "The following planning.md is the Skill-derived contract for draft coding. "
                        "Do not request or rely on raw Skill files. Implement solution.py from this plan.\n\n"
                        f"{planning_text}"
                    ),
                    build_branch_execution_guard(branch_decision),
                ]
                if part
            ]
        )
    else:
        if planning_file.exists():
            planning_file.unlink()
        if solution_file.exists():
            solution_file.unlink()
        skill_route = route_skills_for_branch(
            task_name=task_name,
            branch=current_branch,
            task_skills_dir=task_skills_dir,
            error_skill_file=error_skill_file,
        )
        planning_text = build_no_plan_placeholder(
            current_branch, branch_decision, skill_route
        )
        planning_file.write_text(planning_text, encoding="utf-8")
        planning_raw_text = ""
        coding_skill_context = skill_route.content
        memory = load_memory_for_task(task_name, output_dir)
        refinement_context = "\n\n".join(
            [
                part
                for part in [
                    build_refinement_context(
                        output_dir, round_num, higher_is_better=higher_is_better
                    ),
                    f"[BRANCH DECISION]\n{json.dumps(branch_decision, indent=2, ensure_ascii=False)}",
                    build_search_intent_context(branch_decision),
                    build_branch_execution_guard(branch_decision),
                    f"[TASK MEMORY]\n{memory}",
                    (
                        "[DIRECT SKILL CODING]\n"
                        f"Branch `{current_branch}` skips planning. "
                        "Use the selected full routed Skill directly while creating solution.py.\n"
                        f"Routing reason: {skill_route.reason}\n"
                        f"Sources: {json.dumps(skill_route.sources, ensure_ascii=False)}"
                    ),
                ]
                if part
            ]
        )

    max_no_solution_retries = 3
    response_text = ""

    for retry_idx in range(max_no_solution_retries + 1):
        retry_suffix = f"_retry_{retry_idx}" if retry_idx > 0 else ""
        trace_file = (
            output_dir / "traces" / f"round_{round_num}{retry_suffix}_trace.json"
        )

        current_refinement = refinement_context
        if retry_idx > 0:
            retry_hint = (
                f"\n\n[CRITICAL - RETRY {retry_idx}/{max_no_solution_retries}]\n"
                "Your previous attempt did NOT produce a solution.py file.\n"
                "You MUST create a file named `solution.py` in the current working directory.\n"
                "Write the complete solution code to solution.py.\n"
                "Do NOT output the code only as markdown in your response."
            )
            current_refinement = (current_refinement or "") + retry_hint
            if solution_file.exists():
                solution_file.unlink()

        try:
            response_text, attempt_usage = await call_harness_cli(
                work_dir=work_dir,
                prompt_messages=prompt_messages,
                metadata=metadata,
                system_prompt=SYSTEM_PROMPT,
                harness_model=harness_model,
                model=model,
                reasoning_level=reasoning_level,
                max_tokens=max_tokens,
                temperature=temperature,
                trace_file=trace_file,
                refinement_context=current_refinement,
                skill_context=coding_skill_context,
                phase_name="coding",
            )
            usage["input_tokens"] += attempt_usage.get("input_tokens", 0)
            usage["output_tokens"] += attempt_usage.get("output_tokens", 0)
        except Exception as e:
            logger.error(
                "[%s] Round %s attempt %s - CLI call failed: %s",
                task_name,
                round_num + 1,
                retry_idx + 1,
                e,
            )
            if retry_idx == max_no_solution_retries:
                return {
                    "round": round_num,
                    "task_name": task_name,
                    "error": str(e),
                    "status": "agent_error",
                }
            continue

        if solution_file.exists():
            break

        logger.info(
            "[%s] Round %s attempt %s - solution.py not found, extracting from response",
            task_name,
            round_num + 1,
            retry_idx + 1,
        )
        code = extract_python_code(response_text)
        if code and len(code.strip()) > 100:
            solution_file.write_text(code, encoding="utf-8")
            logger.info(
                "[%s] Round %s - Extracted %s chars from response",
                task_name,
                round_num + 1,
                len(code),
            )
            break

        if retry_idx < max_no_solution_retries:
            logger.warning(
                "[%s] Round %s - No solution on attempt %s, retrying (%s/%s)",
                task_name,
                round_num + 1,
                retry_idx + 1,
                retry_idx + 1,
                max_no_solution_retries,
            )
            continue

        logger.error(
            "[%s] Round %s - No code after %s attempts",
            task_name,
            round_num + 1,
            max_no_solution_retries + 1,
        )
        return {
            "round": round_num,
            "task_name": task_name,
            "branch": get_current_branch(output_dir),
            "branch_decision": branch_decision,
            "raw_text": response_text,
            "planning": planning_text,
            "planning_raw_text": planning_raw_text,
            "skill_route": {
                "branch": skill_route.branch,
                "reason": skill_route.reason,
                "sources": skill_route.sources,
            },
            "error": f"solution.py not created after {max_no_solution_retries + 1} attempts",
            "status": "no_solution",
            "round_wall_time": time.time() - round_start_time,
            **usage,
        }

    code = solution_file.read_text(encoding="utf-8")
    solution_contract = inspect_solution_contract(code)
    if solution_contract["status"] != "pass":
        logger.warning(
            "[%s] Round %s - solution.py contract warnings: %s",
            task_name,
            round_num + 1,
            solution_contract["missing"],
        )

    logger.info(
        "[%s] Round %s - Running validation with resource_type=%s",
        task_name,
        round_num + 1,
        resource_type,
    )
    val_ctx = EvalContext(phase="validation", data_dir=val_data_dir)
    val_status_code, val_payload = await get_sandbox_result(
        client=sandbox_client,
        code_str=code,
        data_dir=val_ctx.data_dir,
        resource_type=resource_type,
        job_timeout=86400,
        wait_timeout=186400,
        poll_interval=5,
    )

    val_job_id = val_payload.get("job_id")
    val_result_payload = val_payload.get("result") or {}
    val_status = str(val_result_payload.get("result") or "unknown")
    val_score_value = val_result_payload.get("score")
    val_score = (
        float(val_score_value)
        if (val_status_code == 200 and val_score_value is not None)
        else None
    )
    val_reward = (
        score2reward(val_score, metadata, mode="power_sigmoid")
        if val_score is not None
        else 0.0
    )

    val_queue_time = None
    val_run_time = None
    if val_status_code == 200:
        created_at = val_payload.get("created_at")
        started_at = val_payload.get("started_at")
        completed_at = val_payload.get("completed_at")
        if started_at and completed_at:
            started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            completed_ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            val_run_time = (completed_ts - started_ts).total_seconds()
            if created_at:
                created_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                val_queue_time = (started_ts - created_ts).total_seconds()

    val_run_log = val_result_payload.get("run_log")
    val_clear_log = get_clear_log(val_run_log)
    val_feedback = format_sandbox_feedback(val_status_code, val_payload)
    failure_taxonomy = (
        classify_validation_failure(val_status, val_feedback)
        if val_score is None
        else {
            "primary": "none",
            "all": [],
            "debug_instruction": "Validation produced a score; debug taxonomy is not active.",
        }
    )

    round_wall_time = time.time() - round_start_time
    current_branch = get_current_branch(output_dir)
    branch_decision_file = output_dir / "index" / "current_branch_decision.json"
    branch_decision = (
        json.loads(branch_decision_file.read_text(encoding="utf-8"))
        if branch_decision_file.exists()
        else {}
    )
    result = {
        "round": round_num,
        "task_name": task_name,
        "branch": current_branch,
        "branch_decision": branch_decision,
        "planning": planning_text,
        "planning_raw_text": planning_raw_text,
        "skill_route": {
            "branch": skill_route.branch,
            "reason": skill_route.reason,
            "sources": skill_route.sources,
        },
        "raw_text": response_text,
        "code": code,
        "solution_contract": solution_contract,
        **usage,
        "round_wall_time": round_wall_time,
        "validation": {
            "phase": "validation",
            "status_code": val_status_code,
            "status": val_status,
            "score": val_score,
            "reward": val_reward,
            "failure_taxonomy": failure_taxonomy,
            "job_id": val_job_id,
            "queue_time": val_queue_time,
            "run_time": val_run_time,
            "raw_run_log": val_run_log,
            "clear_run_log": val_clear_log,
            "feedback": val_feedback,
        },
    }

    summary_trace_file = output_dir / "traces" / f"round_{round_num}_summary_trace.json"
    try:
        round_summary, summary_usage = await call_harness_round_summary(
            work_dir=output_dir / "traces",
            metadata=metadata,
            round_num=round_num,
            solution_code=code,
            validation_feedback=val_feedback,
            validation_status=val_status,
            validation_score=val_score,
            harness_model=harness_model,
            model=model,
            reasoning_level=reasoning_level,
            trace_file=summary_trace_file,
        )
        usage["input_tokens"] += summary_usage.get("input_tokens", 0)
        usage["output_tokens"] += summary_usage.get("output_tokens", 0)
        result["summary_usage"] = summary_usage
    except Exception as e:
        logger.warning(
            "[%s] Round %s - Summary call failed: %s", task_name, round_num + 1, e
        )
        score_text = f"{val_score:.4f}" if val_score is not None else "N/A"
        round_summary = {
            "method_summary": "Harness summary unavailable; inspect solution.py for method details.",
            "result_reflection": f"Validation status={val_status}, score={score_text}; inspect feedback before reusing this attempt.",
            "method_category": "",
            "relative_change": "",
        }

    result["round_summary"] = round_summary
    result["input_tokens"] = usage["input_tokens"]
    result["output_tokens"] = usage["output_tokens"]

    commit_hash = create_commit_hash(round_num, datetime.now().isoformat())
    result["commit_hash"] = commit_hash
    save_commit(
        output_dir,
        commit_hash,
        planning_text,
        code,
        val_feedback,
        result,
        round_summary,
    )

    if solution_file.exists():
        solution_file.unlink()
    if planning_file.exists():
        planning_file.unlink()

    if val_score is not None:
        commit_msg = f"Round {round_num + 1}: {val_status} score={val_score:.4f}"
    else:
        commit_msg = f"Round {round_num + 1}: {val_status}"

    commit_status = "success" if val_score is not None else val_status
    update_commit_log(
        output_dir,
        commit_hash,
        current_branch,
        commit_msg,
        val_score,
        round_wall_time,
        commit_status,
        round_summary,
    )
    update_branch_ref(
        task_dir=output_dir,
        branch=current_branch,
        commit_hash=commit_hash,
        score=val_score,
        status=commit_status,
        wall_time=round_wall_time,
        round_num=round_num,
        higher_is_better=higher_is_better,
    )

    if val_score is not None:
        is_best = best_score_so_far is None or (
            (higher_is_better and val_score > best_score_so_far)
            or (not higher_is_better and val_score < best_score_so_far)
        )
        if is_best:
            create_tag(
                output_dir,
                "best_local_cv",
                commit_hash,
                f"Best validation score so far: {val_score:.4f}",
                val_score,
                current_branch,
            )
            logger.info(
                "[%s] Round %s - New best, tagged as best_local_cv",
                task_name,
                round_num + 1,
            )

        branch_tag = f"best_{current_branch}"
        branch_summary = load_branch_summary(output_dir).get(current_branch, {})
        if (
            branch_summary.get("head") == commit_hash
            and branch_summary.get("best_score") == val_score
        ):
            create_tag(
                output_dir,
                branch_tag,
                commit_hash,
                f"Best score on branch {current_branch}: {val_score:.4f}",
                val_score,
                current_branch,
            )

    if val_score is not None:
        logger.info(
            "[%s] Round %s - val_score=%.6f, commit=%s",
            task_name,
            round_num + 1,
            val_score,
            commit_hash,
        )
    else:
        logger.info(
            "[%s] Round %s - Validation failed, commit=%s",
            task_name,
            round_num + 1,
            commit_hash,
        )

    try:
        memory_result = await update_memory_after_round(
            task_dir=output_dir,
            metadata=metadata,
            round_num=round_num,
            commit_hash=commit_hash,
            branch=current_branch,
            planning_text=planning_text,
            solution_code=code,
            validation_feedback=val_feedback,
            validation_status=val_status,
            validation_score=val_score,
            round_summary=round_summary,
            harness_model=harness_model,
            model=model,
            reasoning_level=reasoning_level,
        )
        result["memory_update"] = memory_result
        memory_usage = (
            memory_result.get("usage", {}) if isinstance(memory_result, dict) else {}
        )
        result["input_tokens"] += memory_usage.get("input_tokens", 0)
        result["output_tokens"] += memory_usage.get("output_tokens", 0)
        commit_dir = output_dir / "commits" / commit_hash
        if commit_dir.exists():
            (commit_dir / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
    except Exception as e:
        logger.warning(
            "[%s] Round %s - Memory update failed: %s", task_name, round_num + 1, e
        )

    logger.info(
        "[%s] Round %s wall time: %.2fs", task_name, round_num + 1, round_wall_time
    )
    return result


async def submit_best_round(
    sandbox_client: httpx.AsyncClient,
    best_round: dict[str, Any],
    task: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run submit phase for the best validation round."""
    metadata = task["metadata"]
    task_name = metadata["task_name"]
    resource_type = metadata.get("cpu_gpu", "cpu")
    resource_type = "cpu" if str(resource_type).lower() == "cpu" else "gpu"
    val_data_dir = metadata.get("data_dir", "")

    round_num = best_round["round"]
    code = best_round["code"]
    val_score = best_round["validation"]["score"]
    commit_hash = best_round.get("commit_hash", "unknown")
    current_branch = get_current_branch(output_dir)

    logger.info(
        "[%s] Submitting best round %s (val_score=%.6f, commit=%s)",
        task_name,
        round_num + 1,
        val_score,
        commit_hash,
    )

    submit_data_dir = resolve_submit_data_dir(val_data_dir)
    submit_ctx = EvalContext(phase="submit", data_dir=submit_data_dir)
    submit_status_code, submit_payload = await get_sandbox_result(
        client=sandbox_client,
        code_str=code,
        data_dir=submit_ctx.data_dir,
        resource_type=resource_type,
        job_timeout=86400,
        wait_timeout=86400,
        poll_interval=5,
    )

    submit_job_id = submit_payload.get("job_id")
    submit_result_payload = submit_payload.get("result") or {}
    submit_status = str(submit_result_payload.get("result") or "unknown")
    submit_score_value = submit_result_payload.get("score")
    submit_score = (
        float(submit_score_value)
        if (submit_status_code == 200 and submit_score_value is not None)
        else None
    )
    submit_reward = (
        score2reward(submit_score, metadata, mode="power_sigmoid")
        if submit_score is not None
        else 0.0
    )

    submit_queue_time = None
    submit_run_time = None
    if submit_status_code == 200:
        created_at = submit_payload.get("created_at")
        started_at = submit_payload.get("started_at")
        completed_at = submit_payload.get("completed_at")
        if started_at and completed_at:
            started_ts = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            completed_ts = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            submit_run_time = (completed_ts - started_ts).total_seconds()
            if created_at:
                created_ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                submit_queue_time = (started_ts - created_ts).total_seconds()

    submit_run_log = submit_result_payload.get("run_log")
    submit_clear_log = get_clear_log(submit_run_log)
    submit_feedback = format_sandbox_feedback(submit_status_code, submit_payload)

    leaderboard = eval_utils.load_leaderboard(
        {**metadata, "leaderboard_dir": LEADERBOARD_DIR}
    )
    submit_grade, submit_medal = eval_utils.build_submit_grade_and_medal(
        submit_score, leaderboard
    )

    commit_dir = output_dir / "commits" / commit_hash
    if commit_dir.exists():
        (commit_dir / "submit_feedback.txt").write_text(
            submit_feedback, encoding="utf-8"
        )

    best_round["submit"] = {
        "phase": "submit",
        "status_code": submit_status_code,
        "status": submit_status,
        "score": submit_score,
        "reward": submit_reward,
        "grade": submit_grade,
        "medal": submit_medal,
        "job_id": submit_job_id,
        "queue_time": submit_queue_time,
        "run_time": submit_run_time,
        "raw_run_log": submit_run_log,
        "clear_run_log": submit_clear_log,
        "feedback": submit_feedback,
    }

    if commit_dir.exists():
        (commit_dir / "result.json").write_text(
            json.dumps(best_round, indent=2), encoding="utf-8"
        )

    if submit_score is not None:
        create_tag(
            output_dir,
            "best_public_lb",
            commit_hash,
            f"Submitted score: {submit_score:.4f}, grade: {submit_grade}, medal: {submit_medal}",
            submit_score,
            current_branch,
        )

    logger.info(
        "[%s] Submit complete (submit_score=%s, grade=%s, medal=%s)",
        task_name,
        submit_score,
        submit_grade,
        submit_medal,
    )
    return best_round


async def evaluate_single_task_multi_rounds(
    sandbox_client: httpx.AsyncClient,
    task: dict[str, Any],
    output_dir: Path,
    harness_model: str,
    model: str,
    reasoning_level: str,
    max_tokens: int,
    temperature: float,
    num_rounds: int,
    time_budget: float = 43200.0,
    branch_strategy: str = "adaptive",
    warmup_branches: tuple[str, ...] = DEFAULT_WARMUP_BRANCHES,
    task_skills_dir: Path = DEFAULT_TASK_SKILLS_DIR,
    error_skill_file: Path = DEFAULT_ERROR_SKILL_FILE,
) -> list[dict[str, Any]]:
    """Evaluate a single task across multiple rounds."""
    task_name = (
        task["metadata"]["task_name"] if "metadata" in task else task["task_name"]
    )
    logger.info(
        "[%s] Starting multi-round evaluation (%s rounds, time_budget=%ss)",
        task_name,
        num_rounds,
        time_budget,
    )

    init_git_structure(output_dir)
    set_current_branch(output_dir, "draft")

    metadata = task["metadata"]
    higher_is_better = metadata.get("higher_is_better", True)

    all_rounds: list[dict[str, Any]] = []
    total_time = 0.0
    best_score_so_far = None
    summary_file = output_dir / "rounds_summary.json"

    for round_num in range(num_rounds):
        if total_time >= time_budget:
            logger.info(
                "[%s] Time budget exceeded (%.2fs >= %.2fs), stopping at round %s",
                task_name,
                total_time,
                time_budget,
                round_num,
            )
            break

        branch_decision = choose_branch_for_round(
            task_dir=output_dir,
            round_num=round_num,
            all_rounds=all_rounds,
            higher_is_better=higher_is_better,
            branch_strategy=branch_strategy,
            warmup_branches=warmup_branches,
        )
        set_current_branch(output_dir, branch_decision["branch"])
        logger.info(
            "[%s] Round %s branch=%s reason=%s",
            task_name,
            round_num + 1,
            branch_decision["branch"],
            branch_decision["reason"],
        )

        round_result = await evaluate_single_task_single_round(
            sandbox_client=sandbox_client,
            task=task,
            round_num=round_num,
            output_dir=output_dir,
            harness_model=harness_model,
            model=model,
            reasoning_level=reasoning_level,
            max_tokens=max_tokens,
            temperature=temperature,
            best_score_so_far=best_score_so_far,
            higher_is_better=higher_is_better,
            task_skills_dir=task_skills_dir,
            error_skill_file=error_skill_file,
        )
        if "branch" not in round_result:
            round_result["branch"] = branch_decision["branch"]
            round_result["branch_decision"] = branch_decision
        all_rounds.append(round_result)

        if "commit_hash" not in round_result:
            record_branch_attempt_without_commit(
                task_dir=output_dir,
                branch=branch_decision["branch"],
                status=str(round_result.get("status", "no_commit")),
                wall_time=float(round_result.get("round_wall_time", 0.0) or 0.0),
                round_num=round_num,
            )

        val_score = round_result.get("validation", {}).get("score")
        if val_score is not None:
            if best_score_so_far is None:
                best_score_so_far = val_score
            elif higher_is_better and val_score > best_score_so_far:
                best_score_so_far = val_score
            elif not higher_is_better and val_score < best_score_so_far:
                best_score_so_far = val_score

        round_time = round_result.get("validation", {}).get(
            "run_time"
        ) or round_result.get("round_wall_time", 0.0)
        total_time += round_time
        logger.info(
            "[%s] Round %s complete, round_time=%.2fs, total_time=%.2fs",
            task_name,
            round_num + 1,
            round_time,
            total_time,
        )

        summary_data = {
            "rounds": all_rounds,
            "total_time": total_time,
            "time_budget": time_budget,
            "branch_strategy": branch_strategy,
            "branch_summary": load_branch_summary(output_dir),
        }
        summary_file.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    logger.info(
        "[%s] Completed %s rounds (total_time=%.2fs)",
        task_name,
        len(all_rounds),
        total_time,
    )

    tag_registry_file = output_dir / "index" / "tag_registry.json"
    if tag_registry_file.exists():
        registry = json.loads(tag_registry_file.read_text(encoding="utf-8"))
        if "best_local_cv" in registry:
            best_commit_hash = registry["best_local_cv"]["commit"]
            best_round = next(
                (r for r in all_rounds if r.get("commit_hash") == best_commit_hash),
                None,
            )
            if best_round:
                logger.info(
                    "[%s] Best round: %s (val_score=%.6f, commit=%s)",
                    task_name,
                    best_round["round"] + 1,
                    best_round["validation"]["score"],
                    best_commit_hash,
                )
                await submit_best_round(
                    sandbox_client=sandbox_client,
                    best_round=best_round,
                    task=task,
                    output_dir=output_dir,
                )
                summary_data = {
                    "rounds": all_rounds,
                    "total_time": total_time,
                    "time_budget": time_budget,
                    "branch_strategy": branch_strategy,
                    "branch_summary": load_branch_summary(output_dir),
                    "best_round": best_round["round"],
                    "best_commit": best_commit_hash,
                }
                summary_file.write_text(
                    json.dumps(summary_data, indent=2), encoding="utf-8"
                )
            else:
                logger.warning(
                    "[%s] best_local_cv tag exists but commit not found in rounds",
                    task_name,
                )
        else:
            logger.warning("[%s] No best_local_cv tag, skipping submit", task_name)
    else:
        logger.warning("[%s] No tag registry, skipping submit", task_name)

    return all_rounds


async def evaluate_tasks_concurrent(
    sandbox_client: httpx.AsyncClient,
    tasks: list[dict[str, Any]],
    output_path: Path,
    harness_model: str,
    model: str,
    reasoning_level: str,
    max_tokens: int,
    temperature: float,
    num_rounds: int,
    concurrency: int,
    time_budget: float = 43200.0,
    branch_strategy: str = "adaptive",
    warmup_branches: tuple[str, ...] = DEFAULT_WARMUP_BRANCHES,
    task_skills_dir: Path = DEFAULT_TASK_SKILLS_DIR,
    error_skill_file: Path = DEFAULT_ERROR_SKILL_FILE,
) -> list[dict[str, Any]]:
    """Evaluate multiple tasks concurrently."""
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_with_semaphore(task: dict[str, Any]) -> list[dict[str, Any]]:
        async with semaphore:
            task_name = (
                task["metadata"]["task_name"]
                if "metadata" in task
                else task["task_name"]
            )
            task_output_dir = output_path / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            return await evaluate_single_task_multi_rounds(
                sandbox_client=sandbox_client,
                task=task,
                output_dir=task_output_dir,
                harness_model=harness_model,
                model=model,
                reasoning_level=reasoning_level,
                max_tokens=max_tokens,
                temperature=temperature,
                num_rounds=num_rounds,
                time_budget=time_budget,
                branch_strategy=branch_strategy,
                warmup_branches=warmup_branches,
                task_skills_dir=task_skills_dir,
                error_skill_file=error_skill_file,
            )

    task_results = await asyncio.gather(
        *[evaluate_with_semaphore(task) for task in tasks],
        return_exceptions=True,
    )

    all_results = []
    for idx, result in enumerate(task_results):
        if isinstance(result, Exception):
            task_name = (
                tasks[idx]["metadata"]["task_name"]
                if "metadata" in tasks[idx]
                else tasks[idx]["task_name"]
            )
            logger.error("Task %s failed: %s", task_name, result)
            all_results.append(
                {
                    "task_name": task_name,
                    "error": str(result),
                    "status": "task_error",
                }
            )
        else:
            all_results.extend(result)

    return all_results


async def main(
    output_dir: str,
    data_file: str = str(DEFAULT_DATA_FILE),
    harness_model: str = HARNESS_CODEX,
    model: str | None = None,
    reasoning_level: str = "high",
    max_tokens: int = 100000,
    temperature: float = 0.6,
    num_rounds: int = 3,
    concurrency: int = 1,
    time_budget: float = 43200.0,
    branch_strategy: str = "adaptive",
    warmup_branches: str = ",".join(DEFAULT_WARMUP_BRANCHES),
    task_skills_dir: str = str(DEFAULT_TASK_SKILLS_DIR),
    error_skill_file: str = str(DEFAULT_ERROR_SKILL_FILE),
) -> None:
    """Main evaluation function."""
    harness_model, model = resolve_harness_and_model(harness_model, model)
    data_path = Path(data_file)
    if data_path.suffix == ".json":
        tasks = json.loads(data_path.read_text(encoding="utf-8"))
    elif data_path.suffix == ".parquet":
        tasks = pd.read_parquet(data_path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info("Loaded %s tasks from %s", len(tasks), data_file)
    raw_warmup = [
        branch.strip() for branch in warmup_branches.split(",") if branch.strip()
    ]
    warmup_tuple = normalize_branch_sequence(raw_warmup)
    invalid_warmup = [
        branch for branch in warmup_tuple if branch not in BRANCH_SPEC_BY_NAME
    ]
    if invalid_warmup:
        raise ValueError(
            f"Unknown warmup branch(es): {invalid_warmup}. Valid branches: {sorted(BRANCH_SPEC_BY_NAME)}"
        )
    if branch_strategy not in {"adaptive", "branch_cycle"}:
        raise ValueError("branch_strategy must be one of: adaptive, branch_cycle")

    logger.info(
        "Configuration: harness_model=%s, model=%s, reasoning_level=%s, rounds=%s, concurrency=%s, time_budget=%ss",
        harness_model,
        model,
        reasoning_level,
        num_rounds,
        concurrency,
        time_budget,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_task_skills_dir = Path(task_skills_dir)
    resolved_error_skill_file = Path(error_skill_file)

    config = {
        "data_file": data_file,
        "harness_model": harness_model,
        "model": model,
        "reasoning_level": reasoning_level,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_rounds": num_rounds,
        "concurrency": concurrency,
        "time_budget": time_budget,
        "branch_strategy": branch_strategy,
        "warmup_branches": list(warmup_tuple),
        "task_skills_dir": str(resolved_task_skills_dir),
        "error_skill_file": str(resolved_error_skill_file),
        "branch_specs": {
            spec.name: {
                "title": spec.title,
                "goal": spec.goal,
                "instructions": spec.instructions,
            }
            for spec in BRANCH_SPECS
        },
        "timestamp": datetime.now().isoformat(),
        "backend": BACKEND_ID,
    }
    (output_path / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    sandbox_client = httpx.AsyncClient(
        base_url=SANDBOX_BASE_URL,
        timeout=httpx.Timeout(300.0, connect=60.0),
        trust_env=False,
    )

    try:
        all_results = await evaluate_tasks_concurrent(
            sandbox_client=sandbox_client,
            tasks=tasks,
            output_path=output_path,
            harness_model=harness_model,
            model=model,
            reasoning_level=reasoning_level,
            max_tokens=max_tokens,
            temperature=temperature,
            num_rounds=num_rounds,
            concurrency=concurrency,
            time_budget=time_budget,
            branch_strategy=branch_strategy,
            warmup_branches=warmup_tuple,
            task_skills_dir=resolved_task_skills_dir,
            error_skill_file=resolved_error_skill_file,
        )
        (output_path / "summary.json").write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        logger.info("Evaluation complete. Results saved to %s", output_path)
    finally:
        await sandbox_client.aclose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Experience Routed Search for multi-round MLE-bench evaluation"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=os.environ.get("MLE_DATA_FILE", str(DEFAULT_DATA_FILE)),
        help=f"Path to task data file (JSON or Parquet; default: {DEFAULT_DATA_FILE})",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--harness-model",
        type=str,
        default=os.environ.get("MLE_HARNESS_MODEL", HARNESS_CODEX),
        choices=HARNESS_CHOICES,
        help="Coding-agent harness: codex for GPT models, claude-code for non-GPT API models",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MLE_MODEL") or None,
        help="API model name; defaults to gpt-5.4 for codex or claude-sonnet-4-6-cc for claude-code",
    )
    parser.add_argument(
        "--reasoning-level",
        type=str,
        default="high",
        help="Codex reasoning level (accepted but not passed to Claude Code)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=100000, help="Maximum tokens for generation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature for generation"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=256,
        help="Number of refinement rounds per task",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of tasks to process concurrently",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=43200.0,
        help="Total time budget per task in seconds",
    )
    parser.add_argument(
        "--branch-strategy",
        type=str,
        default="adaptive",
        choices=["adaptive", "branch_cycle"],
        help="Branch scheduler to use",
    )
    parser.add_argument(
        "--warmup-branches",
        type=str,
        default=",".join(DEFAULT_WARMUP_BRANCHES),
        help="Comma-separated branch names for initial exploration",
    )
    parser.add_argument(
        "--task-skills-dir",
        type=str,
        default=str(DEFAULT_TASK_SKILLS_DIR),
        help="Directory containing task-specific SKILL_<task>.md files",
    )
    parser.add_argument(
        "--error-skill-file",
        type=str,
        default=str(DEFAULT_ERROR_SKILL_FILE),
        help="Markdown file containing the error-prevention skill",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            data_file=args.data_file,
            output_dir=args.output_dir,
            harness_model=args.harness_model,
            model=args.model,
            reasoning_level=args.reasoning_level,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_rounds=args.num_rounds,
            concurrency=args.concurrency,
            time_budget=args.time_budget,
            branch_strategy=args.branch_strategy,
            warmup_branches=args.warmup_branches,
            task_skills_dir=args.task_skills_dir,
            error_skill_file=args.error_skill_file,
        )
    )
