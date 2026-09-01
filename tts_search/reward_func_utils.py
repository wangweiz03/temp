import asyncio
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import black
import httpx
import pandas as pd

# Configure logging format with timestamp at module level
logger = logging.getLogger("mle_agent")
if not logger.handlers:
    # Set up console handler with timestamp format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)


async def get_sandbox_result(
    client: httpx.AsyncClient,
    code_str: str,
    data_dir: str,
    *,
    resource_type: str = "all",
    job_timeout: int = 3600,
    wait_timeout: int = 86400,
    # wait_timeout: int = 100,
    poll_interval: int = 5,
):
    normalized_resource = (resource_type or "all").lower()
    sandbox_environment = {
        "EXECUTION_MODE": "shell",
        "DATA_DIR": data_dir,
        "SANDBOX_DATA_DIR": data_dir,
    }
    hf_endpoint = os.environ.get("MLE_HF_ENDPOINT")
    if hf_endpoint:
        sandbox_environment["HF_ENDPOINT"] = hf_endpoint

    payload = {
        "name": data_dir,
        "code": code_str,
        "data_dir": data_dir,
        "timeout": job_timeout,
        "resource_type": normalized_resource,
        "environment": sandbox_environment,
    }
    sandbox_api_key = os.environ.get("MLE_SANDBOX_API_KEY")
    headers = {"X-API-Key": sandbox_api_key} if sandbox_api_key else {}

    def safe_json(resp: httpx.Response):
        try:
            data = resp.json()
            return {} if data is None else data
        except Exception:
            return {}

    # Submit the job without blocking, with bounded retries.
    submit_error_retries = 0
    while True:
        try:
            submit_resp = await client.post(
                "/api/v1/jobs",
                json=payload,
                headers=headers,
            )
            if submit_resp.status_code in (502, 503, 504, 429):
                if submit_error_retries < 5:
                    submit_error_retries += 1
                    await asyncio.sleep(float(1))
                    logger.info(
                        f"WARNING: transient error {submit_resp.status_code} when submitting job, retrying..."
                    )
                    continue
            break
        except Exception as e:
            if submit_error_retries < 5:
                submit_error_retries += 1
                await asyncio.sleep(float(1))
                logger.info(
                    f"WARNING: exception {type(e).__name__} when submitting job, retrying..."
                )
                continue
            error_msg = (
                f"Failed to connect to sandbox API at {client.base_url}: "
                f"{type(e).__name__}: {str(e)}"
            )
            logger.info(f"ERROR: {error_msg}")
            return 503, {
                "error": "connection_failed",
                "detail": error_msg,
                "type": type(e).__name__,
            }
    submit_data = safe_json(submit_resp)
    logger.info("submit_data: %s", submit_data)
    if submit_resp.status_code >= 400:
        return submit_resp.status_code, {"error": "submit failed", "data": submit_data}

    job_id = submit_data.get("job_id")
    if not job_id:
        return 500, {"error": "no job_id returned", "data": submit_data}

    # Poll until the job finishes or the overall wait timeout expires.
    wait_status = ["running", "queued"]
    deadline = time.monotonic() + wait_timeout
    poll_error_retries = 0
    while time.monotonic() < deadline:
        try:
            r = await client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            if r.status_code in (502, 503, 504, 429):
                if poll_error_retries < 5:
                    poll_error_retries += 1
                    await asyncio.sleep(float(1))
                    logger.info(
                        f"WARNING: transient error {r.status_code} when polling job status, retrying..."
                    )
                    continue
        except Exception as e:
            error_msg = f"Failed to poll job status: {type(e).__name__}: {str(e)}"
            logger.info(f"ERROR: {error_msg}")
            return 503, {
                "error": "poll_failed",
                "detail": error_msg,
                "job_id": job_id,
                "type": type(e).__name__,
            }
        data = safe_json(r)
        status = data.get("status")
        logger.info(f"job_id:{job_id}, status({time.monotonic()}):{status}")
        await asyncio.sleep(poll_interval)
        if status in wait_status:
            continue
        else:
            return 200, data

    return 504, {"error": "wait_timeout exceeded", "job_id": job_id}


def is_valid_python_script(script):
    """Check if a script is a valid Python script."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def format_code(code) -> str:
    """Format Python code using Black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore
        return code


def extract_code(text):
    """Extract python code blocks from the text."""
    # logger.info(f"raw text: {text}")
    parsed_codes = []

    # When code is in a text or python block
    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for match in matches:
        code_block = match[1]
        parsed_codes.append(code_block)

    # When the entire text is code or backticks of the code block is missing
    if len(parsed_codes) == 0:
        matches = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if matches:
            code_block = matches[0][2]
            parsed_codes.append(code_block)

    # validate the parsed codes
    valid_code_blocks = [
        format_code(c) for c in parsed_codes if is_valid_python_script(c)
    ]
    # logger.info(f"valid_code_blocks: {valid_code_blocks}")
    return format_code("\n\n".join(valid_code_blocks))


_RECENT = {}
_RECENT_MAXLEN = 20


def _stable_sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _signed(score: float, meta: dict) -> float:
    higher = meta.get("higher_is_better", True)
    s = float(score)
    return s if higher else -s


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _bounds_signed(meta: dict):
    higher = meta.get("higher_is_better", True)

    def to_signed(v):
        if not _finite(v):
            return None
        v = float(v)
        return v if higher else -v

    tmin = to_signed(meta.get("theoretical_min"))
    tmax = to_signed(meta.get("theoretical_max"))
    lmin = to_signed(meta.get("leaderboard_min"))
    lmax = to_signed(meta.get("leaderboard_max"))

    cands = [tmin, tmax, lmin, lmax]
    cands = [x for x in cands if x is not None]
    if len(cands) < 2:
        return None, None

    best = max(cands)
    worst = min(cands)

    MAX_RANGE = 1e6
    if best - worst > MAX_RANGE:
        if lmax is not None and lmin is not None and (lmax - lmin) <= MAX_RANGE:
            best, worst = max(lmax, lmin), min(lmax, lmin)
        else:
            worst = best - MAX_RANGE

    return best, worst


def score2reward(score, metadata, mode="logistic"):
    """Map a metric score to a normalized reward."""
    higher_is_better = metadata.get("higher_is_better")
    theoretical_min = metadata.get("theoretical_min")
    theoretical_max = metadata.get("theoretical_max")
    leaderboard_min = metadata.get("leaderboard_min")
    leaderboard_max = metadata.get("leaderboard_max")

    def safe_max(*vals):
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def safe_min(*vals):
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    # assert mode=="linear_sign" , f"Unsupported mode: {mode}"

    if mode == "linear_sign":
        if abs(float(score)) < 1e-7:
            return -100.0

        if higher_is_better is False:
            return -float(score)
        else:
            return float(score)

    s = _signed(score, metadata)

    # A: Power sigmoid on min-max bounds.
    if mode == "power_sigmoid":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return _stable_sigmoid(s)

        rng = max(best - worst, 1e-9)
        # p in [0,1]
        p = (s - worst) / rng
        p = 0.0 if p < 0 else (1.0 if p > 1 else p)

        T = 0.50
        x = (p - 0.5) / max(T, 1e-9)
        r = _stable_sigmoid(x)  # (0,1)

        alpha = 2.0
        return r**alpha

    # B: Tanh margin shaping.
    if mode == "margin_tanh":
        best, worst = _bounds_signed(metadata)
        if best is None:
            return 0.5 + 0.5 * math.tanh(s)  # Neutral fallback.

        center = 0.5 * (best + worst)
        scale = max(0.5 * (best - worst), 1e-9)

        m = (s - center) / scale

        T = 0.50
        y = math.tanh(m / max(T, 1e-9))  # (-1,1)
        r = 0.5 + 0.5 * y  # (0,1)

        gamma = 2.0
        return r**gamma

    # C: Online percentile.
    if mode == "online_percentile":
        key = str(metadata.get("data_dir", "unknown_task"))
        dq = _RECENT.get(key)
        if dq is None:
            from collections import deque

            dq = deque(maxlen=_RECENT_MAXLEN)
            _RECENT[key] = dq

        if len(dq) < 10:
            dq.append(s)
            return _stable_sigmoid(s)

        sorted_vals = sorted(dq)
        # rank = count(vals <= s)
        import bisect

        rank = bisect.bisect_right(sorted_vals, s)
        p = rank / len(sorted_vals)  # [0,1]

        dq.append(s)

        gamma = 2.0
        return p**gamma

    # initial sigmoid reward
    if higher_is_better:
        # best score
        score_range_best_score = (
            theoretical_max if theoretical_max is not None else leaderboard_max
        )

        # worst score
        score_range_worst_score = safe_max(leaderboard_min, theoretical_min, -100)

    else:
        # best score
        score_range_best_score = (
            theoretical_min if theoretical_min is not None else leaderboard_min
        )

        # worst score
        score_range_worst_score = safe_min(leaderboard_max, theoretical_max, 100)

    # Return a neutral reward when the range is still unknown.
    if score_range_best_score is None or score_range_worst_score is None:
        return 0.5

    score_range = score_range_best_score - score_range_worst_score
    if score_range == 0 or math.isnan(score_range):
        return 0.5

    x = 2.0 * (score - score_range_worst_score) / score_range
    if x >= 0:
        reward = 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        reward = exp_x / (1.0 + exp_x)

    return reward


def test_score2reward(base_dir: str):
    """
    Test score2reward function for all tasks in Selected_Dojo.

    For each task, displays score and reward at:
    - Rank 1 (best)
    - 20th percentile
    - 40th percentile
    - 60th percentile
    - 80th percentile
    - Last rank (worst)

    Args:
        base_dir: Base directory containing task folders
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    # Find all task directories
    task_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            metadata_path = item / "info" / "task_metadata.json"
            leaderboard_path = item / "info" / "public_leaderboard.csv"
            if metadata_path.exists() and leaderboard_path.exists():
                task_dirs.append(item)

    if not task_dirs:
        print(f"No valid tasks found in {base_dir}")
        return

    print(f"Found {len(task_dirs)} tasks to test\n")
    print("=" * 100)

    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        metadata_path = task_dir / "info" / "task_metadata.json"
        leaderboard_path = task_dir / "info" / "public_leaderboard.csv"

        # Load metadata
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {task_name}: {e}")
            continue

        # Load leaderboard
        try:
            df = pd.read_csv(leaderboard_path)
            if "score" not in df.columns:
                print(f"Warning: {task_name} leaderboard does not have 'score' column")
                continue
            scores = df["score"].dropna().astype(float).tolist()
            if not scores:
                print(f"Warning: {task_name} leaderboard has no valid scores")
                continue
        except Exception as e:
            print(f"Error loading leaderboard for {task_name}: {e}")
            continue

        # Sort scores based on higher_is_better
        higher_is_better = metadata.get("higher_is_better", True)
        if higher_is_better:
            scores_sorted = sorted(scores, reverse=True)  # Best to worst
        else:
            scores_sorted = sorted(scores)  # Best (lowest) to worst (highest)

        total_entries = len(scores_sorted)
        if total_entries == 0:
            continue

        # Calculate indices for percentiles
        # Use floor to get the index, ensuring we get the entry at or below the percentile
        def percentile_index(pct: float) -> int:
            """Calculate index for a given percentile (0.0 to 1.0)."""
            idx = int((total_entries - 1) * pct)
            return max(0, min(idx, total_entries - 1))

        indices = [
            0,  # Rank 1 (best)
            percentile_index(0.2),  # 20%
            percentile_index(0.4),  # 40%
            percentile_index(0.6),  # 60%
            percentile_index(0.8),  # 80%
            total_entries - 1,  # Last (worst)
        ]

        # Get scores at these positions
        selected_scores = [scores_sorted[i] for i in indices]
        selected_ranks = [i + 1 for i in indices]  # Rank is 1-indexed

        # Calculate rewards
        rewards = []
        for score in selected_scores:
            try:
                reward = score2reward(score, metadata)
                rewards.append(reward)
            except Exception as e:
                print(
                    f"  Warning: Failed to calculate reward for score {score:.6f}: {e}"
                )
                rewards.append(float("nan"))

        # Display results
        print(f"\nTask: {task_name}")
        print(f"  UUID: {metadata.get('uuid', 'N/A')}")
        print(f"  Higher is better: {higher_is_better}")
        print(f"  Theoretical min: {metadata.get('theoretical_min', 'N/A')}")
        print(f"  Theoretical max: {metadata.get('theoretical_max', 'N/A')}")
        print(f"  Total entries: {total_entries}")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"\n  {'Position':<12} {'Rank':<8} {'Score':<15} {'Reward':<10}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 15} {'-' * 10}")

        labels = ["1st (best)", "20%", "40%", "60%", "80%", "Last (worst)"]
        for label, rank, score, reward in zip(
            labels, selected_ranks, selected_scores, rewards
        ):
            if math.isnan(reward):
                reward_str = "N/A"
            else:
                reward_str = f"{reward:.6f}"
            print(f"  {label:<12} {rank:<8} {score:<15.6f} {reward_str:<10}")

        print("=" * 100)


def get_clear_log(run_log: str | None) -> str:
    """Extract wrapped sandbox output and remove heartbeat log blocks."""
    if not run_log:
        return ""

    markers = [
        ("--- OUTPUT START ---", "--- OUTPUT END ---"),
        ("--- SANDBOX STDOUT START ---", "--- SANDBOX STDOUT END ---"),
    ]

    hb_marker = "[HB]"
    hb_len = len(hb_marker)

    sections: list[str] = []

    for start_marker, end_marker in markers:
        search_pos = 0
        while True:
            start_idx = run_log.find(start_marker, search_pos)
            if start_idx == -1:
                break

            content_start = start_idx + len(start_marker)
            if content_start < len(run_log) and run_log[content_start] == "\n":
                content_start += 1

            end_idx = run_log.find(end_marker, content_start)
            if end_idx == -1:
                content = run_log[content_start:].strip()
            else:
                content = run_log[content_start:end_idx].strip()

            # Remove every [HB] ... [HB] block without a regular expression.
            if content:
                parts: list[str] = []
                pos = 0
                while True:
                    s = content.find(hb_marker, pos)
                    if s == -1:
                        parts.append(content[pos:])
                        break
                    parts.append(content[pos:s])
                    e = content.find(hb_marker, s + hb_len)
                    if e == -1:
                        # Discard an unterminated heartbeat block.
                        break
                    pos = e + hb_len

                content = "".join(parts).strip()

            if content:
                sections.append(f"\n{content}")

            if end_idx == -1:
                break
            search_pos = end_idx + len(end_marker)

    return "\n\n".join(sections)


def format_sandbox_feedback(status_code: int, payload: dict) -> str:
    """Format sandbox execution results into feedback message"""
    if status_code == 200:
        result = payload.get("result") or {}
        status = payload.get("status", "unknown")
        run_log = get_clear_log(result.get("run_log"))
        run_result = result.get("result", "")
        score = result.get("score")

        feedback = "## Execution Result\n"
        feedback += f"**Status**: {status}\n\n"

        if score is not None:
            feedback += f"**Score**: {score}\n\n"

        if run_result:
            feedback += f"**Result**: {run_result}\n\n"

        if run_log:
            # Truncate log if too long
            if len(run_log) > 2000:
                run_log = run_log[:1000] + "\n... (truncated) ...\n" + run_log[-1000:]
            feedback += f"**Execution Log**:\n```\n{run_log}\n```\n\n"

    elif status_code == 503:
        error_type = payload.get("type", "unknown")
        error_detail = payload.get("detail", "Connection failed")
        feedback = "## Connection Error\n"
        feedback += f"**Type**: {error_type}\n"
        feedback += f"**Detail**: {error_detail}\n\n"
        feedback += "Please check your code and try again."
    else:
        error_msg = payload.get("error", "unknown error")
        detail = payload.get("detail", {})
        feedback = "## Execution Error\n"
        feedback += f"**Error**: {error_msg}\n"
        if isinstance(detail, dict):
            if detail.get("status"):
                feedback += f"**Status**: {detail.get('status')}\n"
            if detail.get("message"):
                feedback += f"**Message**: {detail.get('message')}\n"
        feedback += "\nPlease fix the errors and try again."

    return feedback


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect score-to-reward mapping for MLE tasks"
    )
    parser.add_argument(
        "base_dir", help="Directory containing task folders and leaderboard files"
    )
    test_score2reward(parser.parse_args().base_dir)
